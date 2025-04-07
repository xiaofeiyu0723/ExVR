import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np

from mobileposer.config import *
from mobileposer.utils.model_utils import reduced_pose_to_full
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN


class Poser(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations).
    """
    def __init__(self, finetune: bool=False):
        super().__init__()
        
        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = RNN(self.C.n_output_joints*3 + self.C.n_imu, joint_set.n_reduced*6, 256) # pose estimation model

        # loss function
        self.loss = nn.MSELoss()
        self.t_weight = 1e-5 
        self.use_pos_loss = True

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = Poser.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        for ignore in joint_set.ignored: pred_pose[:, ignore] = torch.eye(3, device=self.C.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(B, S, -1)

        # generate noise for target joints for beter robustness
        noise = torch.randn(target_joints.size()).to(self.C.device) * 0.04 # gaussian noise with std = 0.04
        noisy_joints = target_joints + noise

        # predict pose
        pose_input = torch.cat((noisy_joints, imu_inputs), dim=-1)
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].view(B, S, -1)
        loss = self.loss(pose_p, pose_t)
        loss += self.t_weight*self.compute_jerk_loss(pose_p)

        # joint position loss
        if self.use_pos_loss:
            full_pose_p = self._reduced_global_to_full(pose_p)
            joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[1].view(B, S, -1)
            loss += self.loss(joints_p, target_joints)

        return loss

    def compute_jerk_loss(self, pred_pose):
        jerk = pred_pose[:, 3:, :] - 3*pred_pose[:, 2:-1, :] + 3*pred_pose[:, 1:-2, :] - pred_pose[:, :-3, :]
        l1_norm = torch.norm(jerk, p=1, dim=2)
        return l1_norm.sum(dim=1).mean()

    def compute_temporal_loss(self, pred_pose):
        acc = pred_pose[:, 2:, :] + pred_pose[:, :-2, :] - 2*pred_pose[:, 1:-1, :]
        l1_norm = torch.norm(acc, p=1, dim=2)
        return l1_norm.sum(dim=1).mean() 

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        # log average loss
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)
        # log learning late
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 
        return optimizer
