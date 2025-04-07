import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import StepLR 

from mobileposer.articulate.model import ParametricModel
from mobileposer.models.rnn import RNN
from mobileposer.config import *


class Velocity(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Per-Frame Root Velocity. 
    """

    def __init__(self):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers
        self.bodymodel = ParametricModel(paths.smpl_file, device=self.C.device)

        # model definitions
        self.vel = RNN(self.C.n_output_joints * 3 + self.C.n_imu, 24 * 3, 256, bidirectional=False)  # per-frame velocity of the root joint. 
        self.rnn_state = None

        # loss function 
        self.loss = nn.MSELoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def forward(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, _ = self.vel(batch, input_lengths)
        return vel

    def forward_online(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, self.rnn_state = self.vel(batch, input_lengths, self.rnn_state)
        return vel

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)

        # target velocity
        target_vel = outputs['vels'].view(joints.shape[0], joints.shape[1], 72)

        # add noise to target joints for beter robustness
        noise = torch.randn(target_joints.size()).to(self.C.device) * 0.025 # gaussian noise with std = 0.025
        target_joints += noise
        
        # predict root joint velocity
        tran_input = torch.cat((target_joints, imu_inputs), dim=-1)
        pred_vel, _, _ = self.vel(tran_input, input_lengths)
        loss = self.compute_loss(pred_vel, target_vel)

        return loss

    def compute_loss(self, pred_vel, gt_vel):
        loss = sum(self.compute_vel_loss(pred_vel, gt_vel, i) for i in [1, 3, 9])
        return loss

    def compute_vel_loss(self, pred_vel, gt_vel, n=1):
        T = pred_vel.shape[1]
        loss = 0.0

        for m in range(0, T//n):
            end = min(n*m+n, T)
            loss += self.loss(pred_vel[:, m*n:end, :], gt_vel[:, m*n:end, :])

        return loss

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
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 