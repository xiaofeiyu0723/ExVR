import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
import numpy as np

from mobileposer.config import *
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN


class FootContact(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Foot Contact Probability ([s_lfoot, s_rfoot]).
    """

    def __init__(self):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers

        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.footcontact = RNN(self.C.n_output_joints * 3 + self.C.n_imu, 2, 64)  # foot-ground probability model

        # loss function (binary cross-entropy)
        self.loss = nn.BCEWithLogitsLoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def forward(self, batch, input_lengths=None):
        # forward foot contact model
        foot_contact, _, _ = self.footcontact(batch, input_lengths)
        return foot_contact

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)

        # ground-truth foot contacts
        foot_contacts = outputs['foot_contacts']

        # add noise to target joints for beter robustness
        noise = torch.randn(target_joints.size()).to(self.C.device) * 0.04 # gaussian noise with std = 0.04
        target_joints += noise
        
        # predict foot-ground contact probability
        tran_input = torch.cat((target_joints, imu_inputs), dim=-1)
        pred_contacts, _, _ = self.footcontact(tran_input, input_lengths)
        loss = self.loss(pred_contacts, foot_contacts)

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