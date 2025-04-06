import os
import numpy as np
import torch

from mobileposer.config import *
from mobileposer.utils.model_utils import *


class DataLoader:
    def __init__(self, dataset: str='dip', combo: str='lw_rp', device: str='cpu'):
        self.dataset = dataset
        self.combo = combo
        self.device = device

        if combo not in amass.combos:
            raise ValueError(f"Invalid Combo: {combo}")
        self.combo = amass.combos[combo]

    def _get_sequence_data(self, file_path,seq_num):
        data = torch.load(file_path)
        if seq_num >= len(data['acc']):
            raise ValueError(f"Provided seq. # ({seq_num}) is greater than seq. length")
        return data['acc'][seq_num], data['ori'][seq_num], data['pose'][seq_num], data['tran'][seq_num]

    def _get_sequence(self, seq_num):
        """Get sequence data for a given sequence number."""
        if self.dataset in datasets.amass_datasets:
            file_path = os.path.join(paths.processed_datasets, f'{self.dataset}.pt')
        elif self.dataset == 'totalcapture':
            file_path = os.path.join(paths.processed_datasets, 'eval', 'totalcapture.pt')
        elif self.dataset == 'dip':
            file_path = os.path.join(paths.processed_datasets, 'eval', 'dip_test.pt')
        elif self.dataset == 'imuposer':
            file_path = os.path.join(paths.processed_datasets, 'eval', 'imuposer_test.pt')
        elif self.dataset == 'dev':
            file_path = os.path.join(paths.dev_data, f'dev.pt')
        return self._get_sequence_data(file_path, seq_num)

    def _get_imu(self, glb_acc, glb_ori):
        """Process IMU data."""
        acc = torch.zeros_like(glb_acc)
        ori = torch.zeros_like(glb_ori)
        acc[:, self.combo] = glb_acc[:, self.combo] / amass.acc_scale
        ori[:, self.combo] = glb_ori[:, self.combo]
        acc = acc[:, amass.all_imu_ids]
        ori = ori[:, amass.all_imu_ids]
        acc = smooth_avg(acc)
        data = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
        return data, acc, ori

    def load_data(self, seq_num):
        """Load data for a given sequence number."""
        acc, ori, pose, tran = self._get_sequence(seq_num)
        data, acc, ori = self._get_imu(acc, ori)

        data = data.to(self.device)
        acc = acc.to(self.device)
        ori = ori.to(self.device)
        pose = pose.to(self.device)
        tran = tran.to(self.device)

        return {
            'imu': data,
            'acc': acc,
            'ori': ori,
            'pose': pose,
            'tran': tran    
        }
