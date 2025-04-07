import torch
import numpy as np

from mobileposer.helpers import *
from mobileposer.config import *
from mobileposer.constants import NUM_VERTICES
import mobileposer.articulate as art


class SMPLViewer:
    def __init__(self, fps: int=25):
        self.fps = fps
        self.colors = None
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=model_config.device)

    def _assign_colors(self, seq_length):
        v = NUM_VERTICES
        colors = np.zeros((seq_length, NUM_VERTICES*2, 3))
        colors[:, :v] = np.array([0.7, 0.65, 0.65])  # tinted-red 
        colors[:, v:] = np.array([0.65, 0.65, 0.65]) # gray
        return colors

    def view(self, pose_p, tran_p, pose_t, tran_t, with_tran: bool=False): 
        if not with_tran:
            # set translation to None
            tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device) 
            tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device) 

        pose_p, tran_p = pose_p.view(-1, 24, 3, 3), tran_p.view(-1, 3)
        pose_t, tran_t = pose_t.view(-1, 24, 3, 3), tran_t.view(-1, 3)

        poses, trans = [pose_p], [tran_p]
        if getenv("GT") == 1: 
            # visualize prediction and ground-truth
            poses.append(pose_t)
            trans.append(tran_t)
            self.colors = self._assign_colors(len(pose_p)) # assign colors to distinguish prediction and ground-truth
        elif getenv("GT") == 2:
            # visualize ground truth only
            poses = [pose_t]
            trans = [tran_t]
        
        self.bodymodel.view_motion(poses, trans, fps=self.fps, colors=self.colors, distance_between_subjects=0)
