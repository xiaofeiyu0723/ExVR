import torch

from mobileposer.config import *


def load_model(model_path: str):
    """Load MobilePoser model."""
    from mobileposer.models import MobilePoserNet
    device = model_config.device
    try: 
        model = MobilePoserNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        model = MobilePoserNet.load_from_checkpoint(model_path)
    return model


def reduced_pose_to_full(reduced_pose):
    """Transform reduced pose to full pose."""
    B, S = reduced_pose.shape[0], reduced_pose.shape[1]
    reduced_pose = reduced_pose.view(B, S, joint_set.n_reduced, 3, 3)
    full_pose = torch.eye(3, device=reduced_pose.device).repeat(B, S, 24, 1, 1)
    full_pose[:, :, joint_set.reduced] = reduced_pose
    full_pose = full_pose.view(B, S, -1)
    return full_pose


def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L-(s-i-1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed


def normalize_and_concat(glb_acc, glb_ori):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_ori = glb_ori.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / amass.acc_scale
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data
