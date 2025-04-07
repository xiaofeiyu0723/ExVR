import numpy as np
from pathlib import Path
from pathlib import Path
import smplx

import torch

root_dir = Path().absolute()
model_path = root_dir/"model/SMPL_NEUTRAL.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpl_model = smplx.create(model_path, model_type='smpl')
transl = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
def GetPosition(rotmats):
    # 拆分根节点和其他关节
    global_orient = rotmats[0:1].unsqueeze(0)
    body_pose = rotmats[1:].unsqueeze(0)
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl
    )

    # 提取左右手位置
    joint_positions = output.joints.detach().cpu().numpy()
    left_wrist = joint_positions[0, 21]
    right_wrist = joint_positions[0, 20]
    left_wrist[1]-=0.2
    right_wrist[1]-=0.2
    return left_wrist, right_wrist

def compute_hand_positions_from_rotmat(rot_mats, joint_offsets, joint_parents):
    """
    从旋转矩阵计算左右手腕的全局位置
    :param rot_mats: 24个关节的3x3局部旋转矩阵（列表或数组）
    :param joint_offsets: 每个关节的偏移量（24x3数组）
    :param joint_parents: 每个关节的父节点索引（长度为24的列表）
    :return: left_wrist_pos, right_wrist_pos （3x1）
    """
    num_joints = len(rot_mats)
    global_transforms = [np.eye(4) for _ in range(num_joints)]

    #TODO 添加平移
    global_transforms[0][:3, :3] = rot_mats[0]

    # 逐关节计算全局变换
    for j in range(1, num_joints):
        parent_idx = joint_parents[j]
        parent_transform = global_transforms[parent_idx]

        # 局部变换矩阵（旋转 + 偏移）
        local_transform = np.eye(4)
        local_transform[:3, :3] = rot_mats[j]  # 应用旋转
        local_transform[:3, 3] = joint_offsets[j]  # 应用偏移

        # 全局变换 = 父变换 × 局部变换
        global_transforms[j] = parent_transform @ local_transform

    # 提取左右手腕位置
    left_wrist_pos = global_transforms[20][:3, 3]
    right_wrist_pos = global_transforms[21][:3, 3]

    return left_wrist_pos, right_wrist_pos