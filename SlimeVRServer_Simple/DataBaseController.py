import threading
import time
from pathlib import Path

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R
from SlimeVRServer_Simple.SLMPTransverter import GetPosition
from mobileposer.articulate.math import quaternion_to_rotation_matrix, rotation_matrix_to_axis_angle
from mobileposer.utils.model_utils import load_model
import utils.globals as g

rotateOffset=(0, -90, 0)
class DataBaseController(threading.Thread):
    def __init__(self, device_handlers):
        self.lock = threading.Lock()
        super().__init__()
        self.acc_IMUs = None
        self.raw_IMUs = None
        self.Rot_L = []
        self.Rot_R = []
        self.devices = device_handlers
        self.stopped = threading.Event()
        self.interval = 1 / 120
        root_dir = Path().absolute()
        path = root_dir / "model/weights.pth"
        self.model = load_model(path)
        self.model.eval()

    def run(self):

        next_execution = time.perf_counter()  # 使用高精度计时器
        while not self.stopped.is_set():
            # 执行你的任务逻辑
            self.task()

            # 计算并补偿时间误差
            next_execution += self.interval
            sleep_time = next_execution - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 处理超时情况（任务执行时间超过间隔）
                pass  # 可在此记录日志或调整策略

    def task(self):
        flag = True
        for device in self.devices.values():
            if device and not device.check_is_full():
                flag = False
        if flag:
            acc = []
            raw = []
            self.acc_IMUs = None
            self.raw_IMUs = None
            for device in self.devices.values():
                if device:
                    acc.extend(device.acc)
                    raw.extend(device.raw)
                else:
                    acc.extend([0, 0, 0])
                    raw.extend([1, 0, 0, 0])
            self.Rot_L = [raw[0], raw[1], raw[2], raw[3]]
            self.Rot_R = [raw[4], raw[5], raw[6], raw[7]]
            acc = self.expand_list(acc, 15)
            raw = self.pad_quaternions(raw, 20)
            # print("acc:",acc)
            # print("raw:",raw)
            q = torch.from_numpy(np.array(raw)).float()
            q = quaternion_to_rotation_matrix(q).view(-1, 5, 3, 3)
            a = torch.from_numpy(np.array(acc)).float()

            _acc = a.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]]
            _ori = q.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
            acc = torch.zeros_like(_acc)
            ori = torch.zeros_like(_ori)
            acc[:, g.combo] = _acc[:, g.combo]
            ori[:, g.combo] = _ori[:, g.combo]
            self.acc_IMUs = _acc
            self.raw_IMUs = _ori
        if self.acc_IMUs is None or self.raw_IMUs is None:
            return
        # USEMODEL
        # normalization
        imu_input = torch.cat([self.acc_IMUs.flatten(1), self.raw_IMUs.flatten(1)], dim=1)
        # imu_input = torch.cat([_acc[:, c].flatten(1), _ori[:, c].flatten(1)], dim=1)
        # predict pose and translation
        with torch.no_grad():
            output = self.model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
            pred_pose = output[0]  # [24, 3, 3]
            pred_tran = output[2]  # [3]
        # convert rotmatrix to axis angle

        pose = rotation_matrix_to_axis_angle(pred_pose)

        Pos_L, Pos_R = GetPosition(pose)
        print(Pos_L, Pos_R, self.Rot_L, self.Rot_R)
        self.Save2Data(Pos_L, Pos_R, self.Rot_L, self.Rot_R)

    def expand_list(self, lst, target_length):
        lst.extend([0] * (target_length - len(lst)))  # 用0填充到目标长度
        return lst

    def pad_quaternions(self, input_data, target_length=20):
        # 计算需要填充的四元数数量
        num_input_quats = len(input_data) // 4
        num_pad_quats = (target_length // 4) - num_input_quats

        # 生成填充数据（单位四元数）
        padding = [1.0, 0.0, 0.0, 0.0] * num_pad_quats

        # 合并数据
        padded_data = input_data + padding
        return padded_data

    def Save2Data(self, Pos_L, Pos_R, Rot_L, Rot_R):
        Pos = [Pos_L, Pos_R]
        Rot = [Rot_L, Rot_R]
        if g.config["Smoothing"]["enable"]:
            for i in g.combo:
                print(i)
                g.latest_data[117+i*3]=Pos[i][0]
                g.latest_data[118+i*3]=Pos[i][1]
                g.latest_data[119+i*3]=Pos[i][2]
                g.data["SlimeRotation"+str(i)][0]['v'] = Rot[i][0]
                g.data["SlimeRotation"+str(i)][1]['v'] = Rot[i][1]
                g.data["SlimeRotation"+str(i)][2]['v'] = Rot[i][2]
                g.data["SlimeRotation"+str(i)][3]['v'] = Rot[i][3]
        else:
            for i in g.combo:
                g.data["SlimePosition"+str(i)][0]['v'] = Pos[i][0]
                g.data["SlimePosition"+str(i)][1]['v'] = Pos[i][1]
                g.data["SlimePosition"+str(i)][2]['v'] = Pos[i][2]
                g.data["SlimeRotation"+str(i)][0]['v'] = Rot[i][0]
                g.data["SlimeRotation"+str(i)][1]['v'] = Rot[i][1]
                g.data["SlimeRotation"+str(i)][2]['v'] = Rot[i][2]
                g.data["SlimeRotation"+str(i)][3]['v'] = Rot[i][3]
