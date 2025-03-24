import queue
import threading
from scipy.spatial.transform import Rotation as R
import numpy as np
import utils.globals as g


class Kalman3D:
    """三轴协同卡尔曼滤波器"""

    def __init__(self, init_acc, init_cov=np.eye(3), Q=np.eye(3) * 0.01, R=np.eye(3) * 0.1):
        self.x = init_acc  # 状态向量 [ax, ay, az]
        self.P = init_cov  # 协方差矩阵
        self.Q = Q  # 过程噪声
        self.R = R  # 测量噪声
        self.F = np.eye(3)  # 状态转移矩阵
        self.H = np.eye(3)  # 观测矩阵

    def update(self, z):
        # 预测阶段
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新阶段
        y = z - self.H @ x_pred  # 残差
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益

        # 状态修正
        self.x = x_pred + K @ y
        self.P = (np.eye(3) - K @ self.H) @ P_pred
        return self.x.copy()


class AccelerometerProcessor:
    def __init__(self, type, calibration_samples=100):
        self.quaternion_data = None
        self.accel_data = None
        self.type = type
        self.gravity = np.array([0, 0, 0])  # 标准重力

        self.accel_queue = queue.Queue(maxsize=100)
        self.quaternion_queue = queue.Queue(maxsize=100)

        self.kf = None
        self.last_timestamp = None

        self.calibration_samples = calibration_samples
        self.static_error = None
        self.dynamic_R = None

        self.process_thread = threading.Thread(target=self._processing_loop)
        self.running = True
        self.process_thread.start()
        self.lock = threading.Lock()

    def _calibrate_sensor(self):
        static_samples = []
        while len(static_samples) < self.calibration_samples:
            try:
                accel = self.accel_queue.get(timeout=0.5)

                static_samples.append(accel)
            except queue.Empty:
                continue

        self.static_error = np.mean(static_samples, axis=0)
        self.dynamic_R = np.cov(static_samples, rowvar=False) + np.eye(3) * 1e-6  # 正则化

        init_cov = np.diag([0.1, 0.1, 0.1])
        Q = np.diag([0.01, 0.01, 0.005])
        self.kf = Kalman3D(
            init_acc=self.gravity,
            init_cov=init_cov,
            Q=Q,
            R=self.dynamic_R
        )

    def _processing_loop(self):
        """ Main processing loop """
        self._calibrate_sensor()
        while self.running:
            try:
                accel_data = self.accel_queue.get(timeout=0.5)
                quaternion_data = self.quaternion_queue.get(timeout=0.5)
                calibrated_accel = accel_data
                # Kalman filtering
                filtered_accel = self.kf.update(calibrated_accel)
                # Update global status
                self._update_global_state(filtered_accel, quaternion_data)

            except queue.Empty:
                continue

    def _update_global_state(self, accel, quaternion):
        g.data[f"{self.type}Rotation"][0]["v"] = quaternion[0]
        g.data[f"{self.type}Rotation"][1]["v"] = quaternion[1]
        g.data[f"{self.type}Rotation"][2]["v"] = quaternion[2]
        g.data[f"{self.type}Rotation"][3]["v"] = quaternion[3]
        dt = 0.1

        for i in range(3):
            current_v = g.data[f"{self.type}Velocity"][i]["v"]
            new_v = current_v + accel[i] * dt
            g.data[f"{self.type}Velocity"][i]["v"] = np.clip(new_v, -2, 2)

    def add_acceleration_data(self, acceleration):
        with self.lock:
            self.accel_data = acceleration
        self.add_data()

    def add_quaternion_data(self, quaternion_angles):
        with self.lock:
            self.quaternion_data = quaternion_angles
        self.add_data()

    def add_data(self):
        if self.quaternion_data is not None and self.accel_data is not None:
            self.accel_queue.put(self.accel_data)
            self.quaternion_queue.put(self.quaternion_data)
            self.accel_data = None
            self.quaternion_data = None

    def stop(self):
        self.running = False
        self.process_thread.join()
