import time
import numpy as np
import utils.data as data_utils
from utils.json_manager import load_json
import utils.globals as g

class VectorKalmanFilter:
    def __init__(self, q_process: float, r_measure: float, dim: int):
        self.dim = dim
        self.Q   = np.eye(dim) * q_process
        self.R   = np.eye(dim) * r_measure
        self.x   = None
        self.P   = np.eye(dim)

    def predict(self, dt: float = 1.0):
        if self.x is None:
            return
        self.P = self.P + self.Q * dt

    def update(self, z, is_rotation: bool = False):
        z = np.asarray(z, dtype=np.float64)
        if self.x is None:
            self.x = z.copy()
            return self.x.copy()
        if is_rotation:
            y = np.vectorize(angle_diff)(z, self.x)
        else:
            y = z - self.x

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K) @ self.P
        return self.x.copy()

def angle_diff(current: float, target: float) -> float:
    diff = (current - target) % 360
    return diff - 360 if diff > 180 else diff


def update_target_value(target: dict, smoothed_delta: float, is_rotation: bool):
    if is_rotation:
        target["v"] = (target["v"] + smoothed_delta) % 360
    else:
        target["v"] += smoothed_delta

def setup_smoothing():
    smoothing_config = load_json("./settings/smoothing.json")

    g.kalman_filters = {}   # action → VectorKalmanFilter
    g.indices_map    = {}   # action → indices 列表
    g.smoothing_config = smoothing_config

    for action, params in smoothing_config["Parameters"].items():
        indices = params.get("indices", [])
        g.indices_map[action] = indices

        if "kalman_params" in params and indices:
            q = params["kalman_params"]["q_process"]
            r = params["kalman_params"]["r_measure"]
            dim = len(indices)
            g.kalman_filters[action] = VectorKalmanFilter(q, r, dim)

    return smoothing_config


def apply_smoothing():
    last_time = time.perf_counter()
    frame_duration = 1.0 / 1000.0

    while not g.stop_event.is_set() and g.config["Smoothing"]["enable"]:
        now       = time.perf_counter()
        dt_base   = now - last_time

        for action, params in g.smoothing_config["Parameters"].items():
            indices = g.indices_map.get(action, [])
            if not indices:
                continue

            target_key  = params["key"]
            shifting    = params.get("shifting", 0)
            is_rotation = params.get("is_rotation", False)
            dt_mul      = params.get("dt_multiplier", 20)
            dt          = dt_base * dt_mul

            try:
                obs_vec = [g.latest_data[idx] for idx in indices]
            except IndexError:
                continue

            if action in g.kalman_filters:
                kf = g.kalman_filters[action]
                kf.predict(dt_base)
                filt_vec = kf.update(obs_vec, is_rotation)
            else:
                filt_vec = obs_vec

            for local_i, idx in enumerate(indices):
                target_idx = idx - shifting
                data_array = g.data[target_key]
                if not (0 <= target_idx < len(data_array)):
                    continue

                target = data_array[target_idx]
                raw    = filt_vec[local_i]
                delta  = angle_diff(raw, target["v"]) if is_rotation else raw - target["v"]
                sm_delta = delta * dt
                update_target_value(target, sm_delta, is_rotation)

        last_time = now
        time.sleep(frame_duration)
