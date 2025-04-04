import time
import numpy as np
import utils.data as data_utils
from utils.json_manager import load_json
import utils.globals as g


class SimpleKalmanFilter:
    def __init__(self, q_process, r_measure):
        self.q = q_process
        self.r = r_measure
        self.x = None
        self.p = 1.0

    def predict(self, dt=1.0):
        if self.x is None:
            return
        self.p += self.q * dt

    def update(self, z, is_rotation=False):
        if self.x is None:
            self.x = z
            self.p = self.r
            return self.x

        if is_rotation:
            # Use angle_diff to compute the difference for rotation
            delta = angle_diff(z, self.x)
        else:
            delta = z - self.x

        k = self.p / (self.p + self.r)
        self.x += k * delta
        self.p *= (1 - k)
        return self.x


def setup_smoothing():
    smoothing_config = load_json("./settings/smoothing.json")
    g.kalman_filters = {}
    for action, params in smoothing_config["Parameters"].items():
        if "kalman_params" in params:
            indices = params.get("indices", [])
            q = params["kalman_params"]["q_process"]
            r = params["kalman_params"]["r_measure"]
            for idx in indices:
                g.kalman_filters[idx] = SimpleKalmanFilter(q, r)
    return smoothing_config


def update_target_value(target, smoothed_delta, is_rotation):
    if is_rotation:
        target["v"] = (target["v"] + smoothed_delta) % 360
    else:
        target["v"] += smoothed_delta

def angle_diff(current, target):
    diff = (current - target) % 360
    return diff - 360 if diff > 180 else diff

def apply_smoothing():
    last_time = time.time()
    frame_duration = 1.0 / 1000.0
    index_to_action = {
        idx: action
        for action, params in g.smoothing_config["Parameters"].items()
        for idx in params.get("indices", [])
    }

    while not g.stop_event.is_set() and g.config["Smoothing"]["enable"]:
        current_time = time.time()
        dt_base = current_time - last_time

        for i, current_data in enumerate(g.latest_data):
            action = index_to_action.get(i, "OtherBlendShapes")
            config_params = g.smoothing_config["Parameters"][action]

            target_key = config_params["key"]
            shifting = config_params.get("shifting", 0)
            is_rotation = config_params.get("is_rotation", False)
            dt = dt_base * config_params.get("dt_multiplier", 20)

            if i in g.kalman_filters:
                kf = g.kalman_filters[i]
                kf.predict(dt_base)
                filtered_data = kf.update(current_data, is_rotation)
            else:
                filtered_data = current_data

            target = g.data[target_key][i - shifting]
            delta = (
                angle_diff(filtered_data, target["v"])
                if is_rotation
                else filtered_data - target["v"]
            )

            # Only use Kalman filter's output as smoothed value
            smoothed_delta = delta * dt

            update_target_value(target, smoothed_delta, is_rotation)

        last_time = current_time
        time.sleep(frame_duration)
