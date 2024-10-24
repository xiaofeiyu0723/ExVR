import time
import numpy as np
import utils.data as data_utils
from utils.json_manager import load_json
import utils.globals as g

def setup_smoothing():
    smoothing_config = load_json("./smoothing.json")
    return smoothing_config

def update_target_value(target, smoothed_delta, is_rotation):
    if is_rotation:
        target["v"] = (target["v"] + smoothed_delta) % 360
    else:
        target["v"] += smoothed_delta


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
            max_delta = config_params.get("max_delta")
            deadzone = config_params.get("deadzone")
            is_rotation = config_params.get("is_rotation", False)
            dt = dt_base * config_params.get("dt_multiplier", 20)

            target = g.data[target_key][i - shifting]
            delta = (
                (current_data - target["v"] + 180) % 360 - 180
                if is_rotation
                else current_data - target["v"]
            )

            if (
                max_delta is not None
                and deadzone is not None
                and np.abs(delta) > deadzone
            ):
                smoothed_delta = np.clip(delta, -max_delta, max_delta) * dt
            else:
                smoothed_delta = delta * dt

            update_target_value(target, smoothed_delta, is_rotation)

        last_time = current_time
        time.sleep(frame_duration)
