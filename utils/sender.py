import socket
import struct
import time
import utils.globals as g
from scipy.spatial.transform import Rotation as R
import numpy as np

def get_value(value, value_d):
    if value["e"]:
        return value["v"] + value["s"]
    else:
        return value_d["v"] + value_d["s"]


def pack_data(data, default_data):
    packed_data = b""
    for value, value_d in zip(data["BlendShapes"][1:], default_data["BlendShapes"][1:]):
        v = get_value(value, value_d) * value["w"]
        if np.abs(v) > value["max"]:
            v = np.sign(v) * value["max"]
        packed_data += struct.pack(
            ">f",  v
        )
    return packed_data

def pack_hmd_data(data, default_data):
    yaw = get_value(data["Rotation"][0], default_data["Rotation"][0])
    pitch = get_value(data["Rotation"][1], default_data["Rotation"][1])
    roll = get_value(data["Rotation"][2], default_data["Rotation"][2])

    x = get_value(data["Position"][0], default_data["Position"][0])
    y = get_value(data["Position"][1], default_data["Position"][1])
    z = get_value(data["Position"][2], default_data["Position"][2])
    packed_hmd_data = struct.pack("6d", x, y, z, yaw, pitch, roll)
    return packed_hmd_data


def handling_hand_data(data, default_data):
    # Process left hand data
    yaw_l = get_value(data["LeftHandRotation"][0], default_data["LeftHandRotation"][0])
    pitch_l = get_value(
        data["LeftHandRotation"][1], default_data["LeftHandRotation"][1]
    )
    roll_l = get_value(data["LeftHandRotation"][2], default_data["LeftHandRotation"][2])
    x_l = get_value(data["LeftHandPosition"][0], default_data["LeftHandPosition"][0])
    y_l = get_value(data["LeftHandPosition"][1], default_data["LeftHandPosition"][1])
    z_l = get_value(data["LeftHandPosition"][2], default_data["LeftHandPosition"][2])
    quat_l = R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_quat()

    # Process right hand data
    yaw_r = get_value(
        data["RightHandRotation"][0], default_data["RightHandRotation"][0]
    )
    pitch_r = get_value(
        data["RightHandRotation"][1], default_data["RightHandRotation"][1]
    )
    roll_r = get_value(
        data["RightHandRotation"][2], default_data["RightHandRotation"][2]
    )
    x_r = get_value(data["RightHandPosition"][0], default_data["RightHandPosition"][0])
    y_r = get_value(data["RightHandPosition"][1], default_data["RightHandPosition"][1])
    z_r = get_value(data["RightHandPosition"][2], default_data["RightHandPosition"][2])
    quat_r = R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_quat()

    wrist_position_l = (x_l, y_l, z_l)
    g.controller.left_hand.position = wrist_position_l
    g.controller.left_hand.rotation = quat_l

    wrist_position_r = (x_r, y_r, z_r)
    g.controller.right_hand.position = wrist_position_r
    g.controller.right_hand.rotation = quat_r
    finger_l = tuple(
        get_value(v, v_d)
        for v, v_d in zip(data["LeftHandFinger"], default_data["LeftHandFinger"])
    )
    finger_r = tuple(
        get_value(v, v_d)
        for v, v_d in zip(data["RightHandFinger"], default_data["RightHandFinger"])
    )
    g.controller.left_hand.finger = finger_l
    g.controller.right_hand.finger = finger_r


def data_send_thread(target_ip):
    frame_duration = 1.0 / 60.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not g.stop_event.is_set():
        if g.config["Tracking"]["Head"]["enable"]:
            packed_hmd_data = pack_hmd_data(g.data, g.default_data)
            sock.sendto(packed_hmd_data, (target_ip, 4242))
        if g.config["Tracking"]["Face"]["enable"]:
            packed_data = pack_data(g.data, g.default_data)
            sock.sendto(packed_data, (target_ip, 11111))
        if g.config["Tracking"]["Hand"]["enable"]:
            handling_hand_data(g.data, g.default_data)
            g.controller.update()
        time.sleep(frame_duration)
