import math
import socket
import struct
import time
import utils.globals as g
from scipy.spatial.transform import Rotation as R
import numpy as np
from utils.actions import set_head_yaw, set_head_pitch

def get_value(value, value_d):
    if value["e"]:
        return value["v"] + value["s"]
    else:
        return value_d["v"] + value_d["s"]

def get_value_without_shifting(value, value_d):
    if value["e"]:
        return value["v"]
    else:
        return value_d["v"]

def get_shift(value, value_d):
    if value["e"]:
        return value["s"]
    else:
        return value_d["s"]

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

hmd_data_prev=None
hmd_data_curr=None
def pack_hmd_data(data, default):
    global hmd_data_prev, hmd_data_curr
    rot = [get_value(a, b) for a, b in zip(data["Rotation"], default["Rotation"])]
    pos = [get_value(a, b) for a, b in zip(data["Position"], default["Position"])]
    qx, qy, qz, qw = R.from_euler('xzy', [rot[1], rot[2], -rot[0]], degrees=True).as_quat()
    if hmd_data_prev is None:
        hmd_data_prev = hmd_data_curr = pos
    delta = R.from_euler(
        'z', get_shift(data["Rotation"][0], default["Rotation"][0]), degrees=True
    ).apply([p - q for p, q in zip(pos, hmd_data_prev)])
    hmd_data_curr = [c + d for c, d in zip(hmd_data_curr, delta)]
    hmd_data_prev = pos
    return struct.pack("7d", *hmd_data_curr, qw, qx, qy, qz)

def calculate_endpoint(start_point, length, euler_angles):
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    direction_vector = np.array([0, 0, -length])
    rotated_vector = rotation.apply(direction_vector)
    endpoint = np.array(start_point) + rotated_vector
    return endpoint


_VMT_WRIST_BONE_POSITION = {
    True: np.asarray([-0.03403769, 0.03650266, 0.16472160], dtype=float),
    False: np.asarray([0.03403769, 0.03650266, 0.16472160], dtype=float),
}

def apply_vmt_wrist_alignment(position, quat, is_left_hand):
    hand_config = g.config["Tracking"]["Hand"]
    controller_position = np.asarray(position, dtype=float)
    controller_rotation = R.from_quat(quat)
    if hand_config.get("vmt_align_wrist_bone", False):
        wrist_position = _VMT_WRIST_BONE_POSITION[is_left_hand]
        controller_position = controller_position - controller_rotation.apply(wrist_position)

    return tuple(controller_position), quat

def _apply_controller_endpoint(data, hand_name, yaw, pitch, roll):
    config = g.config["Tracking"][f"{hand_name}Controller"]
    position = calculate_endpoint(
        [config["base_x"], config["base_y"], config["base_z"]],
        config["length"],
        [yaw - 40, pitch, roll],
    )
    for index, value in enumerate(position):
        data[f"{hand_name}ControllerPosition"][index]["v"] = value


def _update_hand_target(data, default_data, hand_name, source_type, target, is_left_hand):
    yaw = get_value(data[f"{hand_name}{source_type}Rotation"][0], default_data[f"{hand_name}{source_type}Rotation"][0])
    pitch = get_value(data[f"{hand_name}{source_type}Rotation"][1], default_data[f"{hand_name}{source_type}Rotation"][1])
    roll = get_value(data[f"{hand_name}{source_type}Rotation"][2], default_data[f"{hand_name}{source_type}Rotation"][2])

    if source_type == "Controller":
        _apply_controller_endpoint(data, hand_name, yaw, pitch, roll)

    x = get_value(data[f"{hand_name}{source_type}Position"][0], default_data[f"{hand_name}{source_type}Position"][0])
    y = get_value(data[f"{hand_name}{source_type}Position"][1], default_data[f"{hand_name}{source_type}Position"][1])
    z = get_value(data[f"{hand_name}{source_type}Position"][2], default_data[f"{hand_name}{source_type}Position"][2])
    quat = R.from_euler("xyz", [yaw, pitch, roll], degrees=True).as_quat()

    position, quat = apply_vmt_wrist_alignment((x, y, z), quat, is_left_hand)
    target.position = position
    target.rotation = quat
    target.finger = tuple(
        get_value(value, default)
        for value, default in zip(
            data[f"{hand_name}{source_type}Finger"],
            default_data[f"{hand_name}{source_type}Finger"],
        )
    )

    splay_key = f"{hand_name}{source_type}Splay"
    if source_type == "Hand" and splay_key in data:
        target.splay = tuple(
            get_value(value, default)
            for value, default in zip(data[splay_key], default_data[splay_key])
        )
    else:
        target.splay = (0.0, 0.0, 0.0, 0.0, 0.0)


def handling_hand_data(data, default_data):
    _update_hand_target(data, default_data, "Left", "Hand", g.controller.left_hand, True)
    _update_hand_target(data, default_data, "Right", "Hand", g.controller.right_hand, False)

    g.controller.left_controller.enable = g.config["Tracking"]["LeftController"]["enable"]
    g.controller.right_controller.enable = g.config["Tracking"]["RightController"]["enable"]

    if g.controller.left_controller.enable:
        g.controller.left_hand.enable = False
    if g.controller.right_controller.enable:
        g.controller.right_hand.enable = False

    if g.controller.left_controller.enable or g.controller.left_controller.force_enable:
        _update_hand_target(data, default_data, "Left", "Controller", g.controller.left_controller, True)
    if g.controller.right_controller.enable or g.controller.right_controller.force_enable:
        _update_hand_target(data, default_data, "Right", "Controller", g.controller.right_controller, False)

prev_x=None
prev_y=None
def send_mouse_position(data, default_data):
    global prev_x, prev_y
    x = data["MousePosition"][0]["v"]
    y = data["MousePosition"][1]["v"]
    if abs(x)>=g.config["Mouse"]["bound_threshold"]:
        data["MousePosition"][0]["s"]+=math.copysign(1,x)*g.config["Mouse"]["dx"]/10
        data["MousePosition"][0]["s"]=data["MousePosition"][0]["s"]%360
    x = get_value(data["MousePosition"][0],default_data["MousePosition"][0])
    y = get_value(data["MousePosition"][1],default_data["MousePosition"][1])
    if prev_x is None or prev_y is None:
        prev_x = x
        prev_y = y
    else:
        if prev_x == x and prev_y == y:
            pass
        else:
            set_head_yaw(x * g.config['Mouse']["scalar_x"])
            set_head_pitch(y * g.config['Mouse']["scalar_y"])
            prev_x = x
            prev_y = y

def data_send_thread(target_ip):
    frame_duration = 1.0 / 60.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not g.stop_event.is_set():
        if g.config['Mouse']["enable"]:
            send_mouse_position(g.data, g.default_data)
        if g.config["Tracking"]["Head"]["enable"] or g.config["Mouse"]["enable"]:
            packed_hmd_data = pack_hmd_data(g.data, g.default_data)
            sock.sendto(packed_hmd_data, (target_ip, 4242))
        if g.config["Tracking"]["Face"]["enable"]:
            packed_data = pack_data(g.data, g.default_data)
            sock.sendto(packed_data, (target_ip, 11111))
        if g.config["Tracking"]["Hand"]["enable"] or g.config["Tracking"]["LeftController"]["enable"] or g.config["Tracking"]["RightController"]["enable"]:
            handling_hand_data(g.data, g.default_data)
            g.controller.update()
        time.sleep(frame_duration)
