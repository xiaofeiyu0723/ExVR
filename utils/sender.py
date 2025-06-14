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
    x = get_value_without_shifting(data["Position"][0], default_data["Position"][0])
    y = get_value_without_shifting(data["Position"][1], default_data["Position"][1])
    z = get_value_without_shifting(data["Position"][2], default_data["Position"][2])
    yaw_calibration = g.data["Rotation"][0]["s"]
    calibration_rot = R.from_euler("z", -yaw_calibration, degrees=True)
    calibrated_diff = calibration_rot.apply([
        g.data["Position"][0]["s"],
        g.data["Position"][1]["s"],
        g.data["Position"][2]["s"]
    ])
    x += calibrated_diff[0]
    y += calibrated_diff[1]
    z += calibrated_diff[2]

    yaw = get_value_without_shifting(data["Rotation"][0], default_data["Rotation"][0])
    pitch = get_value_without_shifting(data["Rotation"][1], default_data["Rotation"][1])
    roll = get_value_without_shifting(data["Rotation"][2], default_data["Rotation"][2])
    yaw_shifting=data["Rotation"][0]["s"]
    pitch_shifting=data["Rotation"][1]["s"]
    roll_shifting=data["Rotation"][2]["s"]
    # packed_hmd_data = struct.pack("6d", x, y, z, yaw, pitch, roll)
    packed_hmd_data = struct.pack("9d", x, y, z, yaw, pitch, roll, yaw_shifting, pitch_shifting, roll_shifting)

    return packed_hmd_data

def calculate_endpoint(start_point, length, euler_angles):
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    direction_vector = np.array([0, 0, -length])
    rotated_vector = rotation.apply(direction_vector)
    endpoint = np.array(start_point) + rotated_vector
    return endpoint

def handling_hand_data(data, default_data):
    if g.config["Tracking"]["LeftController"]["enable"]:
        left_hand_type="Controller"
    else:
        left_hand_type = "Hand"
    if g.config["Tracking"]["RightController"]["enable"]:
        right_hand_type="Controller"
    else:
        right_hand_type="Hand"

    # Process left hand data
    yaw_l = get_value(data[f"Left{left_hand_type}Rotation"][0], default_data[f"Left{left_hand_type}Rotation"][0])
    pitch_l = get_value(
        data[f"Left{left_hand_type}Rotation"][1], default_data[f"Left{left_hand_type}Rotation"][1]
    )
    roll_l = get_value(data[f"Left{left_hand_type}Rotation"][2], default_data[f"Left{left_hand_type}Rotation"][2])
    if g.config["Tracking"]["LeftController"]["enable"]:
        base_x_l=g.config["Tracking"]["LeftController"]["base_x"]
        base_y_l=g.config["Tracking"]["LeftController"]["base_y"]
        base_z_l=g.config["Tracking"]["LeftController"]["base_z"]
        length_l=g.config["Tracking"]["LeftController"]["length"]
        data[f"Left{left_hand_type}Position"][0]["v"],data[f"Left{left_hand_type}Position"][1]["v"],data[f"Left{left_hand_type}Position"][2]["v"] = calculate_endpoint([base_x_l,base_y_l,base_z_l], length_l, [yaw_l-40,pitch_l,roll_l])

    x_l = get_value(data[f"Left{left_hand_type}Position"][0], default_data[f"Left{left_hand_type}Position"][0])
    y_l = get_value(data[f"Left{left_hand_type}Position"][1], default_data[f"Left{left_hand_type}Position"][1])
    z_l = get_value(data[f"Left{left_hand_type}Position"][2], default_data[f"Left{left_hand_type}Position"][2])
    quat_l = R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_quat()
    matrix_l=R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_matrix()
    # Process right hand data
    yaw_r = get_value(
        data[f"Right{right_hand_type}Rotation"][0], default_data[f"Right{right_hand_type}Rotation"][0]
    )
    pitch_r = get_value(
        data[f"Right{right_hand_type}Rotation"][1], default_data[f"Right{right_hand_type}Rotation"][1]
    )
    roll_r = get_value(
        data[f"Right{right_hand_type}Rotation"][2], default_data[f"Right{right_hand_type}Rotation"][2]
    )
    if g.config["Tracking"]["RightController"]["enable"]:
        base_x_r=g.config["Tracking"]["RightController"]["base_x"]
        base_y_r=g.config["Tracking"]["RightController"]["base_y"]
        base_z_r=g.config["Tracking"]["RightController"]["base_z"]
        length_r=g.config["Tracking"]["RightController"]["length"]
        data[f"Right{right_hand_type}Position"][0]["v"],data[f"Right{right_hand_type}Position"][1]["v"],data[f"Right{right_hand_type}Position"][2]["v"] = calculate_endpoint([base_x_r,base_y_r,base_z_r], length_r, [yaw_r-40,pitch_r,roll_r])

    x_r = get_value(data[f"Right{right_hand_type}Position"][0], default_data[f"Right{right_hand_type}Position"][0])
    y_r = get_value(data[f"Right{right_hand_type}Position"][1], default_data[f"Right{right_hand_type}Position"][1])
    z_r = get_value(data[f"Right{right_hand_type}Position"][2], default_data[f"Right{right_hand_type}Position"][2])
    quat_r = R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_quat()
    matrix_r=R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_matrix()

    if not g.config["Tracking"]["Pose"]["enable"]:
        center_l = np.array([g.config["Tracking"]["Hand"]["center_l_x"], g.config["Tracking"]["Hand"]["center_l_y"],
                           g.config["Tracking"]["Hand"]["center_l_z"]])
    else:
        center_l = np.array([g.config["Tracking"]["Pose"]["center_l_x"], g.config["Tracking"]["Pose"]["center_l_y"],
                           g.config["Tracking"]["Pose"]["center_l_z"]])

    wrist_position_l = (x_l, y_l, z_l)
    wrist_position_l = np.array(wrist_position_l)
    wrist_position_l = matrix_l @ center_l + wrist_position_l
    wrist_position_l=(wrist_position_l[0], wrist_position_l[1] ,wrist_position_l[2])
    g.controller.left_hand.position = wrist_position_l
    g.controller.left_hand.rotation = quat_l

    if not g.config["Tracking"]["Pose"]["enable"]:
        center_r = np.array([g.config["Tracking"]["Hand"]["center_r_x"], g.config["Tracking"]["Hand"]["center_r_y"],
                           g.config["Tracking"]["Hand"]["center_r_z"]])
    else:
        center_r = np.array([g.config["Tracking"]["Pose"]["center_r_x"], g.config["Tracking"]["Pose"]["center_r_y"],
                           g.config["Tracking"]["Pose"]["center_r_z"]])

    wrist_position_r = (x_r, y_r, z_r)
    wrist_position_r = np.array(wrist_position_r)
    wrist_position_r = matrix_r @ center_r + wrist_position_r
    wrist_position_r=(wrist_position_r[0], wrist_position_r[1] ,wrist_position_r[2])
    g.controller.right_hand.position = wrist_position_r
    g.controller.right_hand.rotation = quat_r

    finger_l = tuple(
        get_value(v, v_d)
        for v, v_d in zip(data[f"Left{left_hand_type}Finger"], default_data[f"Left{left_hand_type}Finger"])
    )
    finger_r = tuple(
        get_value(v, v_d)
        for v, v_d in zip(data[f"Right{right_hand_type}Finger"], default_data[f"Right{right_hand_type}Finger"])
    )
    # print(f"Right{right_hand_type}Finger",finger_r)

    g.controller.left_hand.finger = finger_l
    g.controller.right_hand.finger = finger_r

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
