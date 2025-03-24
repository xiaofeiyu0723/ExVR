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
            ">f", v
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

def calculate_endpoint(start_point, length, euler_angles):
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    direction_vector = np.array([0, 0, -length])
    rotated_vector = rotation.apply(direction_vector)
    endpoint = np.array(start_point) + rotated_vector
    return endpoint


def apply_hand_velocity(hand_type: str):
    delta_time = 1 / 60
    velocity_type = hand_type + "SlimeVRControllerVelocity"
    position_type = hand_type + "SlimeVRControllerPosition"
    g.data[position_type][0]["v"] = min(max((g.data[position_type][0]["v"]+g.data[velocity_type][0]["v"]* delta_time),-1),1)
    g.data[position_type][1]["v"] = min(max((g.data[position_type][1]["v"]+g.data[velocity_type][1]["v"]* delta_time),-1),1)
    g.data[position_type][2]["v"] = min(max((g.data[position_type][2]["v"]+g.data[velocity_type][2]["v"]* delta_time),-1),1)
    sign = g.config["Tracking"]["Hand"]["lose_rate_per_second"] if g.data[velocity_type][0]["v"] > 0 else  \
        -g.config["Tracking"]["Hand"]["lose_rate_per_second"]
    if (abs(g.data[velocity_type][0]["v"]) >= abs(sign)):
        g.data[velocity_type][0]["v"] = g.data[velocity_type][0]["v"] - sign
    else:
        g.data[velocity_type][0]["v"] = 0
    sign = g.config["Tracking"]["Hand"]["lose_rate_per_second"] if g.data[velocity_type][1]["v"] > 0 else \
        -g.config["Tracking"]["Hand"]["lose_rate_per_second"]
    if (abs(g.data[velocity_type][1]["v"]) >= abs(sign)):
        g.data[velocity_type][1]["v"] = g.data[velocity_type][1]["v"] - sign
    else:
        g.data[velocity_type][1]["v"] = 0
    sign = g.config["Tracking"]["Hand"]["lose_rate_per_second"] if g.data[velocity_type][2]["v"] > 0 else \
        -g.config["Tracking"]["Hand"]["lose_rate_per_second"]
    if (abs(g.data[velocity_type][2]["v"]) >= abs(sign)):
        g.data[velocity_type][2]["v"] = g.data[velocity_type][2]["v"] - sign
    else:
        g.data[velocity_type][2]["v"] = 0
    return g.data[position_type][0]["v"],g.data[position_type][1]["v"],g.data[position_type][2]["v"]


def handling_hand_data(data, default_data):
    if g.config["Tracking"]["LeftController"]["enable"] :
        left_hand_type = "Controller"
    else:
        left_hand_type = "Hand"
    if g.config["Tracking"]["RightController"]["enable"] :
        right_hand_type = "Controller"
    else:
        right_hand_type = "Hand"

    # Process left hand data

    yaw_l = get_value(data[f"Left{left_hand_type}Rotation"][0], default_data[f"Left{left_hand_type}Rotation"][0])
    pitch_l = get_value(
        data[f"Left{left_hand_type}Rotation"][1], default_data[f"Left{left_hand_type}Rotation"][1]
    )
    roll_l = get_value(data[f"Left{left_hand_type}Rotation"][2], default_data[f"Left{left_hand_type}Rotation"][2])

    if g.slimeVR_device_enable["slimeVR_controller_left"]:
        if(g.video_media_pipline["Left"]):
            x_l = get_value(data[f"LeftHandPosition"][0], default_data[f"LeftHandPosition"][0])
            y_l = get_value(data[f"LeftHandPosition"][1], default_data[f"LeftHandPosition"][1])
            z_l = get_value(data[f"LeftHandPosition"][2], default_data[f"LeftHandPosition"][2])
            g.data["LeftSlimeVRControllerPosition"][0]["v"]=x_l
            g.data["LeftSlimeVRControllerPosition"][1]["v"]=y_l
            g.data["LeftSlimeVRControllerPosition"][2]["v"]=z_l
            g.data["LeftSlimeVRControllerVelocity"][0]["v"]= 0
            g.data["LeftSlimeVRControllerVelocity"][1]["v"]= 0
            g.data["LeftSlimeVRControllerVelocity"][2]["v"]= 0
        else:
            x,y,z=apply_hand_velocity("Left")
            data[f"Left{left_hand_type}Position"][0]["v"] = x
            data[f"Left{left_hand_type}Position"][1]["v"] = y
            data[f"Left{left_hand_type}Position"][2]["v"] = z
    elif g.config["Tracking"]["LeftController"]["enable"]:
        base_x_l = g.config["Tracking"]["LeftController"]["base_x"]
        base_y_l = g.config["Tracking"]["LeftController"]["base_y"]
        base_z_l = g.config["Tracking"]["LeftController"]["base_z"]
        length_l = g.config["Tracking"]["LeftController"]["length"]
        (data[f"Left{left_hand_type}Position"][0]["v"],
         data[f"Left{left_hand_type}Position"][1]["v"],
         data[f"Left{left_hand_type}Position"][2]["v"]) = \
            calculate_endpoint([base_x_l, base_y_l, base_z_l], length_l, [yaw_l - 40, pitch_l, roll_l])

    x_l = get_value(data[f"Left{left_hand_type}Position"][0], default_data[f"Left{left_hand_type}Position"][0])
    y_l = get_value(data[f"Left{left_hand_type}Position"][1], default_data[f"Left{left_hand_type}Position"][1])
    z_l = get_value(data[f"Left{left_hand_type}Position"][2], default_data[f"Left{left_hand_type}Position"][2])
    if(g.slimeVR_device_enable["slimeVR_controller_left"]):
        quat_l=(g.data["LeftSlimeVRControllerRotation"][0]["v"],  g.data["LeftSlimeVRControllerRotation"][1]["v"] ,
                g.data["LeftSlimeVRControllerRotation"][2]["v"] ,g.data["LeftSlimeVRControllerRotation"][3]["v"])
    else:
        quat_l = R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_quat()

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
    if g.slimeVR_device_enable["slimeVR_controller_right"]:
        if (g.video_media_pipline["Right"]):
            x_r = get_value(data[f"RightHandPosition"][0], default_data[f"RightHandPosition"][0])
            y_r = get_value(data[f"RightHandPosition"][1], default_data[f"RightHandPosition"][1])
            z_r = get_value(data[f"RightHandPosition"][2], default_data[f"RightHandPosition"][2])
            g.data["RightSlimeVRControllerPosition"][0]["v"] = x_r
            g.data["RightSlimeVRControllerPosition"][1]["v"] = y_r
            g.data["RightSlimeVRControllerPosition"][2]["v"] = z_r
            g.data["RightSlimeVRControllerVelocity"][0]["v"]= 0
            g.data["RightSlimeVRControllerVelocity"][1]["v"]= 0
            g.data["RightSlimeVRControllerVelocity"][2]["v"]= 0


        else:
            x, y, z = apply_hand_velocity("Right")
            data[f"Right{left_hand_type}Position"][0]["v"] = x
            data[f"Right{left_hand_type}Position"][1]["v"] = y
            data[f"Right{left_hand_type}Position"][2]["v"] = z

    if g.config["Tracking"]["RightController"]["enable"]:
        base_x_r = g.config["Tracking"]["RightController"]["base_x"]
        base_y_r = g.config["Tracking"]["RightController"]["base_y"]
        base_z_r = g.config["Tracking"]["RightController"]["base_z"]
        length_r = g.config["Tracking"]["RightController"]["length"]
        data[f"Right{right_hand_type}Position"][0]["v"], data[f"Right{right_hand_type}Position"][1]["v"], \
        data[f"Right{right_hand_type}Position"][2]["v"] = calculate_endpoint([base_x_r, base_y_r, base_z_r], length_r,
                                                                             [yaw_r - 40, pitch_r, roll_r])

    x_r = get_value(data[f"Right{right_hand_type}Position"][0], default_data[f"Right{right_hand_type}Position"][0])
    y_r = get_value(data[f"Right{right_hand_type}Position"][1], default_data[f"Right{right_hand_type}Position"][1])
    z_r = get_value(data[f"Right{right_hand_type}Position"][2], default_data[f"Right{right_hand_type}Position"][2])
    if (g.slimeVR_device_enable["slimeVR_controller_right"]):
        quat_r = (g.data["RightSlimeVRControllerRotation"][0]["v"], g.data["RightSlimeVRControllerRotation"][1]["v"],
                  g.data["RightSlimeVRControllerRotation"][2]["v"], g.data["RightSlimeVRControllerRotation"][3]["v"])
    else:
        quat_r = R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_quat()

    wrist_position_l = (x_l, y_l, z_l)
    #print(f"AAA{wrist_position_l}")
    g.controller.left_hand.position = wrist_position_l
    g.controller.left_hand.rotation = quat_l

    wrist_position_r = (x_r, y_r, z_r)
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
        if (g.config["Tracking"]["Hand"]["enable"]
                or g.config["Tracking"]["LeftController"]["enable"]
                or g.config["Tracking"]["RightController"]["enable"]
                or g.slimeVR_device_enable["slimeVR_controller_left"]
                or g.slimeVR_device_enable["slimeVR_controller_right"]):
            handling_hand_data(g.data, g.default_data)
            g.controller.update()
        time.sleep(frame_duration)
