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
    """Pack ExVR head pose for the upstream VRto3D OpenTrack receiver.

    The old ExVR VirtualHMD fork accepted 7 doubles:
        X Y Z + quaternion W X Y Z

    Upstream VRto3D accepts the standard OpenTrack packet:
        X Y Z + Yaw Pitch Roll
    where all six values are little-endian doubles and position is in cm.

    Upstream rebuilds the orientation in OpenTrackThread as
        HmdQuaternion_FromEulerAngles(Roll, Pitch, -Yaw)
    and that function (utils/vrmath/vrmath.h) matches scipy's 'zxy'
    extrinsic convention: R = Ry(yaw) @ Rx(pitch) @ Rz(roll).
    Decompose the old fork's quaternion in that exact convention and
    negate the third angle for OpenTrack's Yaw field.

    Preserve the old fork's final SteamVR pose by applying the
    corresponding coordinate-system conversion for position as well.
    """
    global hmd_data_prev, hmd_data_curr
    rot = [get_value(a, b) for a, b in zip(data["Rotation"], default["Rotation"])]
    pos = [get_value(a, b) for a, b in zip(data["Position"], default["Position"])]
    # This is the exact orientation that the old VirtualHMD fork put into
    # DriverPose_t. Keep it as a quaternion before converting to the Euler
    # representation expected by upstream VRto3D.
    old_rotation = R.from_euler('xzy', [rot[1], rot[2], -rot[0]], degrees=True)
    if hmd_data_prev is None:
        hmd_data_prev = hmd_data_curr = pos
    delta = R.from_euler(
        'z', get_shift(data["Rotation"][0], default["Rotation"][0]), degrees=True
    ).apply([p - q for p, q in zip(pos, hmd_data_prev)])
    hmd_data_curr = [c + d for c, d in zip(hmd_data_curr, delta)]
    hmd_data_prev = pos

    # Upstream VRto3D reconstructs the pose as:
    #   HmdQuaternion_FromEulerAngles(Roll, Pitch, -Yaw)
    # which uses the 'zxy' extrinsic convention (R = Ry @ Rx @ Rz, angles
    # ordered z, x, y — exactly the packet's Roll, Pitch, Yaw fields).
    # Decompose the old quaternion into that convention and negate the
    # final angle for OpenTrack's Yaw field.
    roll_zxy, pitch_zxy, yaw_zxy = old_rotation.as_euler('zxy')
    open_track_yaw = -yaw_zxy
    open_track_pitch = pitch_zxy
    open_track_roll = roll_zxy

    # Old fork -> SteamVR position:
    #   VR X = pX / 100, VR Y = pZ / 100, VR Z = pY / 100
    # Upstream OpenTrack -> SteamVR position:
    #   VR X = -X / 100, VR Y = -Y / 100, VR Z = Z / 100
    # Therefore send X=-pX, Y=-pZ, Z=pY.
    open_track_pos = (
        -hmd_data_curr[0],
        -hmd_data_curr[2],
        hmd_data_curr[1],
    )

    # OpenTrack's UDP protocol is six little-endian doubles; angles in
    # degrees (upstream applies DEG_TO_RAD when it reconstructs the pose).
    return struct.pack(
        "<6d",
        *open_track_pos,
        *(np.degrees(angle) for angle in (open_track_yaw, open_track_pitch, open_track_roll)),
    )
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
