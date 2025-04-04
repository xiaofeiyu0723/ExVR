import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from tracker.hand.gesture import get_fingers_angle_from_hand3d
from copy import deepcopy
import utils.globals as g
from collections import deque

left_position_queue = deque(maxlen=g.config["Tracking"]["Hand"]["queue_length"])
right_position_queue = deque(maxlen=g.config["Tracking"]["Hand"]["queue_length"])

def draw_hand_landmarks(rgb_image):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    hand_landmarks_list = g.hand_landmarks
    handedness_list = g.handedness

    if hand_landmarks_list is None or handedness_list is None:
        return rgb_image

    # Loop through each detected hand using zip
    for idx, (hand, hand_landmarks) in enumerate(zip(g.handedness, g.hand_landmarks)):
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks.landmark  # Access through .landmark
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = rgb_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            rgb_image,
            f"{'Right' if hand.classification[0].label == 'Left' else 'Left'}",  # Handedness label
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return rgb_image


def get_hand_pose(landmarks, reverse_flag=True):
    hand_pose = np.asarray([[l.x, l.y, l.z] for l in landmarks])
    if reverse_flag:
        hand_pose[:, 0] = -hand_pose[:, 0]  # flip the points a bit since steamvrs coordinate system is a bit diffrent
        hand_pose[:, 1] = -hand_pose[:, 1]
    return hand_pose

class InverseKinematicsSystem:
    @staticmethod
    def safe_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])  # DEFAULT Z-axis direction
        return vec / norm

    @staticmethod
    def safe_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cross = np.cross(a, b)
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])  # DEFAULT Z-axis direction
        return cross / norm

    @staticmethod
    def vector_rotate(vec: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
        radians = np.radians(angle)
        rotation = R.from_rotvec(radians * axis)
        return rotation.apply(vec)

    @staticmethod
    def ccd_solve(joints: list, target: np.ndarray, bone_lengths: list, max_iter: int = 10) -> list:
        joints = [j.copy() for j in joints]  # Create copies to avoid modifying original data
        for _ in range(max_iter):
            for i in reversed(range(len(joints) - 1)):
                current = joints[i]
                to_end = InverseKinematicsSystem.safe_normalize(joints[-1] - current)
                to_target = InverseKinematicsSystem.safe_normalize(target - current)
                # Skip invalid calculations
                if np.isclose(np.linalg.norm(to_end), 0) or np.isclose(np.linalg.norm(to_target), 0):
                    continue
                # Calculate rotation parameters
                rot_axis = InverseKinematicsSystem.safe_cross(to_end, to_target)
                cos_theta = np.clip(np.dot(to_end, to_target), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                angle = min(15.0, angle)  # Limit single step rotation angle
                # Apply rotation to subsequent joints
                for j in range(i + 1, len(joints)):
                    vec = joints[j] - current
                    rotated = InverseKinematicsSystem.vector_rotate(vec, angle, rot_axis)
                    joints[j] = current + rotated
                # Enforce bone length constraints
                joints = InverseKinematicsSystem.enforce_lengths(joints, bone_lengths)
                # Early termination check
                if np.linalg.norm(joints[-1] - target) < 0.01:
                    return joints
        return joints

    @staticmethod
    def enforce_lengths(joints: list, lengths: list) -> list:
        """骨骼长度约束"""
        new_joints = [joints[0].copy()]
        for i in range(1, len(joints)):
            direction = InverseKinematicsSystem.safe_normalize(joints[i] - new_joints[i - 1])
            new_joints.append(new_joints[i - 1] + direction * lengths[i - 1])
        return new_joints

def calc_angle(vec1, vec2):
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = np.linalg.norm(np.cross(vec1, vec2))
        dot = np.dot(vec1, vec2)
        return np.degrees(np.abs(np.arctan2(cross, dot)))

def apply_ccd_ik(world_landmarks):
    bone_lengths = {
        'thumb': [0.03, 0.03, 0.02],
        'index': [0.03, 0.03, 0.02],
        'middle': [0.04, 0.03, 0.02],
        'ring': [0.03, 0.03, 0.02],
        'pinky': [0.03, 0.02, 0.02]
    }
    finger_indices = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }

    # Build wrist position
    wrist_pos = np.array([
        world_landmarks[0].x * 1.5,
        world_landmarks[0].y * 1.5,
        world_landmarks[0].z
    ])
    processed_joints = {}
    for finger, indices in finger_indices.items():
        # Extract raw joint positions
        raw_joints = [np.array([
            world_landmarks[i].x * 1.5,
            world_landmarks[i].y * 1.5,
            world_landmarks[i].z
        ]) for i in indices]
        # Apply CCD inverse kinematics
        solved_joints = InverseKinematicsSystem.ccd_solve(
            joints=raw_joints,
            target=raw_joints[-1].copy(),
            bone_lengths=bone_lengths[finger],
            max_iter=10
        )
        processed_joints[finger] = solved_joints

    # Prepare hand pose for finger angle calculation
    hand_pose = np.array([wrist_pos] + [joint for finger in processed_joints.values() for joint in finger])

    # Calculate finger curl
    finger_curl = finger_handling(hand_pose)
    return finger_curl

def finger_handling(hand_pose):
    finger_curl= {}
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for name in finger_names:
        cfg = g.config["Tracking"]["Finger"]
        base_start, base_end = cfg[f"{name}_base"]
        tip_start, tip_end = cfg[f"{name}_tip"]
        min_val = cfg[f"{name}_min"]
        max_val = cfg[f"{name}_max"]
        base_vec = hand_pose[base_end] - hand_pose[base_start]
        tip_vec = hand_pose[tip_end] - hand_pose[tip_start]
        if np.linalg.norm(base_vec) < 1e-6 or np.linalg.norm(tip_vec) < 1e-6:
            raw_angle = 0.0
        else:
            raw_angle = calc_angle(base_vec, tip_vec)
        clamped = np.clip(raw_angle, min_val, max_val)
        range_val = max_val - min_val
        norm_value = 1 - (clamped - min_val) / range_val
        norm_value = np.clip(norm_value, 0.0, 1.0)
        finger_curl[name] = round(norm_value,1)
    return finger_curl


left_hand_detection_counts = 0
right_hand_detection_counts = 0


def predict_hand_position(queue):
    k = len(queue)
    if k == 0:
        return None
    times = np.arange(k)
    x = np.array([pos[0] for pos in queue])
    y = np.array([pos[1] for pos in queue])
    z = np.array([pos[2] for pos in queue])

    sum_t = np.sum(times)
    sum_t2 = np.sum(times ** 2)
    denominator = k * sum_t2 - sum_t ** 2

    if denominator == 0:
        return np.array([x[-1], y[-1], z[-1]])
    else:
        sum_x = np.sum(x)
        sum_tx = np.sum(times * x)
        a_x = (k * sum_tx - sum_t * sum_x) / denominator
        b_x = (sum_x - a_x * sum_t) / k
        predicted_x = a_x * k + b_x

        sum_y = np.sum(y)
        sum_ty = np.sum(times * y)
        a_y = (k * sum_ty - sum_t * sum_y) / denominator
        b_y = (sum_y - a_y * sum_t) / k
        predicted_y = a_y * k + b_y

        sum_z = np.sum(z)
        sum_tz = np.sum(times * z)
        a_z = (k * sum_tz - sum_t * sum_z) / denominator
        b_z = (sum_z - a_z * sum_t) / k
        predicted_z = a_z * k + b_z

    return np.array([predicted_x, predicted_y, predicted_z])


def hand_pred_handling(detection_result):
    global left_hand_detection_counts, right_hand_detection_counts, left_position_queue, right_position_queue

    g.hand_landmarks = detection_result.multi_hand_landmarks
    g.handedness = detection_result.multi_handedness

    right_hand_detection_counts -= 1
    if right_hand_detection_counts < 0:
        right_hand_detection_counts = 0
    left_hand_detection_counts -= 1
    if left_hand_detection_counts < 0:
        left_hand_detection_counts = 0

    if detection_result.multi_hand_landmarks is not None and detection_result.multi_handedness is not None and detection_result.multi_hand_world_landmarks is not None:
        for idx, (hand, hand_landmarks, hand_world_landmarks) in enumerate(
                zip(detection_result.multi_handedness, detection_result.multi_hand_landmarks,
                    detection_result.multi_hand_world_landmarks)):
            if hand.classification[0].score < g.config["Tracking"]["Hand"]["hand_confidence"]:
                continue
            hand_name = "Right" if hand.classification[0].label == "Left" else "Left"

            if hand_name == "Left":
                left_hand_detection_counts += 2
                if left_hand_detection_counts > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    left_hand_detection_counts = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if left_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue
            else:
                right_hand_detection_counts += 2
                if right_hand_detection_counts > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    right_hand_detection_counts = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if right_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue

            world_landmarks = hand_world_landmarks.landmark
            hand_pose = get_hand_pose(world_landmarks)
            image_landmarks = hand_landmarks.landmark
            image_hand_pose = get_hand_pose(image_landmarks, False)
            hand_position = [g.data["HeadImagePosition"][0]["v"], g.data["HeadImagePosition"][1]["v"],
                             g.data["HeadImagePosition"][2]["v"]] - image_hand_pose[2]
            hand_position[:2] *= [g.config["Tracking"]["Hand"]["x_scalar"], g.config["Tracking"]["Hand"]["y_scalar"]]

            hand_distance = np.linalg.norm(
                np.array(image_hand_pose[1][:2]) - np.array(image_hand_pose[2][:2])) / np.clip(
                g.data["HeadImagePosition"][2]["v"], None, -1e-8)
            hand_distance += g.config["Tracking"]["Hand"]["z_shifting"]
            hand_distance *= g.config["Tracking"]["Hand"]["z_scalar"]
            hand_distance = np.interp(hand_distance, [-2, 2], [-1.2, 1])
            if g.config["Tracking"]["Hand"]["only_front"]:
                hand_distance = np.clip(hand_distance, -0.8, 0.0)

            hand_position[2] = hand_distance

            # 将hand_position转换为数组
            current_position = np.array([hand_position[0], hand_position[1], hand_position[2]])

            # 获取对应的队列
            queue = left_position_queue if hand_name == "Left" else right_position_queue

            # 检测异常移动
            if queue:
                predicted_pos = predict_hand_position(queue)
                if predicted_pos is not None:
                    diff = np.linalg.norm(predicted_pos - current_position)
                    threshold = g.config["Tracking"]["Hand"].get("movement_threshold")
                    # print(diff,threshold)
                    if diff > threshold:
                        queue.append(current_position.copy())
                        continue

            # 将当前位置加入队列
            queue.append(current_position.copy())

            z = hand_pose[0] - hand_pose[17]
            x = np.cross(hand_pose[1] - hand_pose[0], z)
            y = np.cross(z, x)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            z = z / np.linalg.norm(z)
            wrist_matrix = np.vstack((x, y, z)).T

            wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)

            wrist_rot[0] += -g.config["Tracking"]["Head"]["yaw_calibration"]
            wrist_rot[1] += g.config["Tracking"]["Head"]["pitch_calibration"]
            wrist_rot[2] += -g.config["Tracking"]["Head"]["roll_calibration"]

            if g.config["Tracking"]["Finger"]["enable"]:
                finger_curl=apply_ccd_ik(world_landmarks)
                finger_0, finger_1, finger_2, finger_3, finger_4 = finger_curl["thumb"],finger_curl["index"],finger_curl["middle"],finger_curl["ring"],finger_curl["pinky"]
                if finger_1 < 0.5:
                    g.controller.send_trigger(True if hand_name=="Left" else False, 0, 1)
                else:
                    g.controller.send_trigger(True if hand_name=="Left" else False, 0, 0)
            else:
                finger_0, finger_1, finger_2, finger_3, finger_4 = 1.0, 1.0, 1.0, 1.0, 1.0

            if hand_name == "Left":
                if g.config["Smoothing"]["enable"]:
                    g.latest_data[73] = wrist_rot[0]
                    g.latest_data[74] = wrist_rot[1]
                    g.latest_data[75] = wrist_rot[2]

                    g.latest_data[70] = hand_position[0]
                    g.latest_data[71] = hand_position[1]
                    g.latest_data[72] = hand_position[2]

                    g.latest_data[82] = finger_0
                    g.latest_data[83] = finger_1
                    g.latest_data[84] = finger_2
                    g.latest_data[85] = finger_3
                    g.latest_data[86] = finger_4
                else:
                    g.data["LeftHandRotation"][0]["v"] = wrist_rot[0]
                    g.data["LeftHandRotation"][1]["v"] = wrist_rot[1]
                    g.data["LeftHandRotation"][2]["v"] = wrist_rot[2]

                    g.data["LeftHandPosition"][0]["v"] = hand_position[0]
                    g.data["LeftHandPosition"][1]["v"] = hand_position[1]
                    g.data["LeftHandPosition"][2]["v"] = hand_position[2]

                    g.data["LeftHandFinger"][0]["v"] = finger_0
                    g.data["LeftHandFinger"][1]["v"] = finger_1
                    g.data["LeftHandFinger"][2]["v"] = finger_2
                    g.data["LeftHandFinger"][3]["v"] = finger_3
                    g.data["LeftHandFinger"][4]["v"] = finger_4

            else:
                if g.config["Smoothing"]["enable"]:
                    g.latest_data[79] = wrist_rot[0]
                    g.latest_data[80] = wrist_rot[1]
                    g.latest_data[81] = wrist_rot[2]

                    g.latest_data[76] = hand_position[0]
                    g.latest_data[77] = hand_position[1]
                    g.latest_data[78] = hand_position[2]

                    g.latest_data[87] = finger_0
                    g.latest_data[88] = finger_1
                    g.latest_data[89] = finger_2
                    g.latest_data[90] = finger_3
                    g.latest_data[91] = finger_4
                else:
                    g.data["RightHandRotation"][0]["v"] = wrist_rot[0]
                    g.data["RightHandRotation"][1]["v"] = wrist_rot[1]
                    g.data["RightHandRotation"][2]["v"] = wrist_rot[2]

                    g.data["RightHandPosition"][0]["v"] = hand_position[0]
                    g.data["RightHandPosition"][1]["v"] = hand_position[1]
                    g.data["RightHandPosition"][2]["v"] = hand_position[2]

                    g.data["RightHandFinger"][0]["v"] = finger_0
                    g.data["RightHandFinger"][1]["v"] = finger_1
                    g.data["RightHandFinger"][2]["v"] = finger_2
                    g.data["RightHandFinger"][3]["v"] = finger_3
                    g.data["RightHandFinger"][4]["v"] = finger_4

    if left_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and \
            g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["LeftController"][
        "enable"]:
        if g.config["Smoothing"]["enable"]:
            g.latest_data[73] = g.default_data["LeftHandRotation"][0]["v"]
            g.latest_data[74] = g.default_data["LeftHandRotation"][1]["v"]
            g.latest_data[75] = g.default_data["LeftHandRotation"][2]["v"]

            g.latest_data[70] = g.default_data["LeftHandPosition"][0]["v"]
            g.latest_data[71] = g.default_data["LeftHandPosition"][1]["v"]
            g.latest_data[72] = g.default_data["LeftHandPosition"][2]["v"]

            g.latest_data[82] = g.default_data["LeftHandFinger"][0]["v"]
            g.latest_data[83] = g.default_data["LeftHandFinger"][1]["v"]
            g.latest_data[84] = g.default_data["LeftHandFinger"][2]["v"]
            g.latest_data[85] = g.default_data["LeftHandFinger"][3]["v"]
            g.latest_data[86] = g.default_data["LeftHandFinger"][4]["v"]
        else:
            g.data["LeftHandPosition"] = deepcopy(g.default_data["LeftHandPosition"])
            g.data["LeftHandRotation"] = deepcopy(g.default_data["LeftHandRotation"])
            g.data["LeftHandFinger"] = deepcopy(g.default_data["LeftHandFinger"])
        if g.config["Tracking"]["Hand"]["follow"]:
            g.controller.left_hand.follow = False
    else:
        g.controller.left_hand.follow = True

    if right_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and \
            g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["RightController"][
        "enable"]:
        if g.config["Smoothing"]["enable"]:
            g.latest_data[79] = g.default_data["RightHandRotation"][0]["v"]
            g.latest_data[80] = g.default_data["RightHandRotation"][1]["v"]
            g.latest_data[81] = g.default_data["RightHandRotation"][2]["v"]

            g.latest_data[76] = g.default_data["RightHandPosition"][0]["v"]
            g.latest_data[77] = g.default_data["RightHandPosition"][1]["v"]
            g.latest_data[78] = g.default_data["RightHandPosition"][2]["v"]

            g.latest_data[87] = g.default_data["RightHandFinger"][0]["v"]
            g.latest_data[88] = g.default_data["RightHandFinger"][1]["v"]
            g.latest_data[89] = g.default_data["RightHandFinger"][2]["v"]
            g.latest_data[90] = g.default_data["RightHandFinger"][3]["v"]
            g.latest_data[91] = g.default_data["RightHandFinger"][4]["v"]
        else:
            g.data["RightHandPosition"] = deepcopy(g.default_data["RightHandPosition"])
            g.data["RightHandRotation"] = deepcopy(g.default_data["RightHandRotation"])
            g.data["RightHandFinger"] = deepcopy(g.default_data["RightHandFinger"])
        if g.config["Tracking"]["Hand"]["follow"]:
            g.controller.right_hand.follow = False
    else:
        g.controller.right_hand.follow = True


def initialize_hand():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(model_complexity=g.config["Model"]["Hand"]["model_complexity"], max_num_hands=2,
                          min_detection_confidence=g.config["Model"]["Hand"]["min_hand_detection_confidence"],
                          min_tracking_confidence=g.config["Model"]["Hand"]["min_tracking_confidence"])