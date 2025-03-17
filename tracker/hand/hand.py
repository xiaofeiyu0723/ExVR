import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from tracker.hand.gesture import get_fingers_angle_from_hand3d
from copy import deepcopy
import utils.globals as g

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
    for idx, (hand, hand_landmarks) in enumerate(
        zip(handedness_list, hand_landmarks_list)
    ):
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
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
        handedness_label = "Right" if hand.classification[0].label == "Left" else "Left"
        cv2.putText(
            rgb_image,
            handedness_label,
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
        hand_pose[:, 0] = -hand_pose[
            :, 0
        ]  # flip the points a bit since steamvrs coordinate system is a bit diffrent
        hand_pose[:, 1] = -hand_pose[:, 1]
    return hand_pose

finger_status = {"Left": [0] * 5, "Right": [0] * 5}
finger_counts = {"Left": [0] * 5, "Right": [0] * 5}

def update_finger_status(hand_name, index, value):
    global finger_status, finger_counts
    if value < g.config["Tracking"]["Finger"]["finger_confidence"]:
        new_state = 0
    else:
        new_state = 1

    if new_state == finger_status[hand_name][index]:
        finger_counts[hand_name][index] = 0
    else:
        finger_counts[hand_name][index] += 1
        if finger_counts[hand_name][index] >= g.config["Tracking"]["Finger"]["finger_threshold"]:
            finger_status[hand_name][index] = new_state
            finger_counts[hand_name][index] = 0

    return finger_status[hand_name][index]

left_hand_wrong_counts = 0
right_hand_wrong_counts = 0
left_hand_detection_counts = 0
right_hand_detection_counts = 0
def process_single_hand(hand_name, hand, hand_landmarks, hand_world_landmarks):
    global left_hand_wrong_counts, right_hand_wrong_counts
    global left_hand_detection_counts, right_hand_detection_counts

    if hand_name == "Left":
        left_hand_detection_counts += 2
        left_hand_detection_counts = min(left_hand_detection_counts, g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"])
        if left_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
            return
    else:
        right_hand_detection_counts += 2
        right_hand_detection_counts = min(right_hand_detection_counts, g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"])
        if right_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
            return

    # 计算手的姿态和位置
    world_landmarks = hand_world_landmarks.landmark
    hand_pose = get_hand_pose(world_landmarks)
    image_landmarks = hand_landmarks.landmark
    image_hand_pose = get_hand_pose(image_landmarks, False)
    hand_position = calculate_hand_position(image_hand_pose)

    wrist_rot = calculate_wrist_rotation(hand_name,hand_pose)

    if g.config["Tracking"]["Finger"]["enable"]:
        finger_states = [
            update_finger_status(hand_name, 0, (get_fingers_angle_from_hand3d(hand_pose)[0] + get_fingers_angle_from_hand3d(hand_pose)[1]) / 2 - 2.0),
            update_finger_status(hand_name, 1, get_fingers_angle_from_hand3d(hand_pose)[4] - 1.6),
            update_finger_status(hand_name, 2, get_fingers_angle_from_hand3d(hand_pose)[7] - 1.5),
            update_finger_status(hand_name, 3, get_fingers_angle_from_hand3d(hand_pose)[10] - 1.9),
            update_finger_status(hand_name, 4, get_fingers_angle_from_hand3d(hand_pose)[13] - 1.9),
        ]
    else:
        finger_states = [1.0] * 5

    update_hand_data(hand_name, hand_position, wrist_rot, finger_states)

def calculate_hand_position(image_hand_pose):
    hand_position = g.head_pos - image_hand_pose[5]
    hand_position[0] *= g.config["Tracking"]["Hand"]["x_scalar"]
    hand_position[1] *= g.config["Tracking"]["Hand"]["y_scalar"]
    distance = (np.linalg.norm(np.array([image_hand_pose[0][0], image_hand_pose[0][1]]) - np.array([image_hand_pose[1][0], image_hand_pose[1][1]])) / g.head_pos[2]+ g.config["Tracking"]["Hand"]["z_shifting"]) * g.config["Tracking"]["Hand"]["z_scalar"]
    distance = np.interp(distance, [-2, 2], [-1, 1])
    if g.config["Tracking"]["Hand"]["only_front"]:
        distance = distance.clip(-0.5, -0.1)
    hand_position[2] = distance
    return hand_position



def calculate_wrist_rotation(hand_name,hand_pose):
    hand_f = hand_pose[17]
    hand_b = hand_pose[0]
    hand_u = hand_pose[1]
    x = hand_f - hand_b
    w = hand_u - hand_b
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    wrist_matrix = np.vstack((z, y, -x)).T
    if hand_name=="Left":
        angle = np.deg2rad(30)
    else:
        angle = np.deg2rad(-30)
    rotation_z_35 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    wrist_matrix = rotation_z_35 @ wrist_matrix
    wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)

    wrist_rot[0] += -g.config["Tracking"]["Head"]["yaw_calibration"]
    wrist_rot[1] += g.config["Tracking"]["Head"]["pitch_calibration"]
    wrist_rot[2] += -g.config["Tracking"]["Head"]["roll_calibration"]

    return wrist_rot


def update_hand_data(hand_name, hand_position, wrist_rot, finger_states):
    if g.config["Smoothing"]["enable"]:
        index_offset = 70 if hand_name == "Left" else 76
        finger_offset = 82 if hand_name == "Left" else 87

        g.latest_data[index_offset] = hand_position[0]
        g.latest_data[index_offset + 1] = hand_position[1]
        g.latest_data[index_offset + 2] = hand_position[2]

        g.latest_data[index_offset + 3] = wrist_rot[0]
        g.latest_data[index_offset + 4] = wrist_rot[1]
        g.latest_data[index_offset + 5] = wrist_rot[2]

        for i, finger_state in enumerate(finger_states):
            g.latest_data[finger_offset + i] = finger_state
    else:
        for i, val in enumerate(wrist_rot):
            g.data[f"{hand_name}HandRotation"][i]["v"] = val

        for i, val in enumerate(hand_position):
            g.data[f"{hand_name}HandPosition"][i]["v"] = val

        for i, finger_state in enumerate(finger_states):
            g.data[f"{hand_name}HandFinger"][i]["v"] = finger_state


def reset_hand_data(hand_name):
    if g.config["Smoothing"]["enable"]:
        index_offset = 70 if hand_name == "Left" else 76
        finger_offset = 82 if hand_name == "Left" else 87

        g.latest_data[index_offset:index_offset + 3] = [
            g.default_data[f"{hand_name}HandPosition"][i]["v"] for i in range(3)]
        g.latest_data[index_offset + 3:index_offset + 6] = [
            g.default_data[f"{hand_name}HandRotation"][i]["v"] for i in range(3)]
        g.latest_data[finger_offset:finger_offset + 5] = [
            g.default_data[f"{hand_name}HandFinger"][i]["v"] for i in range(5)]
    else:
        g.data[f"{hand_name}HandPosition"] = deepcopy(g.default_data[f"{hand_name}HandPosition"])
        g.data[f"{hand_name}HandRotation"] = deepcopy(g.default_data[f"{hand_name}HandRotation"])
        g.data[f"{hand_name}HandFinger"] = deepcopy(g.default_data[f"{hand_name}HandFinger"])


def hand_pred_handling(detection_result):
    global left_hand_detection_counts, right_hand_detection_counts

    g.hand_landmarks = detection_result.multi_hand_landmarks
    g.handedness = detection_result.multi_handedness

    right_hand_detection_counts = max(0, right_hand_detection_counts - 1)
    left_hand_detection_counts = max(0, left_hand_detection_counts - 1)

    if detection_result.multi_hand_landmarks and detection_result.multi_handedness and detection_result.multi_hand_world_landmarks:
        for hand, hand_landmarks, hand_world_landmarks in zip(
            detection_result.multi_handedness,
            detection_result.multi_hand_landmarks,
            detection_result.multi_hand_world_landmarks,
        ):
            if hand.classification[0].score < g.config["Tracking"]["Hand"]["hand_confidence"]:
                continue

            hand_name = "Right" if hand.classification[0].label == "Left" else "Left"
            if hand_name=="Right" and g.config["Tracking"]["RightController"]["enable"]:
                continue
            if hand_name=="Left" and g.config["Tracking"]["LeftController"]["enable"]:
                continue
            process_single_hand(hand_name, hand, hand_landmarks, hand_world_landmarks)

    if left_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["LeftController"]["enable"]:
        if g.config["Tracking"]["Hand"]["follow"]:
            g.controller.left_hand.follow = False
        reset_hand_data("Left")
    else:
        g.controller.left_hand.follow = True

    if right_hand_detection_counts <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["RightController"]["enable"]:
        if g.config["Tracking"]["Hand"]["follow"]:
            g.controller.right_hand.follow = False
        reset_hand_data("Right")
    else:
        g.controller.right_hand.follow = True


def initialize_hand():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(model_complexity = g.config["Model"]["Hand"]["model_complexity"],max_num_hands=2, min_detection_confidence=g.config["Model"]["Hand"]["min_hand_detection_confidence"], min_tracking_confidence=g.config["Model"]["Hand"]["min_tracking_confidence"])
