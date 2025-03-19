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
        zip(g.handedness, g.hand_landmarks)
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
        cv2.putText(
            rgb_image,
            f"{"Right" if hand.classification[0].label == "Left" else "Left"}",  # Handedness label
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


def finger_handling(hand_name, index, value):
    global finger_status, finger_counts
    if value < g.config["Tracking"]["Finger"]["finger_confidence"]:
        new_state = 0
    else:
        new_state = 1

    if new_state == finger_status[hand_name][index]:
        finger_counts[hand_name][index] = 0
    else:
        finger_counts[hand_name][index] += 1
        if (
            finger_counts[hand_name][index]
            >= g.config["Tracking"]["Finger"]["finger_threshold"]
        ):
            finger_status[hand_name][index] = new_state
            finger_counts[hand_name][index] = 0
    return finger_status[hand_name][index]

left_hand_detection_counts = 0
right_hand_detection_counts = 0

def hand_pred_handling(detection_result):
    global left_hand_detection_counts, right_hand_detection_counts

    g.hand_landmarks = detection_result.multi_hand_landmarks
    g.handedness = detection_result.multi_handedness

    right_hand_detection_counts -= 1
    if right_hand_detection_counts < 0:
        right_hand_detection_counts = 0
    left_hand_detection_counts -= 1
    if left_hand_detection_counts < 0:
        left_hand_detection_counts = 0

    if detection_result.multi_hand_landmarks is not None and detection_result.multi_handedness is not None and detection_result.multi_hand_world_landmarks is not None:
        for idx, (hand,hand_landmarks,hand_world_landmarks) in enumerate(zip(detection_result.multi_handedness, detection_result.multi_hand_landmarks, detection_result.multi_hand_world_landmarks)):
            if hand.classification[0].score < g.config["Tracking"]["Hand"]["hand_confidence"]:
                continue
            hand_name = "Right" if hand.classification[0].label == "Left" else "Left"

            if hand_name == "Left":
                left_hand_detection_counts += 2
                if (
                    left_hand_detection_counts
                    > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                ):
                    left_hand_detection_counts = g.config["Tracking"]["Hand"][
                        "hand_detection_upper_threshold"
                    ]
                if (
                    left_hand_detection_counts
                    <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]
                ):
                    continue
            else:
                right_hand_detection_counts += 2
                if (
                    right_hand_detection_counts
                    > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                ):
                    right_hand_detection_counts = g.config["Tracking"]["Hand"][
                        "hand_detection_upper_threshold"
                    ]
                if (
                    right_hand_detection_counts
                    <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]
                ):
                    continue
            world_landmarks = hand_world_landmarks.landmark
            hand_pose = get_hand_pose(world_landmarks)
            image_landmarks = hand_landmarks.landmark
            image_hand_pose = get_hand_pose(image_landmarks, False)
            hand_position = g.head_pos - image_hand_pose[2]
            hand_position[:2] *= [
                g.config["Tracking"]["Hand"]["x_scalar"],
                g.config["Tracking"]["Hand"]["y_scalar"]
            ]
            hand_distance = np.linalg.norm(
                np.array(image_hand_pose[1][:2]) - np.array(image_hand_pose[2][:2])
            ) / g.head_pos[2]
            hand_distance += g.config["Tracking"]["Hand"]["z_shifting"]
            hand_distance *= g.config["Tracking"]["Hand"]["z_scalar"]
            hand_distance = np.interp(hand_distance, [-2, 2], [-1.2, 1])
            if g.config["Tracking"]["Hand"]["only_front"]:
                hand_distance = np.clip(hand_distance, -0.8, 0.0)

            hand_position[2] = hand_distance

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
            # Here has problem
            # if hand_name=="Left":
            #     angle = np.deg2rad(30)
            # else:
            #     angle = np.deg2rad(-30)
            # rotation_z_35 = np.array([
            #     [np.cos(angle), -np.sin(angle), 0],
            #     [np.sin(angle), np.cos(angle), 0],
            #     [0, 0, 1]
            # ])
            # wrist_matrix = rotation_z_35 @ wrist_matrix
            wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)

            wrist_rot[0] += -g.config["Tracking"]["Head"]["yaw_calibration"]
            wrist_rot[1] += g.config["Tracking"]["Head"]["pitch_calibration"]
            wrist_rot[2] += -g.config["Tracking"]["Head"]["roll_calibration"]

            # Fingers, mediapipe is inaccurate
            if g.config["Tracking"]["Finger"]["enable"]:
                finger_angle = get_fingers_angle_from_hand3d(hand_pose)
                finger_0 = finger_handling(
                    hand_name, 0, (finger_angle[0] + finger_angle[1]) / 2 - 2.0
                )
                finger_1 = finger_handling(hand_name, 1, finger_angle[4] - 1.6)
                finger_2 = finger_handling(hand_name, 2, finger_angle[7] - 1.5)
                finger_3 = finger_handling(hand_name, 3, finger_angle[10] - 1.9)
                finger_4 = finger_handling(hand_name, 4, finger_angle[13] - 1.9)
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
    if (
        left_hand_detection_counts
        <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]
        and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["LeftController"]["enable"]
    ):
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

    if (
        right_hand_detection_counts
        <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]
        and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["RightController"]["enable"]
    ):
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
    return mp_hands.Hands(model_complexity = g.config["Model"]["Hand"]["model_complexity"],max_num_hands=2, min_detection_confidence=g.config["Model"]["Hand"]["min_hand_detection_confidence"], min_tracking_confidence=g.config["Model"]["Hand"]["min_tracking_confidence"])
