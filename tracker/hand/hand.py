import onnxruntime as _onnxruntime_preload
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import utils.globals as g
import joblib
import time
from tracker.hand.directml_hands import DirectMLHands, HAND_CONNECTIONS
from utils.paths import app_path
# from collections import deque
# import threading, queue


class HandDepthPredictor:
    def __init__(self, feature_model, regression_model):
        self.powers = feature_model.powers_.astype(np.int16)
        self.coef = regression_model.coef_.astype(np.float64)
        self.intercept = float(regression_model.intercept_)

    def predict(self, data):
        values = np.asarray(data, dtype=np.float64)
        monomials = np.prod(values[:, None, :] ** self.powers[None, :, :], axis=2)
        return monomials @ self.coef + self.intercept


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
        height, width, _ = rgb_image.shape
        points = [
            (int(round(landmark.x * width)), int(round(landmark.y * height)))
            for landmark in hand_landmarks.landmark
        ]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(rgb_image, points[start], points[end], (255, 214, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(rgb_image, point, 3, (255, 214, 0), -1, cv2.LINE_AA)

        # Get the top left corner of the detected hand's bounding box.
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

def calc_angle(vec1, vec2):
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = np.linalg.norm(np.cross(vec1, vec2))
        dot = np.dot(vec1, vec2)
        return np.degrees(np.abs(np.arctan2(cross, dot)))


def _vrchat_palm_anchor_image(image_hand_pose):
    internal = _vrchat_internal_hand_pose(image_hand_pose)
    return internal[0] + 0.5 * (internal[5] - internal[0])


def _vrchat_head_anchor_image():
    if g.face_landmarks and len(g.face_landmarks[0]) > 356:
        face = g.face_landmarks[0]
        x = 0.5 * (face[127].x + face[356].x) - 0.5
        y = 0.5 - 0.5 * (face[127].y + face[356].y)
        return np.asarray([x, y], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def _soft_limit_depth(value, center=-0.25, scale=0.28):
    return center + scale * np.tanh((value - center) / scale)


def get_fitted_hand_distance(image_hand_pose):
    keypoints = [5, 9, 13]
    data = image_hand_pose - image_hand_pose[0]
    data = np.asarray(data[keypoints].flatten()).reshape(1, -1)
    pred_distance = g.hand_regression_model.predict(data)
    hand_distance = pred_distance[0]

    head_depth = np.round(g.data["HeadImagePosition"][2]["v"], 2)
    distance_scalar = np.clip(head_depth, None, -1e-8)
    hand_distance = hand_distance / distance_scalar

    hand_distance += g.config["Tracking"]["Hand"]["z_shifting"]
    hand_distance *= g.config["Tracking"]["Hand"]["z_scalar"]
    mapped_distance = np.interp(hand_distance, [-2, 2], [-1.2, 1])
    return float(_soft_limit_depth(mapped_distance))


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_VRC_EPSILON = np.float32(9.999999747378752e-06)
_VRC_HALF = np.float32(0.5)
_VRC_ONE = np.float32(1.0)
_VRC_CONFIDENCE_MIN = np.float32(0.800000011920929)
_VRC_C8_NEUTRAL = np.float32(0.6000000238418579)
_VRC_DOT_TO_CURL_SCALE = np.float32(0.44999998807907104)
_CURL_ENDPOINT_LOW = np.asarray([0.12, 0.18, 0.18, 0.18, 0.18], dtype=np.float32)
_CURL_ENDPOINT_HIGH = np.asarray([0.72, 0.84, 0.84, 0.84, 0.84], dtype=np.float32)
_CURL_ENDPOINT_GAIN = np.float32(1.08)
_SPLAY_MAX_ANGLE_DEG = np.float32(24.0)
_SPLAY_OUTWARD_SIDE = np.asarray([1.0, 1.0, 0.0, -1.0, -1.0], dtype=np.float32)
_SPLAY_OUT_GAIN = np.asarray([0.65, 0.72, 0.82, 0.38, 0.24], dtype=np.float32)
_SPLAY_IN_GAIN = np.asarray([0.85, 1.20, 0.82, 1.45, 1.55], dtype=np.float32)
_SPLAY_CLOSE_BIAS = np.asarray([0.0, -0.22, 0.0, 0.46, 0.86], dtype=np.float32)
_SPLAY_CLOSE_FADE_ANGLE_DEG = np.float32(32.0)


def _f32(value):
    return np.float32(value)


def _clamp01(value):
    return _f32(max(0.0, min(1.0, float(_f32(value)))))


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = _f32(np.linalg.norm(v))
    if not (n > _VRC_EPSILON):
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


def _unit2(v):
    v = np.asarray(v, dtype=np.float32)
    n = _f32(np.linalg.norm(v))
    if not (n > _VRC_EPSILON):
        return np.zeros(2, dtype=np.float32)
    return (v / n).astype(np.float32)


def _dot(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return _f32(_f32(a[0] * b[0] + a[1] * b[1]) + _f32(a[2] * b[2]))


def _neutral_blend(value, neutral, alpha):
    neutral = _f32(neutral)
    return _f32(_f32(value - neutral) * alpha + neutral)


def _expand_curl_endpoints(curl):
    curl = np.asarray(curl, dtype=np.float32)
    span = _CURL_ENDPOINT_HIGH - _CURL_ENDPOINT_LOW
    expanded = (curl - _CURL_ENDPOINT_LOW) / span
    expanded = (expanded - _VRC_HALF) * _CURL_ENDPOINT_GAIN + _VRC_HALF
    return np.clip(expanded, 0.0, 1.0).astype(np.float32)


def _vrchat_internal_hand_pose(normalized_pose):
    internal = np.asarray(normalized_pose, dtype=np.float32).copy()
    internal[:, 0] = internal[:, 0] - _VRC_HALF
    internal[:, 1] = _VRC_HALF - internal[:, 1]
    internal[:, 2] = -internal[:, 2]
    return internal


def _splay_from_landmarks(internal, curl):
    forward = _unit2(internal[9, :2] - internal[0, :2])
    if not np.any(forward):
        return np.zeros(5, dtype=np.float32)

    index_side = internal[5, :2] - internal[17, :2]
    right = np.asarray([forward[1], -forward[0]], dtype=np.float32)
    if np.dot(right, index_side) < 0:
        right = -right

    chains = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20))
    values = []
    for i, (mcp, tip) in enumerate(chains):
        direction = _unit2(internal[tip, :2] - internal[mcp, :2])
        if not np.any(direction):
            values.append(0.0)
            continue
        lateral = _f32(np.dot(direction, right))
        forward_component = _f32(np.dot(direction, forward))
        angle = _f32(np.degrees(np.arctan2(lateral, forward_component)))
        side = _SPLAY_OUTWARD_SIDE[i]
        gain = _SPLAY_OUT_GAIN[i] if side == 0.0 or angle * side >= 0.0 else _SPLAY_IN_GAIN[i]
        open_weight = _clamp01((curl[i] - _f32(0.2)) / _f32(0.7))
        close_weight = _clamp01(_VRC_ONE - _f32(abs(float(angle)) / _SPLAY_CLOSE_FADE_ANGLE_DEG))
        raw_splay = _f32((angle / _SPLAY_MAX_ANGLE_DEG) * gain + _f32(_SPLAY_CLOSE_BIAS[i] * close_weight))
        values.append(float(np.clip(raw_splay * open_weight, -1.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def finger_handling(normalized_pose, score=1.0):
    internal = _vrchat_internal_hand_pose(normalized_pose)
    score = _f32(score)
    alpha = _clamp01(_f32(score - _VRC_CONFIDENCE_MIN) * _f32(5.0))

    axis_a = internal[13] - internal[0]
    axis_a_unit = _unit(axis_a)
    index_tip = _unit(internal[8] - internal[7])
    middle_tip = _unit(internal[12] - internal[11])
    ring_tip = _unit(internal[16] - internal[15])
    pinky_tip = _unit(internal[20] - internal[19])
    thumb_tip = _unit(internal[4] - internal[3])
    thumb_base = _unit(internal[1] - internal[0])

    thumb_curl = _clamp01(
        _f32(_f32(_f32(_dot(thumb_base, thumb_tip) + _VRC_ONE) * _VRC_DOT_TO_CURL_SCALE) - _f32(0.15000000596046448))
    )
    raw_curl = np.asarray(
        [
            thumb_curl,
            _f32(_f32(_dot(axis_a_unit, index_tip) + _VRC_ONE) * _VRC_DOT_TO_CURL_SCALE),
            _f32(_f32(_dot(axis_a_unit, middle_tip) + _VRC_ONE) * _VRC_DOT_TO_CURL_SCALE),
            _f32(_f32(_dot(axis_a_unit, ring_tip) + _VRC_ONE) * _VRC_DOT_TO_CURL_SCALE),
            _f32(_f32(_dot(axis_a_unit, pinky_tip) + _VRC_ONE) * _VRC_DOT_TO_CURL_SCALE),
        ],
        dtype=np.float32,
    )

    curl = np.asarray([_neutral_blend(v, _VRC_C8_NEUTRAL, alpha) for v in raw_curl], dtype=np.float32)
    curl = _expand_curl_endpoints(curl)
    splay = _splay_from_landmarks(internal, curl)

    return (
        {name: float(np.clip(value, 0.0, 1.0)) for name, value in zip(FINGER_NAMES, curl)},
        {name: float(value) for name, value in zip(FINGER_NAMES, splay)},
    )

def compute_bounding_size(reference_kp):
    xs = [kp.x for kp in reference_kp]
    ys = [kp.y for kp in reference_kp]
    return max(xs)-min(xs), max(ys)-min(ys)

def calculate_normalized_distance(kp1, kp2, reference_kp, points):
    distances = [
        np.linalg.norm(np.array([kp1[i].x, kp1[i].y]) - np.array([kp2[i].x, kp2[i].y]))
        for i in points
    ]
    avg_distance = np.mean(distances)
    width, height = compute_bounding_size(reference_kp)
    return avg_distance / max(width, height)


prev_hands = {}  # {key: {'left': landmarks, 'right': landmarks}}
def hand_is_changed(key, hand_name, hand_landmarks, change_points, change_threshold, update_flag=True):
    global prev_hands
    all_keypoints = [hand_landmarks.landmark[i] for i in range(21)]
    reference_keypoints = [hand_landmarks.landmark[i] for i in [0, 1, 17]]
    other_hand = 'Right' if hand_name == 'Left' else 'Left'
    if key not in prev_hands:
        prev_hands[key] = {}
    swap_flag = False
    if prev_hands[key]:
        if hand_name in prev_hands[key] and other_hand in prev_hands[key]:
            self_dist = calculate_normalized_distance(
                all_keypoints,
                prev_hands[key][hand_name],
                reference_keypoints,
                change_points
            )
            other_dist = calculate_normalized_distance(
                all_keypoints,
                prev_hands[key][other_hand],
                reference_keypoints,
                change_points
            )
            swap_flag = other_dist < self_dist and (self_dist - other_dist) > g.config["Tracking"]["Hand"]["hand_swap_threshold"]

    changed = False
    norm_distance = 0
    if hand_name in prev_hands[key]:
        norm_distance = calculate_normalized_distance(
            all_keypoints,
            prev_hands[key][hand_name],
            reference_keypoints,
            change_points
        )
        changed = norm_distance >= change_threshold
    else:
        changed = True
    if update_flag and not swap_flag:
        prev_hands[key][hand_name] = all_keypoints
    return changed, norm_distance, swap_flag


hand_detection_counts = {"Left":0,"Right":0}
hand_last_valid_time = {"Left":0.0,"Right":0.0}
prev_distance_scalar = None
def hand_pred_handling(detection_result):
    global hand_detection_counts, hand_last_valid_time, prev_distance_scalar
    now = time.monotonic()
    hand_seen_this_frame = {"Left": False, "Right": False}

    g.hand_landmarks = detection_result.multi_hand_landmarks
    g.handedness = detection_result.multi_handedness

    hand_detection_counts["Left"] -= 1
    if hand_detection_counts["Left"] < 0:
        hand_detection_counts["Left"] = 0
    hand_detection_counts["Right"] -= 1
    if hand_detection_counts["Right"] < 0:
        hand_detection_counts["Right"] = 0

    if detection_result.multi_hand_landmarks is not None and detection_result.multi_handedness is not None and detection_result.multi_hand_world_landmarks is not None:
        same_hand_flag=None
        if len(detection_result.multi_handedness)==2:
            hands=detection_result.multi_handedness
            if hands[0].classification[0].label == hands[1].classification[0].label:
                hand_name = "Right" if hands[0].classification[0].label == "Left" else "Left"
                _,avg_distance_0,_ = hand_is_changed("hand_change", hand_name, detection_result.multi_hand_landmarks[0],
                                                       [0,1,17],
                                                       0, False)
                _,avg_distance_1,_ = hand_is_changed("hand_change", hand_name, detection_result.multi_hand_landmarks[1],
                                                       [0,1,17],
                                                       0, False)
                if avg_distance_0<=avg_distance_1:
                    same_hand_flag=1
                else:
                    same_hand_flag=0

        for idx, (hand, hand_landmarks, hand_world_landmarks) in enumerate(
                zip(detection_result.multi_handedness, detection_result.multi_hand_landmarks,
                    detection_result.multi_hand_world_landmarks)):
            hand_name = "Right" if hand.classification[0].label == "Left" else "Left"
            if same_hand_flag == idx:
                continue
            else:
                _,_,_ = hand_is_changed("hand_change", hand_name, hand_landmarks,
                                                       [0,1,17],
                                                       0)
            if hand.classification[0].score < g.config["Tracking"]["Hand"]["hand_confidence"] or (g.config["Tracking"]["LeftController"]["enable"] and hand_name=="Left") or (g.config["Tracking"]["RightController"]["enable"] and hand_name=="Right"):
                continue
            if hand_name == "Left":
                hand_detection_counts["Left"] += 2
                if hand_detection_counts["Left"] > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    hand_detection_counts["Left"] = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if hand_detection_counts["Left"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue
            else:
                hand_detection_counts["Right"] += 2
                if hand_detection_counts["Right"] > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    hand_detection_counts["Right"] = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if hand_detection_counts["Right"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue

            world_landmarks = hand_world_landmarks.landmark
            hand_pose = get_hand_pose(world_landmarks)
            image_landmarks = hand_landmarks.landmark
            image_hand_pose = get_hand_pose(image_landmarks, False)
            hand_anchor = _vrchat_palm_anchor_image(image_hand_pose)
            head_anchor = _vrchat_head_anchor_image()
            hand_delta = hand_anchor[:2] - head_anchor
            hand_position = np.asarray([-hand_delta[0], hand_delta[1], 0.0], dtype=np.float32)
            hand_position[:2] *= [g.config["Tracking"]["Hand"]["x_scalar"], g.config["Tracking"]["Hand"]["y_scalar"]]

            hand_distance = get_fitted_hand_distance(image_hand_pose)
            # print(hand_distance)
            if g.config["Tracking"]["Hand"]["only_front"]:
                hand_distance = np.clip(hand_distance, -0.8, 0.0)

            hand_position[2] = hand_distance

            position_change_flag,_,swap_flag = hand_is_changed("position",hand_name,hand_landmarks,g.config["Tracking"]["Hand"]["position_change_points"],g.config["Tracking"]["Hand"]["position_change_threshold"])
            if hand_name=="Left":
                g.controller.left_hand.change_flag=position_change_flag
            elif hand_name=="Right":
                g.controller.right_hand.change_flag=position_change_flag

            rotation_change_flag,_,_ = hand_is_changed("rotation",hand_name,hand_landmarks,g.config["Tracking"]["Hand"]["rotation_change_points"],g.config["Tracking"]["Hand"]["rotation_change_threshold"])
            if swap_flag and g.config["Tracking"]["Hand"]["enable_swap_strategy"]:
                continue
            z = hand_pose[0] - hand_pose[17]
            x = np.cross(hand_pose[1] - hand_pose[0], z)
            y = np.cross(z, x)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            z = z / np.linalg.norm(z)

            wrist_matrix = np.vstack((x, y, z)).T
            wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)
            if g.config["Tracking"]["Finger"]["enable"]:
                hand_score = hand.classification[0].score
                finger_curl, finger_splay = finger_handling(image_hand_pose, hand_score)
                finger_0, finger_1, finger_2, finger_3, finger_4 = finger_curl["thumb"],finger_curl["index"],finger_curl["middle"],finger_curl["ring"],finger_curl["pinky"]
                splay_0, splay_1, splay_2, splay_3, splay_4 = finger_splay["thumb"],finger_splay["index"],finger_splay["middle"],finger_splay["ring"],finger_splay["pinky"]
                if g.config["Tracking"]["Hand"]["enable_finger_action"]:
                    is_left_hand = hand_name == "Left"
                    if finger_1 < g.config["Tracking"]["Hand"]["trigger_threshold"]:
                        g.controller.send_trigger(is_left_hand, 0, 1)
                    else:
                        g.controller.send_trigger(is_left_hand, 0, 0)
                    if finger_2 < g.config["Tracking"]["Hand"]["trigger_threshold"]:
                        g.controller.send_trigger(is_left_hand, 2, 1)
                    else:
                        g.controller.send_trigger(is_left_hand, 2, 0)

            else:
                finger_0, finger_1, finger_2, finger_3, finger_4 = 1.0, 1.0, 1.0, 1.0, 1.0
                splay_0, splay_1, splay_2, splay_3, splay_4 = 0.0, 0.0, 0.0, 0.0, 0.0

            hand_last_valid_time[hand_name] = now
            hand_seen_this_frame[hand_name] = True

            if hand_name == "Left":
                if g.config["Smoothing"]["enable"]:
                    if rotation_change_flag:
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
                    g.latest_data[119] = splay_0
                    g.latest_data[120] = splay_1
                    g.latest_data[121] = splay_2
                    g.latest_data[122] = splay_3
                    g.latest_data[123] = splay_4
                else:
                    if rotation_change_flag:
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
                    g.data["LeftHandSplay"][0]["v"] = splay_0
                    g.data["LeftHandSplay"][1]["v"] = splay_1
                    g.data["LeftHandSplay"][2]["v"] = splay_2
                    g.data["LeftHandSplay"][3]["v"] = splay_3
                    g.data["LeftHandSplay"][4]["v"] = splay_4
                g.controller.left_hand.enable = True
            else:
                if g.config["Smoothing"]["enable"]:
                    if rotation_change_flag:
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
                    g.latest_data[124] = splay_0
                    g.latest_data[125] = splay_1
                    g.latest_data[126] = splay_2
                    g.latest_data[127] = splay_3
                    g.latest_data[128] = splay_4
                else:
                    if rotation_change_flag:
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
                    g.data["RightHandSplay"][0]["v"] = splay_0
                    g.data["RightHandSplay"][1]["v"] = splay_1
                    g.data["RightHandSplay"][2]["v"] = splay_2
                    g.data["RightHandSplay"][3]["v"] = splay_3
                    g.data["RightHandSplay"][4]["v"] = splay_4
                g.controller.right_hand.enable = True

    hand_return_time = float(g.config["Tracking"]["Hand"].get("hand_return_time", 0.5))
    if not hand_seen_this_frame["Left"] and now - hand_last_valid_time["Left"] >= hand_return_time and \
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
            g.latest_data[119] = g.default_data["LeftHandSplay"][0]["v"]
            g.latest_data[120] = g.default_data["LeftHandSplay"][1]["v"]
            g.latest_data[121] = g.default_data["LeftHandSplay"][2]["v"]
            g.latest_data[122] = g.default_data["LeftHandSplay"][3]["v"]
            g.latest_data[123] = g.default_data["LeftHandSplay"][4]["v"]
        else:
            g.data["LeftHandPosition"] = deepcopy(g.default_data["LeftHandPosition"])
            g.data["LeftHandRotation"] = deepcopy(g.default_data["LeftHandRotation"])
            g.data["LeftHandFinger"] = deepcopy(g.default_data["LeftHandFinger"])
            g.data["LeftHandSplay"] = deepcopy(g.default_data["LeftHandSplay"])
        g.controller.left_hand.enable = False


    if not hand_seen_this_frame["Right"] and now - hand_last_valid_time["Right"] >= hand_return_time and \
            g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] and not g.config["Tracking"]["RightController"]["enable"]:
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
            g.latest_data[124] = g.default_data["RightHandSplay"][0]["v"]
            g.latest_data[125] = g.default_data["RightHandSplay"][1]["v"]
            g.latest_data[126] = g.default_data["RightHandSplay"][2]["v"]
            g.latest_data[127] = g.default_data["RightHandSplay"][3]["v"]
            g.latest_data[128] = g.default_data["RightHandSplay"][4]["v"]
        else:
            g.data["RightHandPosition"] = deepcopy(g.default_data["RightHandPosition"])
            g.data["RightHandRotation"] = deepcopy(g.default_data["RightHandRotation"])
            g.data["RightHandFinger"] = deepcopy(g.default_data["RightHandFinger"])
            g.data["RightHandSplay"] = deepcopy(g.default_data["RightHandSplay"])
        g.controller.right_hand.enable = False

def initialize_hand():
    return DirectMLHands(model_complexity=g.config["Model"]["Hand"]["model_complexity"], max_num_hands=2,
                         min_detection_confidence=g.config["Model"]["Hand"]["min_hand_detection_confidence"],
                         min_tracking_confidence=g.config["Model"]["Hand"]["min_tracking_confidence"],
                         provider=g.config["Model"]["provider"])


def initialize_hand_depth():
    feature_model = joblib.load(app_path("models", "hand_feature_model.pkl"))
    hand_regression_model = joblib.load(app_path("models", "hand_regression_model.pkl"))
    return None, HandDepthPredictor(feature_model, hand_regression_model)
