import cv2
import numpy as np
# from copy import deepcopy
import utils.globals as g
import onnxruntime as ort
from typing import Tuple, List

# https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

def pose_pred_handling(detection_result):
    g.pose_landmarks=detection_result
    keypoints=detection_result[1]
    scores=detection_result[2]
    if keypoints is not None and len(keypoints)>0 and len(keypoints[0])>0:
        if scores[0][9]>0.3:
            left_hand_position = keypoints[0][0]-keypoints[0][9]
            left_hand_position[0] *= g.config["Tracking"]["Hand"]["x_scalar"]
            left_hand_position[1] *= g.config["Tracking"]["Hand"]["y_scalar"]
            if g.config["Smoothing"]["enable"]:
                g.latest_data[70] = left_hand_position[0]
                g.latest_data[71] = left_hand_position[1]
            else:
                g.data["LeftHandPosition"][0]["v"] = left_hand_position[0]
                g.data["LeftHandPosition"][1]["v"] = left_hand_position[1]

        if scores[0][10]>0.3:
            right_hand_position = keypoints[0][0]-keypoints[0][10]
            right_hand_position[0] *= g.config["Tracking"]["Hand"]["x_scalar"]
            right_hand_position[1] *= g.config["Tracking"]["Hand"]["y_scalar"]
            if g.config["Smoothing"]["enable"]:
                g.latest_data[76] = right_hand_position[0]
                g.latest_data[77] = right_hand_position[1]
            else:
                g.data["RightHandPosition"][0]["v"] = right_hand_position[0]
                g.data["RightHandPosition"][1]["v"] = right_hand_position[1]


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale

def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(w > h * aspect_ratio, np.hstack([w, w / aspect_ratio]), np.hstack([h * aspect_ratio, h]))

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int]) -> np.ndarray:
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

def top_down_affine(input_size, scale, center, img):
    w, h = input_size
    warp_size = (int(w), int(h))
    scale = _fix_aspect_ratio(scale, w / h)
    warp_mat = get_warp_matrix(center, scale, 0, (w, h))
    warped_img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return warped_img, scale

def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, _ = simcc_x.shape
    x_locs = np.argmax(simcc_x.reshape(N * K, -1), axis=1)
    y_locs = np.argmax(simcc_y.reshape(N * K, -1), axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32).reshape(N, K, 2)
    max_val_x = np.max(simcc_x.reshape(N * K, -1), axis=1)
    max_val_y = np.max(simcc_y.reshape(N * K, -1), axis=1)
    max_vals = np.minimum(max_val_x, max_val_y).reshape(N, K)
    locs[max_vals <= 0.] = -1
    return locs, max_vals

def decode(simcc_x, simcc_y, simcc_split_ratio=2.0):
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores

# -------------------- MAIN API --------------------

def draw_pose_landmarks(rgb_image):
    if g.pose_landmarks is not None:
        keypoints_image=g.pose_landmarks[0]
        scores=g.pose_landmarks[2]
        thr = 0.3
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10)]
        for kpts, score in zip(keypoints_image, scores):
            for idx, (x, y) in enumerate(kpts):
                if score[idx] > thr:
                    cv2.circle(rgb_image, (int(x), int(y)), 2, (0, 255, 0), -1)
            for u, v in skeleton:
                if u < len(kpts) and v < len(kpts) and score[u] > thr and score[v] > thr:
                    cv2.line(rgb_image, tuple(kpts[u].astype(int)), tuple(kpts[v].astype(int)), (0, 255, 255), 1)
    return rgb_image


class PoseDetector:
    def __init__(self, model_path="your_model.onnx", provider="DmlExecutionProvider"):
        self.session = ort.InferenceSession(model_path, providers=[provider])
        h, w = self.session.get_inputs()[0].shape[2:]
        self.input_size = (w, h)

    def process(self, image):
        resized_img, center, scale = self._preprocess(image)
        outputs = self._inference(resized_img)
        keypoints_image, keypoints, scores = self._postprocess(outputs, center, scale, image.shape)
        result=[keypoints_image, keypoints, scores]
        return result

    def _preprocess(self, img):
        bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(self.input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        return resized_img, center, scale

    def _inference(self, img):
        input_tensor = img.transpose(2, 0, 1)[None].astype(np.float32)
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)
        return outputs

    def _postprocess(self, outputs, center, scale, image_shape):
        keypoints, scores = decode(outputs[0], outputs[1])
        keypoints_image = keypoints / np.array(self.input_size) * scale + center - scale / 2
        image_h, image_w = image_shape[:2]
        keypoints_normalized = keypoints_image / np.array([image_w, image_h])
        return keypoints_image, keypoints_normalized, scores


def initialize_pose():
    return PoseDetector(model_path="./tracker/pose/end2end_3.onnx", provider="DmlExecutionProvider")
