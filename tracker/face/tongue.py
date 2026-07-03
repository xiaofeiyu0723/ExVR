from __future__ import annotations

import onnxruntime as ort
import cv2
import math
import numpy as np
from scipy.ndimage import uniform_filter

import utils.globals as g
from utils.model_provider import providers_for, session_options_for
from utils.ort_scheduler import ORT_PRIORITY_FACE, run_ort
from utils.paths import app_path


class OnnxTongueModel:
    def __init__(self, model_path: str = "modules/tongue_keypoint.onnx", provider: str = "GPU") -> None:
        self.session = ort.InferenceSession(
            str(app_path(model_path)),
            session_options_for(provider),
            providers=providers_for(provider),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.model_provider = provider

    @property
    def providers(self):
        return self.session.get_providers()

    def predict(self, mouth_image: np.ndarray) -> tuple[np.ndarray, float]:
        input_tensor = mouth_image.astype(np.float32)[None, None, :, :] / 255.0
        heatmap, classification = run_ort(
            self.session,
            None,
            {self.input_name: input_tensor},
            priority=ORT_PRIORITY_FACE,
        )
        return heatmap, float(classification.reshape(-1)[0])


def draw_tongue_position(rgb_image):
    try:
        face_landmarks_list = g.face_landmarks
        if face_landmarks_list is None:
            return rgb_image

        for idx in range(len(face_landmarks_list)):
            mouth_landmarks = [face_landmarks_list[idx][i] for i in [57, 287, 164, 18]]

            min_x = int(min([lm.x for lm in mouth_landmarks]) * rgb_image.shape[1])
            max_x = int(max([lm.x for lm in mouth_landmarks]) * rgb_image.shape[1])
            min_y = int(min([lm.y for lm in mouth_landmarks]) * rgb_image.shape[0])
            max_y = int(max([lm.y for lm in mouth_landmarks]) * rgb_image.shape[0])

            cv2.rectangle(rgb_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            tongue_out = g.data["BlendShapes"][52]["v"]
            if tongue_out >= 0.0001:
                height, width, _ = rgb_image.shape
                left_corner, right_corner = mouth_landmarks[0], mouth_landmarks[1]
                dx = (right_corner.x - left_corner.x) * width
                dy = (right_corner.y - left_corner.y) * height
                angle = math.degrees(math.atan2(dy, dx))

                tongue_x = -g.data["BlendShapes"][62]["v"] * 32
                tongue_y = -g.data["BlendShapes"][63]["v"] * 32

                center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
                m_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                tongue_position = np.array([tongue_x + center[0], tongue_y + center[1], 1])
                tongue_original = m_inv.dot(tongue_position)
                tongue_original[0] = np.clip(tongue_original[0], min_x, max_x)
                tongue_original[1] = np.clip(tongue_original[1], min_y, max_y)
                cv2.circle(rgb_image, (int(tongue_original[0]), int(tongue_original[1])), 4, (255, 0, 0), -1)
        return rgb_image
    except Exception as e:
        print(f"Error: {e}")
        return rgb_image


def initialize_tongue_model():
    return OnnxTongueModel(provider=g.config["Model"]["provider"])


def mouth_roi_on_image(rgb_image, face_landmarks):
    try:
        mouth_landmarks = [face_landmarks[i] for i in [57, 287, 164, 18]]

        min_x = min([lm.x for lm in mouth_landmarks])
        max_x = max([lm.x for lm in mouth_landmarks])
        min_y = min([lm.y for lm in mouth_landmarks])
        max_y = max([lm.y for lm in mouth_landmarks])

        height, width, _ = rgb_image.shape
        start_point = (int(min_x * width), int(min_y * height))
        end_point = (int(max_x * width), int(max_y * height))

        left_corner, right_corner = mouth_landmarks[0], mouth_landmarks[1]
        dx = (right_corner.x - left_corner.x) * width
        dy = (right_corner.y - left_corner.y) * height
        angle = math.degrees(math.atan2(dy, dx))

        m = cv2.getRotationMatrix2D(((width // 2), (height // 2)), angle, 1.0)

        box = np.array(
            [
                [start_point[0], start_point[1]],
                [end_point[0], start_point[1]],
                [end_point[0], end_point[1]],
                [start_point[0], end_point[1]],
            ],
            dtype="float32",
        )
        box_rotated = cv2.transform(np.array([box]), m)[0]

        min_x_rot = max(0, int(min(box_rotated[:, 0])))
        max_x_rot = min(width, int(max(box_rotated[:, 0])))
        min_y_rot = max(0, int(min(box_rotated[:, 1])))
        max_y_rot = min(height, int(max(box_rotated[:, 1])))

        roi_w = max_x_rot - min_x_rot
        roi_h = max_y_rot - min_y_rot
        if roi_w <= 0 or roi_h <= 0:
            return None

        m_roi = m.copy()
        m_roi[0, 2] -= min_x_rot
        m_roi[1, 2] -= min_y_rot
        mouth_roi = cv2.warpAffine(
            rgb_image,
            m_roi,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
        mouth_roi = cv2.resize(mouth_roi, (32, 32), interpolation=cv2.INTER_LINEAR)
        return mouth_roi
    except Exception:
        return None


def max_average_point(heatmap, window_size=5):
    filtered_heatmap = uniform_filter(heatmap, size=window_size)
    return np.unravel_index(np.argmax(filtered_heatmap, axis=None), heatmap.shape)


tongue_count = 0


def detect_tongue(mouth_image, tongue_model, data):
    global tongue_count
    tongue_out, tongue_x, tongue_y = 0.0, 0.0, 0.0
    if mouth_image is not None:
        out_keypoints, out_classification_value = tongue_model.predict(mouth_image)
        _, _, y, x = max_average_point(out_keypoints, 5)
    else:
        out_classification_value = 0.0

    if out_classification_value > g.config["Tracking"]["Tongue"]["tongue_confidence"]:
        tongue_count += 1
        if tongue_count >= g.config["Tracking"]["Tongue"]["tongue_threshold"]:
            tongue_count = g.config["Tracking"]["Tongue"]["tongue_threshold"]
            tongue_out = 1.0
            tongue_x = float(-(x / 32 - 0.5) * g.config["Tracking"]["Tongue"]["tongue_x_scalar"])
            tongue_y = float(-(y / 32 - 0.5) * g.config["Tracking"]["Tongue"]["tongue_y_scalar"])
    else:
        tongue_count -= 1
        if tongue_count <= 0:
            tongue_count = 0
            tongue_out = 0.0
            tongue_x = 0.0
            tongue_y = 0.0
        else:
            tongue_out = g.data["BlendShapes"][52]["v"]
            tongue_x = g.data["BlendShapes"][62]["v"]
            tongue_y = g.data["BlendShapes"][63]["v"]

    if data["BlendShapes"][25]["v"] < g.config["Tracking"]["Tongue"]["mouth_close_threshold"]:
        tongue_out = 0.0
        tongue_x = 0.0
        tongue_y = 0.0

    return tongue_out, tongue_x, tongue_y
