from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import onnxruntime as ort
import cv2
import numpy as np

from utils.mp_compat import Category, Landmark
from utils.model_provider import providers_for, session_options_for
from utils.ort_scheduler import ORT_PRIORITY_FACE, run_ort


FACE_BLENDSHAPE_LABELS = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


FACE_BLENDSHAPE_LANDMARKS = np.asarray(
    [
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
        40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70,
        78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107,
        109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150,
        152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
        168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234,
        246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284,
        285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311,
        312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338,
        356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
        381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398,
        400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471,
        472, 473, 474, 475, 476, 477,
    ],
    dtype=np.int32,
)

_FACE_GEOMETRY_CACHE: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
] | None = None


@dataclasses.dataclass
class Rect:
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float


@dataclasses.dataclass
class Detection:
    score: float
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray


def _normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _session(path: Path, provider: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), session_options_for(provider), providers=providers_for(provider))


def _generate_face_anchors() -> np.ndarray:
    input_size = 128
    strides = [8, 16, 16, 16]
    min_scale = 0.1484375
    max_scale = 0.75
    anchors: list[tuple[float, float, float, float]] = []
    layer_id = 0
    while layer_id < len(strides):
        anchors_per_location = 0
        last_same_stride_layer = layer_id
        while last_same_stride_layer < len(strides) and strides[last_same_stride_layer] == strides[layer_id]:
            scale = min_scale + (max_scale - min_scale) * last_same_stride_layer / (len(strides) - 1.0)
            scale_next = (
                1.0
                if last_same_stride_layer == len(strides) - 1
                else min_scale + (max_scale - min_scale) * (last_same_stride_layer + 1) / (len(strides) - 1.0)
            )
            del scale, scale_next
            anchors_per_location += 2
            last_same_stride_layer += 1

        feature_size = math.ceil(input_size / strides[layer_id])
        for y in range(feature_size):
            for x in range(feature_size):
                for _ in range(anchors_per_location):
                    anchors.append(((x + 0.5) / feature_size, (y + 0.5) / feature_size, 1.0, 1.0))
        layer_id = last_same_stride_layer
    return np.asarray(anchors, dtype=np.float32)


def _letterbox(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, float, int, int]:
    h, w = image_rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=image_rgb.dtype)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    out[int(pad_y) : int(pad_y) + new_h, int(pad_x) : int(pad_x) + new_w] = resized
    return out, pad_x, pad_y, new_w, new_h


def _preprocess_nchw(image_rgb: np.ndarray, norm: str) -> np.ndarray:
    x = image_rgb.astype(np.float32)
    if norm == "minus_one_to_one":
        x = (x - 127.5) / 127.5
    else:
        x /= 255.0
    return np.transpose(x, (2, 0, 1))[None].astype(np.float32)


def _preprocess_nhwc(image_rgb: np.ndarray) -> np.ndarray:
    return (image_rgb.astype(np.float32) / 255.0)[None].astype(np.float32)


def _decode_detections(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    pad_x: float,
    pad_y: float,
    content_w: int,
    content_h: int,
    min_score: float,
) -> list[Detection]:
    raw_boxes = raw_boxes.reshape(-1, 16)
    raw_scores = np.clip(raw_scores.reshape(-1), -100.0, 100.0)
    scores = _sigmoid(raw_scores)
    keep = np.flatnonzero(scores >= min_score)
    if keep.size == 0:
        return []

    boxes = raw_boxes[keep]
    kept_anchors = anchors[keep]
    scores = scores[keep]
    x_center = boxes[:, 0] / 128.0 + kept_anchors[:, 0]
    y_center = boxes[:, 1] / 128.0 + kept_anchors[:, 1]
    width = boxes[:, 2] / 128.0
    height = boxes[:, 3] / 128.0

    def project_x(x: np.ndarray) -> np.ndarray:
        return (x * 128.0 - pad_x) / content_w

    def project_y(y: np.ndarray) -> np.ndarray:
        return (y * 128.0 - pad_y) / content_h

    keypoints = np.zeros((len(keep), 6, 2), dtype=np.float32)
    for k in range(6):
        off = 4 + k * 2
        keypoints[:, k, 0] = boxes[:, off] / 128.0 + kept_anchors[:, 0]
        keypoints[:, k, 1] = boxes[:, off + 1] / 128.0 + kept_anchors[:, 1]

    detections: list[Detection] = []
    for i in range(len(keep)):
        xmin = project_x(x_center[i] - width[i] * 0.5)
        ymin = project_y(y_center[i] - height[i] * 0.5)
        xmax = project_x(x_center[i] + width[i] * 0.5)
        ymax = project_y(y_center[i] + height[i] * 0.5)
        if xmax <= xmin or ymax <= ymin:
            continue
        kps = np.stack((project_x(keypoints[i, :, 0]), project_y(keypoints[i, :, 1])), axis=1)
        detections.append(Detection(float(scores[i]), (float(xmin), float(ymin), float(xmax), float(ymax)), kps))
    return detections


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def _weighted_nms(detections: list[Detection], threshold: float) -> list[Detection]:
    remaining = sorted(detections, key=lambda d: d.score, reverse=True)
    picked: list[Detection] = []
    while remaining and not picked:
        base = remaining[0]
        overlapping = [d for d in remaining if _iou(base.bbox, d.bbox) > threshold]
        if len(overlapping) == 1:
            picked.append(base)
            break
        weights = np.asarray([d.score for d in overlapping], dtype=np.float32)
        weight_sum = float(weights.sum())
        bboxes = np.asarray([d.bbox for d in overlapping], dtype=np.float32)
        kps = np.asarray([d.keypoints for d in overlapping], dtype=np.float32)
        picked.append(
            Detection(
                base.score,
                tuple((bboxes * weights[:, None]).sum(axis=0) / weight_sum),
                (kps * weights[:, None, None]).sum(axis=0) / weight_sum,
            )
        )
    return picked


def _rect_from_detection(detection: Detection, image_w: int, image_h: int) -> Rect:
    xmin, ymin, xmax, ymax = detection.bbox
    x0 = float(detection.keypoints[0, 0] * image_w)
    y0 = float(detection.keypoints[0, 1] * image_h)
    x1 = float(detection.keypoints[1, 0] * image_w)
    y1 = float(detection.keypoints[1, 1] * image_h)
    rotation = _normalize_radians(math.atan2(y1 - y0, x1 - x0))
    return Rect(
        x_center=(xmin + xmax) * 0.5,
        y_center=(ymin + ymax) * 0.5,
        width=xmax - xmin,
        height=ymax - ymin,
        rotation=rotation,
    )


def _transform_rect(rect: Rect, image_w: int, image_h: int, scale: float) -> Rect:
    long_side = max(rect.width * image_w, rect.height * image_h)
    return Rect(
        rect.x_center,
        rect.y_center,
        long_side / image_w * scale,
        long_side / image_h * scale,
        rect.rotation,
    )


def _crop_rect(image_rgb: np.ndarray, rect: Rect, size: int) -> np.ndarray:
    image_h, image_w = image_rgb.shape[:2]
    a = rect.width * image_w
    b = rect.height * image_h
    c = math.cos(rect.rotation)
    d = math.sin(rect.rotation)
    e = rect.x_center * image_w
    f = rect.y_center * image_h

    def src_point(u: float, v: float) -> tuple[float, float]:
        x = u * a * c + v * (-b * d) + (-0.5 * a * c + 0.5 * b * d + e)
        y = u * a * d + v * (b * c) + (-0.5 * b * c - 0.5 * a * d + f)
        return x, y

    src = np.asarray([src_point(0.0, 0.0), src_point(1.0, 0.0), src_point(0.0, 1.0)], dtype=np.float32)
    dst = np.asarray([[0.0, 0.0], [size - 1.0, 0.0], [0.0, size - 1.0]], dtype=np.float32)
    return cv2.warpAffine(
        image_rgb,
        cv2.getAffineTransform(src, dst),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _project_landmarks(crop_landmarks: np.ndarray, rect: Rect, image_w: int, image_h: int) -> np.ndarray:
    a = rect.width * image_w
    b = rect.height * image_h
    c = math.cos(rect.rotation)
    d = math.sin(rect.rotation)
    e = rect.x_center * image_w
    f = rect.y_center * image_h
    out = crop_landmarks.copy()
    x = crop_landmarks[:, 0]
    y = crop_landmarks[:, 1]
    out[:, 0] = x * (a * c / image_w) + y * (-b * d / image_w) + (
        -0.5 * a * c + 0.5 * b * d + e
    ) / image_w
    out[:, 1] = x * (a * d / image_h) + y * (b * c / image_h) + (
        -0.5 * b * c - 0.5 * a * d + f
    ) / image_h
    out[:, 2] = crop_landmarks[:, 2] * (a / image_w)
    return out


def _rect_from_landmarks(landmarks: np.ndarray, image_w: int, image_h: int) -> Rect:
    x0 = float(np.min(landmarks[:, 0]))
    x1 = float(np.max(landmarks[:, 0]))
    y0 = float(np.min(landmarks[:, 1]))
    y1 = float(np.max(landmarks[:, 1]))
    eye_r = landmarks[33]
    eye_l = landmarks[263]
    rotation = _normalize_radians(math.atan2((eye_l[1] - eye_r[1]) * image_h, (eye_l[0] - eye_r[0]) * image_w))
    return Rect((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0, rotation)


def _landmarks_to_objects(points: np.ndarray) -> list[Landmark]:
    return [Landmark(float(x), float(y), float(z)) for x, y, z in points]


def _blendshape_input(landmarks: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    points = landmarks[FACE_BLENDSHAPE_LANDMARKS, :2].copy()
    points[:, 0] *= image_w
    points[:, 1] *= image_h
    return points[None].astype(np.float32)


def _categories(scores: np.ndarray) -> list[Category]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return [
        Category(index=i, score=float(scores[i]), category_name=FACE_BLENDSHAPE_LABELS[i])
        for i in range(min(len(scores), len(FACE_BLENDSHAPE_LABELS)))
    ]


def _load_face_geometry_metadata() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    global _FACE_GEOMETRY_CACHE
    if _FACE_GEOMETRY_CACHE is None:
        path = Path(__file__).resolve().parents[2] / "modules" / "face_geometry_metadata.npz"
        data = np.load(path)
        canonical_metric_landmarks = data["canonical_metric_landmarks"].astype(np.float32).T
        landmark_weights = data["landmark_weights"].astype(np.float32)
        sqrt_weights = np.sqrt(landmark_weights).astype(np.float32)
        weighted_sources = canonical_metric_landmarks * sqrt_weights[None, :]
        total_weight = float(np.sum(landmark_weights))
        source_center = np.sum(
            weighted_sources * sqrt_weights[None, :],
            axis=1,
        ) / total_weight
        centered_weighted_sources = (
            weighted_sources - source_center[:, None] * sqrt_weights[None, :]
        )
        denominator = float(np.sum(centered_weighted_sources * weighted_sources))
        _FACE_GEOMETRY_CACHE = (
            canonical_metric_landmarks,
            landmark_weights,
            sqrt_weights,
            weighted_sources,
            centered_weighted_sources,
            denominator,
            total_weight,
        )
    return _FACE_GEOMETRY_CACHE


def _solve_canonical_weighted_orthogonal_problem(
    target_points: np.ndarray,
    geometry_metadata: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ],
) -> np.ndarray:
    (
        _canonical_metric_landmarks,
        _landmark_weights,
        sqrt_weights,
        weighted_sources,
        centered_weighted_sources,
        denominator,
        total_weight,
    ) = geometry_metadata
    weighted_targets = target_points * sqrt_weights[None, :]

    design_matrix = weighted_targets @ centered_weighted_sources.T
    postrotation, _singular_values, prerotation = np.linalg.svd(design_matrix, full_matrices=True)
    if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0.0:
        postrotation[:, 2] *= -1.0
    rotation = postrotation @ prerotation

    rotated_centered_sources = rotation @ centered_weighted_sources
    if denominator <= 1e-9:
        raise ValueError("Face geometry scale denominator is too small")
    scale = float(np.sum(rotated_centered_sources * weighted_targets) / denominator)
    if scale <= 1e-9:
        raise ValueError("Face geometry scale is too small")

    rotation_and_scale = scale * rotation
    pointwise_diffs = weighted_targets - rotation_and_scale @ weighted_sources
    translation = np.sum(pointwise_diffs * sqrt_weights[None, :], axis=1) / total_weight

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_and_scale
    transform[:3, 3] = translation
    return transform


def _estimate_geometry_transform(landmarks: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    geometry_metadata = _load_face_geometry_metadata()
    canonical_metric_landmarks = geometry_metadata[0]
    screen_landmarks = landmarks[: canonical_metric_landmarks.shape[1], :3].T.astype(np.float32)

    near = 1.0
    height_at_near = 2.0 * near * math.tan(0.5 * math.pi / 180.0 * 63.0)
    width_at_near = image_w * height_at_near / image_h
    left = -0.5 * width_at_near
    bottom = -0.5 * height_at_near

    screen_landmarks[1] = 1.0 - screen_landmarks[1]
    screen_landmarks *= np.asarray([[width_at_near], [height_at_near], [width_at_near]], dtype=np.float32)
    screen_landmarks += np.asarray([[left], [bottom], [0.0]], dtype=np.float32)
    depth_offset = float(np.mean(screen_landmarks[2]))

    intermediate = screen_landmarks.copy()
    intermediate[2] *= -1.0
    first_scale = float(
        np.linalg.norm(
            _solve_canonical_weighted_orthogonal_problem(
                intermediate, geometry_metadata
            )[:3, 0]
        )
    )

    intermediate = screen_landmarks.copy()
    intermediate[2] = (intermediate[2] - depth_offset + near) / first_scale
    intermediate[0] = intermediate[0] * intermediate[2] / near
    intermediate[1] = intermediate[1] * intermediate[2] / near
    intermediate[2] *= -1.0
    second_scale = float(
        np.linalg.norm(
            _solve_canonical_weighted_orthogonal_problem(
                intermediate, geometry_metadata
            )[:3, 0]
        )
    )

    total_scale = first_scale * second_scale
    metric_landmarks = screen_landmarks.copy()
    metric_landmarks[2] = (metric_landmarks[2] - depth_offset + near) / total_scale
    metric_landmarks[0] = metric_landmarks[0] * metric_landmarks[2] / near
    metric_landmarks[1] = metric_landmarks[1] * metric_landmarks[2] / near
    metric_landmarks[2] *= -1.0

    return _solve_canonical_weighted_orthogonal_problem(
        metric_landmarks, geometry_metadata
    )


def _estimate_transform_fallback(landmarks: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    if len(landmarks) <= 263:
        return mat
    eye_r = landmarks[33]
    eye_l = landmarks[263]
    forehead = landmarks[10]
    chin = landmarks[152]
    nose = landmarks[4]

    x_axis = eye_l - eye_r
    y_axis = chin - forehead
    x_axis /= max(np.linalg.norm(x_axis), 1e-6)
    y_axis /= max(np.linalg.norm(y_axis), 1e-6)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(np.linalg.norm(z_axis), 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-6)
    mat[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    mat[0, 3] = (nose[0] - 0.5) * 2.0
    mat[1, 3] = (nose[1] - 0.5) * 2.0
    mat[2, 3] = nose[2]
    return mat


def _estimate_transform(landmarks: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    if len(landmarks) >= 468:
        try:
            return _estimate_geometry_transform(landmarks, image_w, image_h)
        except Exception:
            return _estimate_transform_fallback(landmarks)
    return _estimate_transform_fallback(landmarks)


class DirectMLFaceLandmarker:
    def __init__(
        self,
        result_callback: Callable[[SimpleNamespace, np.ndarray, int], None],
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        provider: str = "GPU",
    ) -> None:
        module_dir = Path(__file__).resolve().parents[2] / "modules"
        self.detector = _session(module_dir / "face_detector.onnx", provider)
        self.landmark = _session(module_dir / "face_landmarks_detector.onnx", provider)
        self.blendshape = _session(module_dir / "face_blendshapes.onnx", provider)
        self.model_provider = provider
        self.detector_input = self.detector.get_inputs()[0].name
        self.landmark_input = self.landmark.get_inputs()[0].name
        self.blendshape_input = self.blendshape.get_inputs()[0].name
        self.anchors = _generate_face_anchors()
        self.result_callback = result_callback
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.prev_rect: Rect | None = None

    @property
    def providers(self) -> tuple[list[str], list[str], list[str]]:
        return self.detector.get_providers(), self.landmark.get_providers(), self.blendshape.get_providers()

    def close(self) -> None:
        self.prev_rect = None

    def process_frame(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int,
        output_blendshapes: bool = True,
        output_transform: bool = True,
    ) -> None:
        result = self.detect(
            image_rgb,
            output_blendshapes=output_blendshapes,
            output_transform=output_transform,
        )
        self.result_callback(result, image_rgb, timestamp_ms)

    def _detect_rect(self, image_rgb: np.ndarray) -> Rect | None:
        image_h, image_w = image_rgb.shape[:2]
        det_img, pad_x, pad_y, content_w, content_h = _letterbox(image_rgb, 128)
        raw_boxes, raw_scores = run_ort(
            self.detector,
            None,
            {self.detector_input: _preprocess_nchw(det_img, "minus_one_to_one")},
            priority=ORT_PRIORITY_FACE,
        )
        detections = _decode_detections(
            raw_boxes,
            raw_scores,
            self.anchors,
            pad_x,
            pad_y,
            content_w,
            content_h,
            self.min_detection_confidence,
        )
        detections = _weighted_nms(detections, 0.3)
        if not detections:
            return None
        return _transform_rect(_rect_from_detection(detections[0], image_w, image_h), image_w, image_h, 1.5)

    def detect(
        self,
        image_rgb: np.ndarray,
        output_blendshapes: bool = True,
        output_transform: bool = True,
    ):
        image_h, image_w = image_rgb.shape[:2]
        rect = self.prev_rect if self.prev_rect is not None else self._detect_rect(image_rgb)
        if rect is None:
            self.prev_rect = None
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])

        crop = _crop_rect(image_rgb, rect, 256)
        landmarks_raw, presence_raw, _extra = run_ort(
            self.landmark,
            None,
            {self.landmark_input: _preprocess_nhwc(crop)},
            priority=ORT_PRIORITY_FACE,
        )
        presence = float(_sigmoid(presence_raw.reshape(-1))[0])
        if presence < self.min_presence_confidence:
            self.prev_rect = None
            return SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])

        crop_landmarks = landmarks_raw.reshape(478, 3).astype(np.float32)
        crop_landmarks[:, 0] /= 256.0
        crop_landmarks[:, 1] /= 256.0
        crop_landmarks[:, 2] /= 256.0
        landmarks = _project_landmarks(crop_landmarks, rect, image_w, image_h)
        self.prev_rect = _transform_rect(_rect_from_landmarks(landmarks, image_w, image_h), image_w, image_h, 1.5)

        face_blendshapes = []
        if output_blendshapes:
            blend_scores = run_ort(
                self.blendshape,
                None,
                {self.blendshape_input: _blendshape_input(landmarks, image_w, image_h)},
                priority=ORT_PRIORITY_FACE,
            )[0]
            face_blendshapes = [_categories(blend_scores)]

        facial_transformation_matrixes = []
        if output_transform:
            facial_transformation_matrixes = [_estimate_transform(landmarks, image_w, image_h)]

        return SimpleNamespace(
            face_landmarks=[_landmarks_to_objects(landmarks)],
            face_blendshapes=face_blendshapes,
            facial_transformation_matrixes=facial_transformation_matrixes,
        )
