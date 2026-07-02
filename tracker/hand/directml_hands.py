from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import onnxruntime as ort
import cv2
import numpy as np
from utils.mp_compat import Classification, ClassificationList, Landmark, LandmarkList, NormalizedLandmarkList
from utils.model_provider import providers_for, session_options_for
from utils.ort_scheduler import ORT_PRIORITY_HAND, run_ort


HAND_CONNECTIONS = (
    (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


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


@dataclasses.dataclass
class HandResult:
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    next_rect: Rect
    handedness: str
    handedness_score: float


def _normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_anchors() -> np.ndarray:
    input_w = 192
    input_h = 192
    strides = [8, 16, 16, 16]
    min_scale = 0.1484375
    max_scale = 0.75
    anchors: list[tuple[float, float, float, float]] = []
    layer_id = 0
    while layer_id < len(strides):
        aspect_ratios: list[float] = []
        scales: list[float] = []
        last_same_stride_layer = layer_id
        while (
            last_same_stride_layer < len(strides)
            and strides[last_same_stride_layer] == strides[layer_id]
        ):
            scale = min_scale + (max_scale - min_scale) * last_same_stride_layer / (len(strides) - 1.0)
            aspect_ratios.append(1.0)
            scales.append(scale)
            scale_next = (
                1.0
                if last_same_stride_layer == len(strides) - 1
                else min_scale + (max_scale - min_scale) * (last_same_stride_layer + 1) / (len(strides) - 1.0)
            )
            aspect_ratios.append(1.0)
            scales.append(math.sqrt(scale * scale_next))
            last_same_stride_layer += 1

        feature_h = math.ceil(input_h / strides[layer_id])
        feature_w = math.ceil(input_w / strides[layer_id])
        for y in range(feature_h):
            for x in range(feature_w):
                for _ in range(len(aspect_ratios)):
                    anchors.append(((x + 0.5) / feature_w, (y + 0.5) / feature_h, 1.0, 1.0))
        layer_id = last_same_stride_layer
    return np.asarray(anchors, dtype=np.float32)


def _preprocess_image(image_rgb: np.ndarray, size: int) -> np.ndarray:
    x = image_rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)


def _letterbox(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, float, int, int]:
    h, w = image_rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=image_rgb.dtype)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    left = int(math.floor(pad_x))
    top = int(math.floor(pad_y))
    out[top : top + new_h, left : left + new_w] = resized
    return out, pad_x, pad_y, new_w, new_h


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
    raw_boxes = raw_boxes.reshape(-1, 18)
    raw_scores = np.clip(raw_scores.reshape(-1), -100.0, 100.0)
    scores = _sigmoid(raw_scores)
    keep = np.flatnonzero(scores >= min_score)
    if keep.size == 0:
        return []

    boxes = raw_boxes[keep]
    kept_anchors = anchors[keep]
    scores = scores[keep]
    x_center = boxes[:, 0] / 192.0 * kept_anchors[:, 2] + kept_anchors[:, 0]
    y_center = boxes[:, 1] / 192.0 * kept_anchors[:, 3] + kept_anchors[:, 1]
    width = boxes[:, 2] / 192.0 * kept_anchors[:, 2]
    height = boxes[:, 3] / 192.0 * kept_anchors[:, 3]

    def project_x(x: np.ndarray) -> np.ndarray:
        return (x * 192.0 - pad_x) / content_w

    def project_y(y: np.ndarray) -> np.ndarray:
        return (y * 192.0 - pad_y) / content_h

    keypoints = np.zeros((len(keep), 7, 2), dtype=np.float32)
    for k in range(7):
        off = 4 + k * 2
        keypoints[:, k, 0] = boxes[:, off] / 192.0 * kept_anchors[:, 2] + kept_anchors[:, 0]
        keypoints[:, k, 1] = boxes[:, off + 1] / 192.0 * kept_anchors[:, 3] + kept_anchors[:, 1]

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


def _rect_to_bbox(rect: Rect) -> tuple[float, float, float, float]:
    return (
        rect.x_center - rect.width * 0.5,
        rect.y_center - rect.height * 0.5,
        rect.x_center + rect.width * 0.5,
        rect.y_center + rect.height * 0.5,
    )


def _weighted_nms(detections: list[Detection], threshold: float, max_hands: int) -> list[Detection]:
    remaining = sorted(detections, key=lambda detection: detection.score, reverse=True)
    picked: list[Detection] = []
    while remaining and len(picked) < max_hands:
        base = remaining[0]
        overlapping = [d for d in remaining if _iou(base.bbox, d.bbox) > threshold]
        remaining = [d for d in remaining if _iou(base.bbox, d.bbox) <= threshold]
        if len(overlapping) == 1:
            picked.append(base)
            continue
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
    x1 = float(detection.keypoints[2, 0] * image_w)
    y1 = float(detection.keypoints[2, 1] * image_h)
    return Rect(
        x_center=(xmin + xmax) * 0.5,
        y_center=(ymin + ymax) * 0.5,
        width=xmax - xmin,
        height=ymax - ymin,
        rotation=_normalize_radians(math.pi * 0.5 - math.atan2(-(y1 - y0), x1 - x0)),
    )


def _transform_rect(
    rect: Rect,
    image_w: int,
    image_h: int,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> Rect:
    width = rect.width
    height = rect.height
    rotation = rect.rotation
    if rotation == 0.0:
        x_center = rect.x_center + width * shift_x
        y_center = rect.y_center + height * shift_y
    else:
        x_shift = (
            image_w * width * shift_x * math.cos(rotation)
            - image_h * height * shift_y * math.sin(rotation)
        ) / image_w
        y_shift = (
            image_w * width * shift_x * math.sin(rotation)
            + image_h * height * shift_y * math.cos(rotation)
        ) / image_h
        x_center = rect.x_center + x_shift
        y_center = rect.y_center + y_shift

    long_side = max(width * image_w, height * image_h)
    width = long_side / image_w
    height = long_side / image_h
    return Rect(x_center, y_center, width * scale_x, height * scale_y, rotation)


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
    z_scale = math.sqrt((a * c / image_w) ** 2 + (a * d / image_h) ** 2)
    out[:, 2] = crop_landmarks[:, 2] * z_scale
    return out


def _project_world_landmarks(world_landmarks: np.ndarray, rect: Rect) -> np.ndarray:
    c = math.cos(rect.rotation)
    s = math.sin(rect.rotation)
    out = world_landmarks.copy()
    x = world_landmarks[:, 0]
    y = world_landmarks[:, 1]
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _rect_from_landmarks(landmarks: np.ndarray, image_w: int, image_h: int) -> Rect:
    partial_idx = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    lms = landmarks[partial_idx]
    x0 = landmarks[0, 0] * image_w
    y0 = landmarks[0, 1] * image_h
    x1 = ((landmarks[6, 0] + landmarks[14, 0]) * 0.5 + landmarks[10, 0]) * 0.5 * image_w
    y1 = ((landmarks[6, 1] + landmarks[14, 1]) * 0.5 + landmarks[10, 1]) * 0.5 * image_h
    rotation = _normalize_radians(math.pi * 0.5 - math.atan2(-(y1 - y0), x1 - x0))
    reverse_angle = _normalize_radians(-rotation)

    min_x = float(np.min(lms[:, 0]))
    max_x = float(np.max(lms[:, 0]))
    min_y = float(np.min(lms[:, 1]))
    max_y = float(np.max(lms[:, 1]))
    axis_x = (min_x + max_x) * 0.5
    axis_y = (min_y + max_y) * 0.5

    original_x = (lms[:, 0] - axis_x) * image_w
    original_y = (lms[:, 1] - axis_y) * image_h
    projected_x = original_x * math.cos(reverse_angle) - original_y * math.sin(reverse_angle)
    projected_y = original_x * math.sin(reverse_angle) + original_y * math.cos(reverse_angle)
    min_px = float(np.min(projected_x))
    max_px = float(np.max(projected_x))
    min_py = float(np.min(projected_y))
    max_py = float(np.max(projected_y))
    center_px = (min_px + max_px) * 0.5
    center_py = (min_py + max_py) * 0.5

    center_x = center_px * math.cos(rotation) - center_py * math.sin(rotation) + image_w * axis_x
    center_y = center_px * math.sin(rotation) + center_py * math.cos(rotation) + image_h * axis_y
    return Rect(
        x_center=center_x / image_w,
        y_center=center_y / image_h,
        width=(max_px - min_px) / image_w,
        height=(max_py - min_py) / image_h,
        rotation=rotation,
    )


def _associate_rects(
    base_rects: list[Rect],
    detected_rects: list[tuple[Rect, float]],
    min_similarity_threshold: float,
    max_hands: int,
) -> list[tuple[Rect, float]]:
    result: list[tuple[Rect, float]] = [(rect, 0.0) for rect in base_rects]
    for rect, score in detected_rects:
        overlaps = any(
            _iou(_rect_to_bbox(rect), _rect_to_bbox(existing)) > min_similarity_threshold
            for existing, _ in result
        )
        if not overlaps:
            result.append((rect, score))
        if len(result) >= max_hands:
            break
    return result[:max_hands]


def _to_normalized_landmark_list(points: np.ndarray) -> NormalizedLandmarkList:
    landmark_list = NormalizedLandmarkList()
    for x, y, z in points:
        landmark_list.landmark.append(Landmark(float(x), float(y), float(z)))
    return landmark_list


def _to_world_landmark_list(points: np.ndarray) -> LandmarkList:
    landmark_list = LandmarkList()
    for x, y, z in points:
        landmark_list.landmark.append(Landmark(float(x), float(y), float(z)))
    return landmark_list


def _to_classification_list(label: str, score: float) -> ClassificationList:
    return ClassificationList([Classification(1 if label == "Right" else 0, float(score), label)])


class DirectMLHands:
    """Small MediaPipe Hands-compatible wrapper backed by ONNX Runtime DirectML."""

    def __init__(
        self,
        model_complexity: int = 0,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        provider: str = "GPU",
    ) -> None:
        module_dir = Path(__file__).resolve().parents[2] / "modules"
        detector_path = module_dir / "hand_detector_tflite2onnx_dml.onnx"
        landmark_paths = {
            0: module_dir / "hand_landmark_lite_tflite2onnx.onnx",
            1: module_dir / "hand_landmarks_detector_tflite2onnx.onnx",
        }
        if model_complexity not in landmark_paths:
            raise ValueError(f"Unsupported hand model_complexity: {model_complexity}")
        landmark_path = landmark_paths[model_complexity]
        if not detector_path.exists() or not landmark_path.exists():
            raise FileNotFoundError(
                "DirectML hand models are missing. Expected "
                f"{detector_path} and {landmark_path}."
            )

        session_options = session_options_for(provider)
        providers = providers_for(provider)
        self.detector = ort.InferenceSession(
            str(detector_path),
            session_options,
            providers=providers,
        )
        self.landmark = ort.InferenceSession(
            str(landmark_path),
            session_options,
            providers=providers,
        )
        self.model_provider = provider
        self.model_complexity = model_complexity
        self.detector_input = self.detector.get_inputs()[0].name
        self.landmark_input = self.landmark.get_inputs()[0].name
        self.anchors = _generate_anchors()
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_presence_confidence = min_presence_confidence
        self.prev_rects: list[Rect] = []
        self.detector_refresh_interval = 10
        self._frames_since_detector = self.detector_refresh_interval

    @property
    def providers(self) -> tuple[list[str], list[str]]:
        return self.detector.get_providers(), self.landmark.get_providers()

    def close(self) -> None:
        self.prev_rects = []
        self._frames_since_detector = self.detector_refresh_interval

    def _detect(self, image_rgb: np.ndarray) -> list[tuple[Rect, float]]:
        det_img, pad_x, pad_y, content_w, content_h = _letterbox(image_rgb, 192)
        raw_boxes, raw_scores = run_ort(
            self.detector,
            None,
            {self.detector_input: _preprocess_image(det_img, 192)},
            priority=ORT_PRIORITY_HAND,
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
        detections = _weighted_nms(detections, 0.3, self.max_num_hands)
        image_h, image_w = image_rgb.shape[:2]
        rects: list[tuple[Rect, float]] = []
        for detection in detections:
            palm_rect = _rect_from_detection(detection, image_w, image_h)
            rects.append((_transform_rect(palm_rect, image_w, image_h, 2.6, 2.6, 0.0, -0.5), detection.score))
        return rects

    def _run_landmark(self, image_rgb: np.ndarray, rect: Rect) -> HandResult | None:
        image_h, image_w = image_rgb.shape[:2]
        crop = _crop_rect(image_rgb, rect, 224)
        landmarks_raw, presence_raw, handedness_raw, world_raw = run_ort(
            self.landmark,
            None,
            {self.landmark_input: _preprocess_image(crop, 224)},
            priority=ORT_PRIORITY_HAND,
        )
        presence = float(presence_raw.reshape(-1)[0])
        if presence < self.min_presence_confidence:
            return None

        crop_landmarks = landmarks_raw.reshape(21, 3).astype(np.float32)
        crop_landmarks[:, 0] /= 224.0
        crop_landmarks[:, 1] /= 224.0
        crop_landmarks[:, 2] = crop_landmarks[:, 2] / 224.0 / 0.4
        landmarks = _project_landmarks(crop_landmarks, rect, image_w, image_h)

        world_landmarks = world_raw.reshape(21, 3).astype(np.float32)
        world_landmarks = _project_world_landmarks(world_landmarks, rect)

        raw_handedness = float(handedness_raw.reshape(-1)[0])
        if raw_handedness >= 0.5:
            handedness = "Left"
            handedness_score = raw_handedness
        else:
            handedness = "Right"
            handedness_score = 1.0 - raw_handedness

        next_rect = _transform_rect(
            _rect_from_landmarks(landmarks, image_w, image_h),
            image_w,
            image_h,
            2.0,
            2.0,
            0.0,
            -0.1,
        )
        return HandResult(landmarks, world_landmarks, next_rect, handedness, handedness_score)

    def process_frame(self, image_rgb: np.ndarray):
        results: list[HandResult] = []

        for rect in self.prev_rects[: self.max_num_hands]:
            result = self._run_landmark(image_rgb, rect)
            if result is not None:
                results.append(result)

        needs_detector = not results
        if results:
            self._frames_since_detector += 1
            needs_detector = (
                len(results) < self.max_num_hands
                and self._frames_since_detector >= self.detector_refresh_interval
            )

        if needs_detector:
            detected_rects = self._detect(image_rgb)
            self._frames_since_detector = 0
            tracked_rects = [result.next_rect for result in results]
            rects_with_scores = _associate_rects(
                tracked_rects,
                detected_rects,
                self.min_tracking_confidence,
                self.max_num_hands,
            )
            new_rects = rects_with_scores[len(tracked_rects) :]
            for rect, _score in new_rects:
                result = self._run_landmark(image_rgb, rect)
                if result is not None:
                    results.append(result)

        self.prev_rects = [result.next_rect for result in results]
        if not results:
            return SimpleNamespace(
                multi_hand_landmarks=None,
                multi_hand_world_landmarks=None,
                multi_handedness=None,
            )

        return SimpleNamespace(
            multi_hand_landmarks=[_to_normalized_landmark_list(result.landmarks) for result in results],
            multi_hand_world_landmarks=[_to_world_landmark_list(result.world_landmarks) for result in results],
            multi_handedness=[
                _to_classification_list(result.handedness, result.handedness_score) for result in results
            ],
        )
