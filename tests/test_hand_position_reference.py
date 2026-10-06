"""Hardware-free regressions using real tracking callbacks and synthetic landmarks.

Run with: python -m unittest discover -s tests -v
Only NumPy/SciPy are needed; camera, model and output dependencies are stubbed.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class HandPositionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.now_ms = 1000
        config = json.loads((ROOT / "settings/config.json").read_text(encoding="utf-8-sig"))
        data = json.loads((ROOT / "settings/data.json").read_text(encoding="utf-8-sig"))
        for name in ("Head", "Face", "Tongue", "Finger", "LeftController", "RightController"):
            config["Tracking"][name]["enable"] = False
        config["Tracking"]["Face"]["block"] = False
        config["Tracking"]["Hand"].update(
            enable=True, hand_detection_lower_threshold=0, enable_swap_strategy=False,
            enable_finger_action=False, only_front=False, hand_return_time=0.5,
        )
        config["Smoothing"]["enable"] = False
        self.g = module(
            "utils.globals", config=config, data=data, default_data=copy.deepcopy(data),
            latest_data=[0.0] * 129, start_time=0, face_landmarks=None,
            hand_landmarks=None, hand_position_reference=None,
            controller=NS(left_hand=NS(enable=False), right_hand=NS(enable=False)),
            hand_regression_model=NS(predict=Mock(return_value=np.asarray([0.05]))),
        )
        # Isolate hardware imports, including globals' device/network initialization.
        modules = {
            "utils": module("utils", __path__=[str(ROOT / "utils")]),
            "utils.globals": self.g,
            "cv2": module("cv2", getTickCount=lambda: self.now_ms, getTickFrequency=lambda: 1000),
            "onnxruntime": module("onnxruntime"),
            "joblib": module("joblib"),
            "tracker.hand.directml_hands": module("hands", DirectMLHands=Mock(), HAND_CONNECTIONS=[]),
            "tracker.face.directml_face": module("face", DirectMLFaceLandmarker=Mock()),
            "tracker.face.tongue": module("tongue", initialize_tongue_model=Mock(),
                                          mouth_roi_on_image=Mock(), detect_tongue=Mock()),
            "utils.sender": module("sender", data_send_thread=Mock()),
            "utils.smoothing": module("smoothing", apply_smoothing=Mock()),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.hand = load_source("test_hand", "tracker/hand/hand.py")
        self.face = load_source("test_face", "tracker/face/face.py")
        sys.modules["tracker.hand.hand"] = self.hand
        sys.modules["tracker.face.face"] = self.face
        self.tracking = load_source("test_tracking", "utils/tracking.py")

    def detect_face(self, present=True, depth=-0.1, timestamp_ms=None):
        landmarks = [NS(x=0.6, y=0.4, z=depth) for _ in range(478)]
        result = NS(face_landmarks=[landmarks] if present else [],
                    face_blendshapes=[], facial_transformation_matrixes=[])
        self.face.face_pred_handling(result, None,
                                     self.now_ms if timestamp_ms is None else timestamp_ms, None)

    def detect_hands(self):
        # Non-collinear wrist/thumb/pinky landmarks give a valid wrist rotation.
        points = [NS(x=0.4 + (i % 5) * 0.025, y=0.5 + (i // 5) * 0.03, z=0.01 + i * 0.002)
                  for i in range(21)]
        result = NS(multi_hand_landmarks=[NS(landmark=points)] * 2,
                    multi_hand_world_landmarks=[NS(landmark=points)] * 2,
                    multi_handedness=[NS(classification=[NS(label=side, score=1.0)])
                                      for side in ("Left", "Right")])
        self.hand.hand_pred_handling(result)

    def test_hand_only_dispatches_landmarks_without_head_or_blendshape_output(self):
        tracker = self.tracking.Tracker.__new__(self.tracking.Tracker)
        tracker.hand_worker, tracker.face_worker = Mock(), Mock()
        tracker.process_frame(np.zeros((2, 2, 3)))
        tracker.hand_worker.submit.assert_called_once()
        tracker.face_worker.submit.assert_called_once()
        self.g.face_detector = Mock()
        tracker._process_face_frame(np.zeros((2, 2, 3)), self.now_ms)
        options = self.g.face_detector.process_frame.call_args.kwargs
        self.assertFalse(options["output_transform"])
        self.assertFalse(options["output_blendshapes"])
        self.g.config["Tracking"]["Hand"]["enable"] = False
        tracker.face_worker.reset_mock()
        tracker.process_frame(np.zeros((2, 2, 3)))
        tracker.face_worker.submit.assert_not_called()

    def test_reference_loss_expiry_and_recovery(self):
        self.assertIsNone(self.hand._hand_position_reference())
        self.detect_face()
        anchor, depth = self.hand._hand_position_reference()
        np.testing.assert_allclose(anchor, [0.1, 0.1])
        self.assertEqual(depth, -0.1)
        self.now_ms += 501
        self.assertIsNone(self.hand._hand_position_reference())
        self.detect_face()
        self.assertIsNotNone(self.hand._hand_position_reference())
        self.detect_face(present=False)
        self.assertIsNone(self.g.face_landmarks)
        self.assertIsNone(self.hand._hand_position_reference())
        self.detect_face()
        self.assertIsNotNone(self.hand._hand_position_reference())

    def test_occluded_or_invalid_reference_is_rejected(self):
        for depth in (0.0, 0.1, float("nan"), float("inf")):
            with self.subTest(depth=depth):
                self.detect_face(depth=depth)
                self.assertIsNone(self.hand._hand_position_reference())
        self.detect_face(timestamp_ms=self.now_ms + 1)
        self.assertIsNone(self.hand._hand_position_reference())
        with patch.object(self.face, "is_hand_in_face", return_value=1.0):
            self.detect_face()
        self.assertIsNone(self.hand._hand_position_reference())

    def test_invalid_reference_depth_does_not_run_regression_model(self):
        for depth in (0.0, -0.004, 0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(depth=depth):
                self.assertIsNone(self.hand.get_fitted_hand_distance(np.zeros((21, 3)), depth))
        self.g.hand_regression_model.predict.assert_not_called()

    def test_valid_depth_mapping_is_preserved(self):
        cfg = self.g.config["Tracking"]["Hand"]
        expected = self.hand._soft_limit_depth(np.interp(
            (0.05 / -0.1 + cfg["z_shifting"]) * cfg["z_scalar"], [-2, 2], [-1.2, 1]))
        self.assertAlmostEqual(self.hand.get_fitted_hand_distance(np.zeros((21, 3)), -0.1), expected)
        self.g.hand_regression_model.predict.return_value = np.asarray([float("nan")])
        self.assertIsNone(self.hand.get_fitted_hand_distance(np.zeros((21, 3)), -0.1))

    def hand_positions(self, smoothing):
        if smoothing:
            return self.g.latest_data[70:73] + self.g.latest_data[76:79]
        return [value["v"] for side in ("Left", "Right")
                for value in self.g.data[side + "HandPosition"]]

    def test_recovered_reference_does_not_depend_on_older_smoothed_depth(self):
        for smoothing in (False, True):
            with self.subTest(smoothing=smoothing):
                self.g.config["Smoothing"]["enable"] = smoothing
                self.detect_face()
                self.g.data["HeadImagePosition"][2]["v"] = -0.1
                self.detect_hands()
                expected = self.hand_positions(smoothing)
                for stale_depth in (0.0, -0.3, float("nan")):
                    with self.subTest(stale_depth=stale_depth):
                        self.detect_face(present=False)
                        self.g.config["Tracking"]["Hand"]["hand_return_time"] = 0
                        self.detect_hands()
                        self.detect_face()
                        self.g.data["HeadImagePosition"][2]["v"] = stale_depth
                        self.detect_hands()
                        self.assertTrue(self.g.controller.left_hand.enable)
                        self.assertTrue(self.g.controller.right_hand.enable)
                        np.testing.assert_allclose(self.hand_positions(smoothing), expected)

    def test_new_face_frame_during_hand_processing_cannot_mix_snapshots(self):
        for smoothing in (False, True):
            with self.subTest(smoothing=smoothing):
                self.g.config["Smoothing"]["enable"] = smoothing
                self.detect_face()
                self.g.data["HeadImagePosition"][2]["v"] = -0.1
                self.detect_hands()
                expected = self.hand_positions(smoothing)

                def publish_newer_reference(_features):
                    # Simulate another worker publishing while the first hand's
                    # depth model runs, before processing the second hand.
                    self.g.hand_position_reference = (0.3, 0.2, -0.4, self.now_ms)
                    self.g.data["HeadImagePosition"][2]["v"] = -0.4
                    return np.asarray([0.05])

                with patch.object(self.g.hand_regression_model, "predict",
                                  side_effect=publish_newer_reference):
                    self.detect_hands()
                np.testing.assert_allclose(self.hand_positions(smoothing), expected)
                # A later hand frame should then consume the newly published reference.
                self.detect_hands()
                self.assertFalse(np.allclose(self.hand_positions(smoothing), expected))

    def test_invalid_face_depth_does_not_poison_smoothing_or_direct_state(self):
        for smoothing in (False, True):
            with self.subTest(smoothing=smoothing):
                self.g.config["Smoothing"]["enable"] = smoothing
                self.detect_face()
                before_data = copy.deepcopy(self.g.data["HeadImagePosition"])
                before_latest = self.g.latest_data[114:117]
                for depth in (float("nan"), float("inf"), 0.0, 0.1):
                    self.detect_face(depth=depth)
                    self.assertIsNone(self.g.hand_position_reference)
                    self.assertEqual(self.g.data["HeadImagePosition"], before_data)
                    self.assertEqual(self.g.latest_data[114:117], before_latest)
                self.detect_face(depth=-0.2)
                self.assertIsNotNone(self.hand._hand_position_reference())
                depth = (self.g.latest_data[116] if smoothing else
                         self.g.data["HeadImagePosition"][2]["v"])
                self.assertEqual(depth, -0.2)

    def test_tracking_loss_respects_return_delay_and_disabled_auto_reset(self):
        self.detect_face()
        with patch.object(self.hand.time, "monotonic", return_value=10.0):
            self.detect_hands()
        pose = copy.deepcopy(self.g.data["LeftHandPosition"])
        self.detect_face(present=False)
        with patch.object(self.hand.time, "monotonic", return_value=10.25):
            self.detect_hands()
        self.assertTrue(self.g.controller.left_hand.enable)
        self.assertEqual(self.g.data["LeftHandPosition"], pose)
        self.g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] = False
        with patch.object(self.hand.time, "monotonic", return_value=11.0):
            self.detect_hands()
        self.assertTrue(self.g.controller.left_hand.enable)
        self.g.config["Tracking"]["Hand"]["enable_hand_auto_reset"] = True
        with patch.object(self.hand.time, "monotonic", return_value=11.0):
            self.detect_hands()
        self.assertFalse(self.g.controller.left_hand.enable)

    def test_cold_start_loss_and_recovery_for_both_hands(self):
        head_before = copy.deepcopy(self.g.data["Position"])
        for smoothing in (False, True):
            with self.subTest(smoothing=smoothing):
                self.g.config["Smoothing"]["enable"] = smoothing
                self.g.hand_position_reference = None
                self.g.config["Tracking"]["Hand"]["hand_return_time"] = 0
                self.detect_hands()
                self.assertFalse(self.g.controller.left_hand.enable)
                self.assertFalse(self.g.controller.right_hand.enable)
                self.detect_face()
                self.detect_hands()
                self.assertTrue(self.g.controller.left_hand.enable)
                self.assertTrue(self.g.controller.right_hand.enable)
                self.detect_face(present=False)
                self.detect_hands()
                self.assertFalse(self.g.controller.left_hand.enable)
                self.assertFalse(self.g.controller.right_hand.enable)
                for side, index in (("Left", 70), ("Right", 76)):
                    expected = [v["v"] for v in self.g.default_data[side + "HandPosition"]]
                    actual = (self.g.latest_data[index:index + 3] if smoothing else
                              [v["v"] for v in self.g.data[side + "HandPosition"]])
                    self.assertEqual(actual, expected)
                self.detect_face()
                self.detect_hands()
                self.assertTrue(self.g.controller.left_hand.enable)
                self.assertTrue(self.g.controller.right_hand.enable)
        self.assertEqual(self.g.data["Position"], head_before)


if __name__ == "__main__":
    unittest.main()
