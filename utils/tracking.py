import threading
import onnxruntime as _onnxruntime_preload
import cv2
from tracker.face.tongue import initialize_tongue_model
from tracker.face.face import initialize_face
from tracker.hand.hand import initialize_hand,hand_pred_handling,initialize_hand_depth
from utils.sender import data_send_thread
from utils.smoothing import apply_smoothing
import utils.globals as g


class LatestFrameWorker:
    def __init__(self, name, process_fn):
        self._process_fn = process_fn
        self._condition = threading.Condition()
        self._latest_args = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name=name)
        self._thread.start()

    def submit(self, *args):
        with self._condition:
            if self._stop:
                return
            self._latest_args = args
            self._condition.notify()

    def close(self):
        with self._condition:
            if self._stop:
                return
            self._stop = True
            self._latest_args = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join()

    def join(self):
        with self._condition:
            while self._busy or self._latest_args is not None:
                self._condition.wait()

    def _worker_loop(self):
        while True:
            with self._condition:
                while self._latest_args is None and not self._stop:
                    self._condition.wait()
                if self._stop and self._latest_args is None:
                    return
                item = self._latest_args
                self._latest_args = None
                self._busy = True
            try:
                self._process_fn(*item)
            except Exception as exc:
                print(f"{threading.current_thread().name} error: {exc}")
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()


class Tracker:
    def __init__(self):
        self.is_running = True
        self.image = None

        provider = g.config["Model"]["provider"]
        hand_model_complexity = g.config["Model"]["Hand"]["model_complexity"]
        if self._vision_config_changed(provider, hand_model_complexity):
            self._close_vision_models()

        if g.face_detector is None or g.hand_detector is None or g.tongue_model is None:
            print("Initializing tongue model")
            g.tongue_model = initialize_tongue_model()
            print(f"Initializing ONNX Runtime vision ({provider})")
            g.face_detector = initialize_face(g.tongue_model)
            g.hand_detector = initialize_hand()
            g.hand_feature_model, g.hand_regression_model = initialize_hand_depth()

        self.hand_worker = LatestFrameWorker("HandWorker", self._process_hand_frame)
        self.face_worker = LatestFrameWorker("FaceWorker", self._process_face_frame)

        # Start data send thread
        self.data_thread = threading.Thread(
            target=data_send_thread, args=(g.config["Sending"]["address"],), daemon=True
        )
        self.data_thread.start()

        # Start smoothing thread if enabled
        self.smoothing_thread=None
        if g.config["Smoothing"]["enable"]:
            self.smoothing_thread = threading.Thread(target=apply_smoothing, daemon=True)
            self.smoothing_thread.start()

    def _vision_config_changed(self, provider, hand_model_complexity):
        models = (g.tongue_model, g.face_detector, g.hand_detector)
        if any(model is not None and model.model_provider != provider for model in models):
            return True
        return (
            g.hand_detector is not None
            and g.hand_detector.model_complexity != hand_model_complexity
        )

    def _close_vision_models(self):
        for model in (g.face_detector, g.hand_detector, g.tongue_model):
            if model is not None and hasattr(model, "close"):
                model.close()
        g.face_detector = None
        g.hand_detector = None
        g.tongue_model = None

    def _process_hand_frame(self, image_rgb):
        hand_result = g.hand_detector.process_frame(image_rgb)
        hand_pred_handling(hand_result)

    def _process_face_frame(self, image_rgb, timestamp_ms):
        g.face_detector.process_frame(
            image_rgb,
            timestamp_ms=timestamp_ms,
            output_blendshapes=(
                g.config["Tracking"]["Face"]["enable"]
                or g.config["Tracking"]["Tongue"]["enable"]
            ),
            output_transform=g.config["Tracking"]["Head"]["enable"],
        )

    def restart_smoothing(self):
        print("restart smoothing")
        if self.smoothing_thread:
            self.smoothing_thread.join()
        if g.config["Smoothing"]["enable"]:
            self.smoothing_thread = threading.Thread(target=apply_smoothing, daemon=True)
            self.smoothing_thread.start()

    def process_frame(self, image_rgb):
        timestamp_ms = int((cv2.getTickCount() - g.start_time) * 1000 / cv2.getTickFrequency())
        needs_frame = (
            g.config["Tracking"]["Hand"]["enable"]
            or g.config["Tracking"]["Head"]["enable"]
            or g.config["Tracking"]["Face"]["enable"]
            or g.config["Tracking"]["Tongue"]["enable"]
        )
        frame = image_rgb.copy() if needs_frame else image_rgb
        if g.config["Tracking"]["Hand"]["enable"]:
            self.hand_worker.submit(frame)
        if (
            g.config["Tracking"]["Head"]["enable"]
            or g.config["Tracking"]["Face"]["enable"]
            or g.config["Tracking"]["Tongue"]["enable"]
        ):
            self.face_worker.submit(frame, timestamp_ms)

    def stop(self):
        self.is_running = False
        self.hand_worker.close()
        self.face_worker.close()
        g.stop_event.set()
        if self.smoothing_thread:
            self.smoothing_thread.join()
        self.data_thread.join()
        g.stop_event.clear()
