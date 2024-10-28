import threading
import cv2
import mediapipe as mp
from tracker.face.tongue import initialize_tongue_model
from tracker.face.face import initialize_face,face_pred_handling
from tracker.hand.hand import initialize_hand,hand_pred_handling
from utils.sender import data_send_thread
from utils.hotkeys import stop_hotkeys,apply_hotkeys
from utils.smoothing import apply_smoothing
import utils.globals as g


class Tracker:
    def __init__(self):
        self.is_running = True
        self.image = None

        g.update_configs(True)

        # TODO: I hate the new version mediapipe
        if g.face_detector is None or g.hand_detector is None or g.tongue_model is None:
            print("Initializing tongue model")
            g.tongue_model = initialize_tongue_model()
            print("Initializing MediaPipe")
            g.face_detector = initialize_face(g.tongue_model)
            g.hand_detector = initialize_hand()

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
        
        apply_hotkeys()

    def restart_smoothing(self):
        print("restart smoothing")
        if self.smoothing_thread:
            self.smoothing_thread.join()
        if g.config["Smoothing"]["enable"]:
            self.smoothing_thread = threading.Thread(target=apply_smoothing, daemon=True)
            self.smoothing_thread.start() 

    def process_frames(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int((cv2.getTickCount() - g.start_time) * 1000 / cv2.getTickFrequency())
        if g.config["Tracking"]["Head"]["enable"] or g.config["Tracking"]["Face"]["enable"]:
            g.face_detector.detect_async(mp_image,timestamp_ms=timestamp_ms)
        if g.config["Tracking"]["Hand"]["enable"]:
            hand_result=g.hand_detector.process(image_rgb)
            hand_pred_handling(hand_result)

    def stop(self):
        self.is_running = False
        g.stop_event.set()
        if self.smoothing_thread:
            self.smoothing_thread.join()
        self.data_thread.join()
        stop_hotkeys()
        g.stop_event.clear()


