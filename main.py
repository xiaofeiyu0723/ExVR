import sys
import onnxruntime as _onnxruntime_preload
import pyuac
if not pyuac.isUserAdmin():
    pyuac.runAsAdmin()
    sys.exit(0)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QComboBox,
    QHBoxLayout,
    QFrame,
    QCheckBox,
    QSlider,
    QMessageBox,
    QDialog,
    QScrollArea,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
import filecmp, os, winreg, shutil
import cv2
import utils.tracking
from utils.actions import *
import utils.globals as g
from utils.data import setup_data,save_data
from utils.hotkeys import stop_hotkeys, apply_hotkeys
from utils.language import LANGUAGE_OPTIONS, LANGUAGE_SYSTEM, install_language, tr
from tracker.face.face import draw_face_landmarks
from tracker.face.tongue import draw_tongue_position
from tracker.hand.hand import draw_hand_landmarks

from ctypes import windll
from cv2_enumerate_cameras import enumerate_cameras
from tracker.controller.controller import *
import warnings
warnings.filterwarnings("ignore")

CAMERA_PERFORMANCE_PRESETS = [
    ("Max Performance", {"4:3": (192, 144), "16:9": (256, 144)}),
    ("Performance", {"4:3": (320, 240), "16:9": (320, 180)}),
    ("Balanced", {"4:3": (640, 480), "16:9": (640, 360)}),
    ("Quality", {"4:3": (800, 600), "16:9": (800, 450)}),
]
CAMERA_TARGET_FPS = 60
PROCESS_PRIORITY_OPTIONS = [
    ("IDLE_PRIORITY_CLASS", "Idle"),
    ("BELOW_NORMAL_PRIORITY_CLASS", "Below Normal"),
    ("NORMAL_PRIORITY_CLASS", "Normal"),
    ("ABOVE_NORMAL_PRIORITY_CLASS", "Above Normal"),
    ("HIGH_PRIORITY_CLASS", "High"),
    ("REALTIME_PRIORITY_CLASS", "Realtime"),
]


class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, source,width=640, height=480, fps=60):
        super().__init__()
        self.source = source
        self.video_capture = None
        self.is_running = True
        self.show_image = False
        self.tracker = utils.tracking.Tracker()
        self.width = int(width)
        self.height = int(height)
        self.capture_width, self.capture_height = self.capture_request_size(self.width, self.height)
        self.fps = fps

    @staticmethod
    def capture_request_size(width, height):
        if width <= 0 or height <= 0:
            return 640, 480
        aspect_ratio = width / height
        if abs(aspect_ratio - 16 / 9) < abs(aspect_ratio - 4 / 3):
            if width <= 1280 and height <= 720:
                return 1280, 720
            return width, height
        if width <= 640 and height <= 480:
            return 640, 480
        if width <= 800 and height <= 600:
            return 800, 600
        return width, height

    def resize_for_processing(self, rgb_image):
        image_height, image_width = rgb_image.shape[:2]
        if image_width <= 0 or image_height <= 0:
            return rgb_image
        target_ratio = self.width / self.height
        image_ratio = image_width / image_height
        if abs(image_ratio - target_ratio) > 0.01:
            if image_ratio > target_ratio:
                crop_width = int(round(image_height * target_ratio))
                x0 = max(0, (image_width - crop_width) // 2)
                rgb_image = rgb_image[:, x0:x0 + crop_width]
            else:
                crop_height = int(round(image_width / target_ratio))
                y0 = max(0, (image_height - crop_height) // 2)
                rgb_image = rgb_image[y0:y0 + crop_height, :]
        if rgb_image.shape[1] != self.width or rgb_image.shape[0] != self.height:
            rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return rgb_image

    def run(self):
        self.video_capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
        print(
            "capture",
            self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.video_capture.get(cv2.CAP_PROP_FPS),
            "process",
            self.width,
            self.height,
        )
        g.current_fps =self.video_capture.get(cv2.CAP_PROP_FPS);
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.is_running:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = self.resize_for_processing(rgb_image)
                if g.config["Setting"]["flip_x"]:
                    rgb_image = cv2.flip(rgb_image, 1)
                if g.config["Setting"]["flip_y"]:
                    rgb_image = cv2.flip(rgb_image, 0)

                self.tracker.process_frame(rgb_image)
                if self.show_image:
                    if g.config["Tracking"]["Head"]["enable"] or g.config["Tracking"]["Face"]["enable"]:
                        rgb_image = draw_face_landmarks(rgb_image)
                    if g.config["Tracking"]["Tongue"]["enable"]:
                        rgb_image = draw_tongue_position(rgb_image)
                    if g.config["Tracking"]["Hand"]["enable"]:
                        rgb_image = draw_hand_landmarks(rgb_image)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(
                        rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                    )
                    self.frame_ready.emit(convert_to_Qt_format)
        self.cleanup()

    def stop(self):
        self.is_running = False
        self.tracker.stop()

    def cleanup(self):
        if self.video_capture:
            self.video_capture.release()
            cv2.destroyAllWindows()

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        screen = QApplication.screens()[0]
        screen_size = screen.size()
        self.width = int(screen_size.width() * 0.3)
        self.height = int(screen_size.height() * 0.65)
        self.half_height = int(self.height / 2)

        # self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setAttribute(Qt.WA_NoSystemBackground, False)
        # self.setAttribute(Qt.WA_PaintOnScreen)
        version=g.config["Version"]
        self.setWindowTitle(
            f"ExVR {version} - Experience Virtual Reality"
        )
        # self.setFixedSize(width, height)
        self.resize(self.width, self.height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)
        self.setMinimumSize(600, 800)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        top_right_layout = QHBoxLayout()
        top_right_layout.addStretch()
        self.language_label = QLabel(self)
        top_right_layout.addWidget(self.language_label)
        self.language_selection = QComboBox(self)
        self.populate_language_list()
        self.language_selection.currentIndexChanged.connect(self.update_language)
        top_right_layout.addWidget(self.language_selection)
        self.steamvr_status_label = QLabel(self)
        top_right_layout.addWidget(self.steamvr_status_label)
        layout.insertLayout(0, top_right_layout)


        flip_layout = QHBoxLayout()  # Create a QHBoxLayout for new reset buttons
        self.flip_x_checkbox = QCheckBox(self)
        self.flip_x_checkbox.clicked.connect(self.flip_x)
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        flip_layout.addWidget(self.flip_x_checkbox)

        self.flip_y_checkbox = QCheckBox(self)
        self.flip_y_checkbox.clicked.connect(self.flip_y)
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        flip_layout.addWidget(self.flip_y_checkbox)
        layout.addLayout(flip_layout)

        self.ip_camera_url_input = QLineEdit(self)
        self.ip_camera_url_input.setPlaceholderText(tr("Enter IP camera URL"))
        self.ip_camera_url_input.textChanged.connect(self.update_camera_ip)
        # use .get() to avoid KeyError with old config
        self.ip_camera_url_input.setText(g.config["Setting"].get("camera_ip", ""))
        layout.addWidget(self.ip_camera_url_input)

        camera_layout = QHBoxLayout()
        self.camera_selection = QComboBox(self)
        self.populate_camera_list()
        camera_layout.addWidget(self.camera_selection)
        self.camera_performance_label = QLabel(self)
        camera_layout.addWidget(self.camera_performance_label)
        self.camera_performance_selection = QComboBox(self)
        camera_layout.addWidget(self.camera_performance_selection)
        self.camera_aspect_label = QLabel(self)
        camera_layout.addWidget(self.camera_aspect_label)
        self.camera_aspect_selection = QComboBox(self)
        self.populate_camera_aspect_list()
        self.populate_camera_performance_list()
        self.camera_performance_selection.currentIndexChanged.connect(lambda: self.update_camera_performance())
        self.camera_aspect_selection.currentIndexChanged.connect(lambda: self.update_camera_performance())
        camera_layout.addWidget(self.camera_aspect_selection)
        layout.addLayout(camera_layout)

        model_provider_layout = QHBoxLayout()
        self.model_provider_label = QLabel(self)
        model_provider_layout.addWidget(self.model_provider_label)
        self.model_provider_selection = QComboBox(self)
        self.model_provider_selection.addItems(["GPU", "CPU"])
        self.model_provider_selection.currentIndexChanged.connect(self.update_model_provider)
        model_provider_layout.addWidget(self.model_provider_selection)
        layout.addLayout(model_provider_layout)
        self.model_provider_selection.setCurrentIndex(
            self.model_provider_selection.findText(g.config["Model"]["provider"])
        )

        priority_layout = QHBoxLayout()
        self.priority_label = QLabel(self)
        priority_layout.addWidget(self.priority_label)
        self.priority_selection = QComboBox(self)
        self.populate_priority_list()
        self.priority_selection.currentIndexChanged.connect(self.set_process_priority)
        priority_layout.addWidget(self.priority_selection)
        layout.addLayout(priority_layout)

        self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        sync_result = self.sync_installed_components(steamvr_driver_path, vrcfacetracking_path)
        if sync_result["error"] is None:
            self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        self.steamvr_installed = check_steamvr_path is not None
        if self.steamvr_installed:
            self.steamvr_status_label.setText(tr("SteamVR Installed"))
            self.steamvr_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.steamvr_status_label.setText(tr("SteamVR Not Installed"))
            self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")
        if sync_result["error"]:
            self.display_message("Driver Update", sync_result["error"])
        elif sync_result["updated"]:
            self.display_message(
                "Driver Update",
                tr("Updated installed components:") + "\n\n" + "\n".join(sync_result["updated"]),
                icon=QMessageBox.Information,
            )
        if self.install_state:
            self.install_button = QPushButton(self)
            self.install_button.setStyleSheet("")
        else:
            self.install_button = QPushButton(self)
            self.install_button.setStyleSheet(
                "QPushButton { background-color: blue; color: white; }"
            )
        self.install_button.clicked.connect(self.install_function)
        layout.addWidget(self.install_button)

        self.toggle_button = QPushButton(self)
        self.toggle_button.setStyleSheet(
            "QPushButton { background-color: green; color: white; }"
        )
        self.toggle_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.toggle_button)

        self.show_frame_button = QPushButton(self)
        self.show_frame_button.clicked.connect(self.toggle_video_display)
        layout.addWidget(self.show_frame_button)

        only_ingame_layout = QHBoxLayout()
        self.only_ingame_checkbox = QCheckBox(self)
        self.only_ingame_checkbox.clicked.connect(lambda: self.toggle_only_in_game(self.only_ingame_checkbox.isChecked()))
        self.only_ingame_checkbox.setChecked(g.config["Setting"]["only_ingame"])
        self.only_ingame_checkbox.setToolTip(tr("Currently this only applies to hotkeys and mouse input and not head movement"))
        self.only_ingame_game_input = QLineEdit(self)
        self.only_ingame_game_input.setPlaceholderText(tr("window title / process name / VRChat, VRChat.exe, javaw.exe"))
        self.only_ingame_game_input.textChanged.connect(self.update_mouse_only_in_game_name)
        self.only_ingame_game_input.setText(g.config["Setting"]["only_ingame_game"])
        only_ingame_layout.addWidget(self.only_ingame_checkbox)
        only_ingame_layout.addWidget(self.only_ingame_game_input)
        layout.addLayout(only_ingame_layout)

        separator_0 = QFrame(self)
        separator_0.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_0.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_0)

        reset_layout = QHBoxLayout()  # Create a QHBoxLayout for new reset buttons
        self.reset_head = QPushButton(self)
        self.reset_head.clicked.connect(reset_head)
        reset_layout.addWidget(self.reset_head)
        self.reset_eyes = QPushButton(self)
        self.reset_eyes.clicked.connect(reset_eye)
        reset_layout.addWidget(self.reset_eyes)
        self.reset_l_hand = QPushButton(self)
        self.reset_l_hand.clicked.connect(lambda: reset_hand(True))
        reset_layout.addWidget(self.reset_l_hand)
        self.reset_r_hand = QPushButton(self)
        self.reset_r_hand.clicked.connect(lambda: reset_hand(False))
        reset_layout.addWidget(self.reset_r_hand)
        layout.addLayout(reset_layout)

        checkbox_layout = QHBoxLayout()
        self.checkbox1 = QCheckBox(self)
        self.checkbox1.clicked.connect(
            lambda: self.set_tracking_config("Head", self.checkbox1.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox1)
        self.checkbox2 = QCheckBox(self)
        self.checkbox2.clicked.connect(
            lambda: self.set_tracking_config("Face", self.checkbox2.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox2)
        self.checkbox3 = QCheckBox(self)
        self.checkbox3.clicked.connect(
            lambda: self.set_tracking_config("Tongue", self.checkbox3.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox3)
        self.checkbox4 = QCheckBox(self)
        self.checkbox4.clicked.connect(
            lambda: self.set_tracking_config("Hand", self.checkbox4.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox4)
        layout.addLayout(checkbox_layout)

        checkbox_layout_1 = QHBoxLayout()
        self.checkbox6 = QCheckBox(self)
        self.checkbox6.clicked.connect(
            lambda: self.toggle_hand_down(self.checkbox6.isChecked())
        )
        checkbox_layout_1.addWidget(self.checkbox6)
        self.checkbox7 = QCheckBox(self)
        self.checkbox7.clicked.connect(
            lambda: self.toggle_finger_action(self.checkbox7.isChecked())
        )
        checkbox_layout_1.addWidget(self.checkbox7)
        self.hand_return_time_label = QLabel(self)
        checkbox_layout_1.addWidget(self.hand_return_time_label)
        self.hand_return_time_input = QLineEdit(self)
        hand_return_time_validator = QDoubleValidator(0.0, 60.0, 2)
        self.hand_return_time_input.setValidator(hand_return_time_validator)
        self.hand_return_time_input.setFixedWidth(60)
        self.hand_return_time_input.editingFinished.connect(self.update_hand_return_time)
        checkbox_layout_1.addWidget(self.hand_return_time_input)
        layout.addLayout(checkbox_layout_1)

        slider_layout = QHBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider1.setRange(1, 200)
        self.slider2.setRange(1, 200)
        self.slider3.setRange(1, 100)
        self.slider1.setSingleStep(1)
        self.slider2.setSingleStep(1)
        self.slider3.setSingleStep(1)
        self.label1 = QLabel(f"x {g.config['Tracking']['Hand']['x_scalar']:.2f}")
        self.label2 = QLabel(f"y {g.config['Tracking']['Hand']['y_scalar']:.2f}")
        self.label3 = QLabel(f"z {g.config['Tracking']['Hand']['z_scalar']:.2f}")
        self.slider1.valueChanged.connect(lambda value: self.set_scalar(value, "x"))
        self.slider2.valueChanged.connect(lambda value: self.set_scalar(value, "y"))
        self.slider3.valueChanged.connect(lambda value: self.set_scalar(value, "z"))
        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.slider1)
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.slider2)
        slider_layout.addWidget(self.label3)
        slider_layout.addWidget(self.slider3)
        layout.addLayout(slider_layout)

        separator_1 = QFrame(self)
        separator_1.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_1.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_1)

        # label_layout = QHBoxLayout()
        # self.left_label = self.create_label("Left Controller", "red")
        # self.right_label = self.create_label("Right Controller", "red")
        # label_layout.addWidget(self.left_label)
        # label_layout.addWidget(self.right_label)
        # layout.addLayout(label_layout)

        controller_checkbox_layout = QHBoxLayout()
        self.controller_checkbox1 = QCheckBox(self)
        self.controller_checkbox1.clicked.connect(
            lambda: self.set_tracking_config("LeftController", self.controller_checkbox1.isChecked())
        )
        self.controller_checkbox2 = QCheckBox(self)
        self.controller_checkbox2.clicked.connect(
            lambda: self.set_tracking_config("RightController", self.controller_checkbox2.isChecked())
        )
        controller_checkbox_layout.addWidget(self.controller_checkbox1)
        controller_checkbox_layout.addWidget(self.controller_checkbox2)
        layout.addLayout(controller_checkbox_layout)

        controller_slider_layout = QHBoxLayout()
        self.controller_slider_x = QSlider(Qt.Horizontal)
        self.controller_slider_y = QSlider(Qt.Horizontal)
        self.controller_slider_z = QSlider(Qt.Horizontal)
        self.controller_slider_l = QSlider(Qt.Horizontal)
        self.controller_slider_x.setRange(-50, 50)
        self.controller_slider_y.setRange(-50, 50)
        self.controller_slider_z.setRange(-50, 50)
        self.controller_slider_l.setRange(0, 100)
        self.controller_slider_x.setSingleStep(1)
        self.controller_slider_y.setSingleStep(1)
        self.controller_slider_z.setSingleStep(1)
        self.controller_slider_l.setSingleStep(1)
        self.controller_label_x = QLabel(f"x {g.config['Tracking']['LeftController']['base_x']:.2f}")
        self.controller_label_y = QLabel(f"y {g.config['Tracking']['LeftController']['base_y']:.2f}")
        self.controller_label_z = QLabel(f"z {g.config['Tracking']['LeftController']['base_z']:.2f}")
        self.controller_label_l = QLabel(f"l {g.config['Tracking']['LeftController']['length']:.2f}")
        self.controller_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, "controller_x"))
        self.controller_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, "controller_y"))
        self.controller_slider_z.valueChanged.connect(lambda value: self.set_scalar(value, "controller_z"))
        self.controller_slider_l.valueChanged.connect(lambda value: self.set_scalar(value, "controller_l"))
        controller_slider_layout.addWidget(self.controller_label_x)
        controller_slider_layout.addWidget(self.controller_slider_x)
        controller_slider_layout.addWidget(self.controller_label_y)
        controller_slider_layout.addWidget(self.controller_slider_y)
        controller_slider_layout.addWidget(self.controller_label_z)
        controller_slider_layout.addWidget(self.controller_slider_z)
        controller_slider_layout.addWidget(self.controller_label_l)
        controller_slider_layout.addWidget(self.controller_slider_l)
        layout.addLayout(controller_slider_layout)


        separator_2 = QFrame(self)
        separator_2.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_2.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_2)
        mouse_layout = QHBoxLayout()
        self.mouse_checkbox = QCheckBox(self)
        self.mouse_checkbox.clicked.connect(lambda: self.toggle_mouse(self.mouse_checkbox.isChecked()))
        self.mouse_checkbox.setChecked(g.config["Mouse"]["enable"])

        self.mouse_slider_x = QSlider(Qt.Horizontal)
        self.mouse_slider_y = QSlider(Qt.Horizontal)
        self.mouse_slider_dx = QSlider(Qt.Horizontal)
        self.mouse_slider_x.setRange(0, 360)
        self.mouse_slider_y.setRange(0, 360)
        self.mouse_slider_dx.setRange(0, 20)
        self.mouse_slider_x.setSingleStep(1)
        self.mouse_slider_y.setSingleStep(1)
        self.mouse_slider_dx.setSingleStep(1)
        self.mouse_label_x = QLabel(f"x {int(g.config['Mouse']['scalar_x']*100)}")
        self.mouse_label_y = QLabel(f"y {int(g.config['Mouse']['scalar_y']*100)}")
        self.mouse_label_dx = QLabel(f"dx {g.config['Mouse']['dx']:.2f}")
        self.mouse_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_x"))
        self.mouse_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_y"))
        self.mouse_slider_dx.valueChanged.connect(lambda value: self.set_scalar(value, "mouse_dx"))
        mouse_layout.addWidget(self.mouse_checkbox)
        mouse_layout.addWidget(self.mouse_label_x)
        mouse_layout.addWidget(self.mouse_slider_x)
        mouse_layout.addWidget(self.mouse_label_y)
        mouse_layout.addWidget(self.mouse_slider_y)
        mouse_layout.addWidget(self.mouse_label_dx)
        mouse_layout.addWidget(self.mouse_slider_dx)
        layout.addLayout(mouse_layout)

        separator_3 = QFrame(self)
        separator_3.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_3.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_3)

        config_layout = QHBoxLayout()
        self.reset_hotkey_button = QPushButton(self)
        self.reset_hotkey_button.clicked.connect(self.reset_hotkeys)
        config_layout.addWidget(self.reset_hotkey_button)
        self.stop_hotkey_button = QPushButton(self)
        self.stop_hotkey_button.clicked.connect(stop_hotkeys)
        config_layout.addWidget(self.stop_hotkey_button)
        self.set_face_button = QPushButton(self)
        self.set_face_button.clicked.connect(self.face_dialog)
        config_layout.addWidget(self.set_face_button)
        self.update_config_button = QPushButton(self)
        self.update_config_button.clicked.connect(self.reload_configs)
        config_layout.addWidget(self.update_config_button)
        self.save_config_button = QPushButton(self)
        self.save_config_button.clicked.connect(g.save_configs)
        config_layout.addWidget(self.save_config_button)
        layout.addLayout(config_layout)

        self.update_checkboxes()
        self.update_sliders()
        self.update_hand_return_time_input()
        self.update_model_provider_selection()
        self.video_thread = None
        self.controller_thread = None
        self.retranslate_ui()

    def populate_language_list(self):
        for code, label in LANGUAGE_OPTIONS:
            self.language_selection.addItem(tr(label), code)
        language = g.config["Setting"].get("language", LANGUAGE_SYSTEM)
        index = self.language_selection.findData(language)
        self.language_selection.setCurrentIndex(index if index >= 0 else 0)

    def update_language_selection(self):
        language = g.config["Setting"].get("language", LANGUAGE_SYSTEM)
        index = self.language_selection.findData(language)
        if index >= 0:
            self.language_selection.blockSignals(True)
            self.language_selection.setCurrentIndex(index)
            self.language_selection.blockSignals(False)

    def update_language(self):
        language = self.language_selection.currentData() or LANGUAGE_SYSTEM
        g.config["Setting"]["language"] = language
        install_language(QApplication.instance(), language)
        self.retranslate_ui()

    def reload_configs(self):
        g.update_configs()
        install_language(QApplication.instance(), g.config["Setting"].get("language", LANGUAGE_SYSTEM))
        self.update_checkboxes()
        self.update_sliders()
        self.update_hand_return_time_input()
        self.update_model_provider_selection()
        self.update_priority_selection()
        self.update_language_selection()
        self.retranslate_ui()

    def retranslate_ui(self):
        version = g.config["Version"]
        self.setWindowTitle(tr("ExVR {version} - Experience Virtual Reality").format(version=version))
        self.language_label.setText(tr("Language"))
        for index, (_, label) in enumerate(LANGUAGE_OPTIONS):
            self.language_selection.setItemText(index, tr(label))
        self.flip_x_checkbox.setText(tr("Flip X"))
        self.flip_y_checkbox.setText(tr("Flip Y"))
        self.ip_camera_url_input.setPlaceholderText(tr("Enter IP camera URL"))
        self.camera_performance_label.setText(tr("Performance"))
        self.camera_aspect_label.setText(tr("Aspect"))
        for index, (label, _) in enumerate(CAMERA_PERFORMANCE_PRESETS):
            self.camera_performance_selection.setItemText(index, tr(label))
        for index, (_, label) in enumerate(PROCESS_PRIORITY_OPTIONS):
            self.priority_selection.setItemText(index, tr(label))
        self.priority_label.setText(tr("Priority"))
        self.model_provider_label.setText(tr("Model Provider"))
        self.steamvr_status_label.setText(tr("SteamVR Installed") if self.steamvr_installed else tr("SteamVR Not Installed"))
        self.install_button.setText(tr("Uninstall Drivers") if self.install_state else tr("Install Drivers"))
        tracking_active = self.video_thread is not None and self.video_thread.isRunning()
        self.toggle_button.setText(tr("Stop Tracking") if tracking_active else tr("Start Tracking"))
        show_frame_active = self.video_thread is not None and self.video_thread.show_image
        self.show_frame_button.setText(tr("Hide Frame") if show_frame_active else tr("Show Frame"))
        self.only_ingame_checkbox.setText(tr("Only Ingame"))
        self.only_ingame_checkbox.setToolTip(tr("Currently this only applies to hotkeys and mouse input and not head movement"))
        self.only_ingame_game_input.setPlaceholderText(tr("window title / process name / VRChat, VRChat.exe, javaw.exe"))
        self.reset_head.setText(tr("Reset Head"))
        self.reset_eyes.setText(tr("Reset Eyes"))
        self.reset_l_hand.setText(tr("Reset LHand"))
        self.reset_r_hand.setText(tr("Reset RHand"))
        self.checkbox1.setText(tr("Head"))
        self.checkbox2.setText(tr("Face"))
        self.checkbox3.setText(tr("Tongue"))
        self.checkbox4.setText(tr("Hand"))
        self.checkbox6.setText(tr("Hand Down"))
        self.checkbox7.setText(tr("Finger Action"))
        self.hand_return_time_label.setText(tr("Hand Return Time (s)"))
        self.controller_checkbox1.setText(tr("Left Controller"))
        self.controller_checkbox2.setText(tr("Right Controller"))
        self.mouse_checkbox.setText(tr("Mouse"))
        self.reset_hotkey_button.setText(tr("Reset Hotkey"))
        self.stop_hotkey_button.setText(tr("Stop Hotkey"))
        self.set_face_button.setText(tr("Set Face"))
        self.update_config_button.setText(tr("Update Config"))
        self.save_config_button.setText(tr("Save Config"))

    def save_data(self):
        data=deepcopy(g.default_data)
        for i, (key, edits) in enumerate(self.lineEdits.items()):
            idx=i+1
            v = float(edits[0].text())
            s = float(edits[1].text())
            w = float(edits[2].text())
            max_value = float(edits[3].text())
            e = self.checkBoxes[key].isChecked()
            data["BlendShapes"][idx]["v"] = v
            data["BlendShapes"][idx]["s"] = s
            data["BlendShapes"][idx]["w"] = w
            data["BlendShapes"][idx]["max"] = max_value
            data["BlendShapes"][idx]["e"] = e
        save_data(data)
        self.dialog.close()

    def face_dialog(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(tr("Face Setting"))
        self.dialog.resize(self.width, self.height)  # Set a fixed size for the dialog

        layout = QVBoxLayout(self.dialog)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)  # Use a grid layout for better alignment

        # Add header labels for the form
        form_layout.addWidget(QLabel(tr("BlendShape")), 0, 0)
        form_layout.addWidget(QLabel(tr("Value")), 0, 1)
        form_layout.addWidget(QLabel(tr("Shifting")), 0, 2)
        form_layout.addWidget(QLabel(tr("Weight")), 0, 3)
        form_layout.addWidget(QLabel(tr("Max")), 0, 4)
        form_layout.addWidget(QLabel(tr("Enabled")), 0, 5)

        # Store QLineEdit and QCheckBox references
        self.lineEdits = {}
        self.checkBoxes = {}

        # Create input fields for each blend shape
        double_validator = QDoubleValidator()
        blendshape_data,_=setup_data()
        for i, blendshape in enumerate(blendshape_data["BlendShapes"][1:], start=1):  # Start from row 1
            key = blendshape["k"]
            v_edit = QLineEdit(str(round(blendshape["v"],2)))
            v_edit.setValidator(double_validator)
            s_edit = QLineEdit(str(round(blendshape["s"],2)))
            s_edit.setValidator(double_validator)
            w_edit = QLineEdit(str(round(blendshape["w"],2)))
            w_edit.setValidator(double_validator)
            max_edit = QLineEdit(str(round(blendshape["max"],2)))
            max_edit.setValidator(double_validator)
            e_check = QCheckBox()
            e_check.setChecked(blendshape["e"])

            # Save references to the QLineEdit and QCheckBox
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check

            # Add widgets to the grid layout
            form_layout.addWidget(QLabel(key), i, 0)  # Blend shape key in column 0
            form_layout.addWidget(v_edit, i, 1)  # v_edit in column 1
            form_layout.addWidget(s_edit, i, 2)  # s_edit in column 2
            form_layout.addWidget(w_edit, i, 3)  # w_edit in column 3
            form_layout.addWidget(max_edit, i, 4)  # max_edit in column 4
            form_layout.addWidget(e_check, i, 5)  # e_check in column 5

        # Add the form layout to the scroll area
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)  # Add the scroll area to the dialog layout

        # Add a Save Config button
        self.face_save_config_button = QPushButton(tr("Save Config"), self.dialog)
        self.face_save_config_button.clicked.connect(self.save_data)
        layout.addWidget(self.face_save_config_button)

        self.dialog.exec_()

    def flip_x(self, value):
        g.config["Setting"]["flip_x"] = value

    def flip_y(self, value):
        g.config["Setting"]["flip_y"] = value

    def set_hand_front(self, value):
        g.config["Tracking"]["Hand"]["only_front"] = value

    def set_face_block(self, value):
        g.config["Tracking"]["Face"]["block"] = value

    def update_checkboxes(self):
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        self.checkbox1.setChecked(g.config["Tracking"]["Head"]["enable"])
        self.checkbox2.setChecked(g.config["Tracking"]["Face"]["enable"])
        self.checkbox3.setChecked(g.config["Tracking"]["Tongue"]["enable"])
        self.checkbox4.setChecked(g.config["Tracking"]["Hand"]["enable"])
        self.checkbox6.setChecked(g.config["Tracking"]["Hand"]["enable_hand_down"])
        self.checkbox7.setChecked(g.config["Tracking"]["Hand"]["enable_finger_action"])
        self.controller_checkbox1.setChecked(g.config["Tracking"]["LeftController"]["enable"])
        self.controller_checkbox2.setChecked(g.config["Tracking"]["RightController"]["enable"])
        self.mouse_checkbox.setChecked(g.config["Mouse"]["enable"])

    def set_scalar(self, value, axis):
        slider_value = value / 100.0
        if axis == "x":
            g.config["Tracking"]["Hand"]["x_scalar"] = slider_value
            self.label1.setText(f"x {slider_value:.2f}")
        elif axis == "y":
            g.config["Tracking"]["Hand"]["y_scalar"] = slider_value
            self.label2.setText(f"y {slider_value:.2f}")
        elif axis == "z":
            g.config["Tracking"]["Hand"]["z_scalar"] = slider_value
            self.label3.setText(f"z {slider_value:.2f}")
        elif axis == "controller_x":
            g.config["Tracking"]["LeftController"]["base_x"] = slider_value
            g.config["Tracking"]["RightController"]["base_x"] = -slider_value
            self.controller_label_x.setText(f"x {slider_value:.2f}")
        elif axis == "controller_y":
            g.config["Tracking"]["LeftController"]["base_y"] = slider_value
            g.config["Tracking"]["RightController"]["base_y"] = slider_value
            self.controller_label_y.setText(f"y {slider_value:.2f}")
        elif axis == "controller_z":
            g.config["Tracking"]["LeftController"]["base_z"] = slider_value
            g.config["Tracking"]["RightController"]["base_z"] = slider_value
            self.controller_label_z.setText(f"z {slider_value:.2f}")
        elif axis == "controller_l":
            g.config["Tracking"]["LeftController"]["length"] = slider_value
            g.config["Tracking"]["RightController"]["length"] = slider_value
            self.controller_label_l.setText(f"l {slider_value:.2f}")
        elif axis == "mouse_x":
            g.config["Mouse"]["scalar_x"]=slider_value*100
            self.mouse_label_x.setText(f"x {int(slider_value*100)}")
        elif axis == "mouse_y":
            g.config["Mouse"]["scalar_y"]=slider_value*100
            self.mouse_label_y.setText(f"y {int(slider_value*100)}")
        elif axis == "mouse_dx":
            g.config["Mouse"]["dx"]=slider_value
            self.mouse_label_dx.setText(f"dx {slider_value:.2f}")


    def update_sliders(self):
        x_scalar = g.config["Tracking"]["Hand"]["x_scalar"]
        y_scalar = g.config["Tracking"]["Hand"]["y_scalar"]
        z_scalar = g.config["Tracking"]["Hand"]["z_scalar"]
        self.slider1.setValue(int(x_scalar * 100))
        self.slider2.setValue(int(y_scalar * 100))
        self.slider3.setValue(int(z_scalar * 100))
        self.label1.setText(f"x {x_scalar:.2f}")
        self.label2.setText(f"y {y_scalar:.2f}")
        self.label3.setText(f"z {z_scalar:.2f}")

        controller_x = g.config["Tracking"]["LeftController"]["base_x"]
        controller_y = g.config["Tracking"]["LeftController"]["base_y"]
        controller_z = g.config["Tracking"]["LeftController"]["base_z"]
        controller_l = g.config["Tracking"]["LeftController"]["length"]
        self.controller_slider_x.setValue(int(controller_x * 100))
        self.controller_slider_y.setValue(int(controller_y * 100))
        self.controller_slider_z.setValue(int(controller_z * 100))
        self.controller_slider_l.setValue(int(controller_l * 100))
        self.controller_label_x.setText(f"x {controller_x:.2f}")
        self.controller_label_y.setText(f"y {controller_y:.2f}")
        self.controller_label_z.setText(f"z {controller_z:.2f}")
        self.controller_label_l.setText(f"l {controller_l:.2f}")

        mouse_x=g.config["Mouse"]["scalar_x"]
        mouse_y=g.config["Mouse"]["scalar_y"]
        mouse_dx=g.config["Mouse"]["dx"]
        self.mouse_slider_x.setValue(int(mouse_x))
        self.mouse_slider_y.setValue(int(mouse_y))
        self.mouse_slider_dx.setValue(int(mouse_dx * 100))
        self.mouse_label_x.setText(f"x {int(mouse_x)}")
        self.mouse_label_y.setText(f"y {int(mouse_y)}")
        self.mouse_label_dx.setText(f"dx {mouse_dx:.2f}")

    def reset_hotkeys(self):
        stop_hotkeys()
        apply_hotkeys()
        if self.video_thread is None:
            stop_hotkeys()

    def set_tracking_config(self, key, value):
        if key in g.config["Tracking"]:
            g.config["Tracking"][key]["enable"] = value
        if key == "LeftController":
            g.controller.left_controller.enable = value
            g.controller.left_controller.force_enable = value
            if not value:
                g.controller.release_controller_inputs(True)
                g.controller.disable_hand(g.controller.left_controller)
        if key == "RightController":
            g.controller.right_controller.enable = value
            g.controller.right_controller.force_enable = value
            if not value:
                g.controller.release_controller_inputs(False)
                g.controller.disable_hand(g.controller.right_controller)

    def on_controller_connection_changed(self, hand, connected):
        checkbox = self.controller_checkbox1 if hand == "Left" else self.controller_checkbox2
        checkbox.blockSignals(True)
        checkbox.setChecked(connected)
        checkbox.blockSignals(False)

    def toggle_mouse(self, value):
        g.config["Mouse"]["enable"] = value

    def toggle_only_in_game(self, value):
        g.config["Setting"]["only_ingame"] = value

    def update_mouse_only_in_game_name(self, value):
        g.config["Setting"]["only_ingame_game"] = value

    def toggle_hand_down(self, value):
        g.config["Tracking"]["Hand"]["enable_hand_down"] = value

    def toggle_finger_action(self, value):
        g.config["Tracking"]["Hand"]["enable_finger_action"] = value

    def update_hand_return_time(self):
        text = self.hand_return_time_input.text().strip()
        if not text:
            self.update_hand_return_time_input()
            return
        value = max(0.0, min(60.0, float(text)))
        g.config["Tracking"]["Hand"]["hand_return_time"] = value
        self.hand_return_time_input.setText(f"{value:g}")

    def update_hand_return_time_input(self):
        value = g.config["Tracking"]["Hand"].get("hand_return_time", 0.5)
        self.hand_return_time_input.setText(f"{float(value):g}")

    def copy_changed_file(self, source, destination):
        if not os.path.exists(source):
            return False
        if os.path.exists(destination) and filecmp.cmp(source, destination, shallow=False):
            return False
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        return True

    def copy_changed_tree(self, source_root, destination_root):
        if not os.path.exists(source_root):
            return False
        changed = False
        for dir_path, _, filenames in os.walk(source_root):
            rel_dir = os.path.relpath(dir_path, source_root)
            target_dir = destination_root if rel_dir == "." else os.path.join(destination_root, rel_dir)
            os.makedirs(target_dir, exist_ok=True)
            for filename in filenames:
                source = os.path.join(dir_path, filename)
                destination = os.path.join(target_dir, filename)
                changed = self.copy_changed_file(source, destination) or changed
        return changed

    def file_matches(self, source, destination):
        return (
            os.path.exists(source)
            and os.path.exists(destination)
            and filecmp.cmp(source, destination, shallow=False)
        )

    def copy_missing_files_and_changed_dlls(self, source_root, destination_root):
        if not os.path.exists(source_root):
            return False
        changed = False
        for dir_path, _, filenames in os.walk(source_root):
            rel_dir = os.path.relpath(dir_path, source_root)
            target_dir = destination_root if rel_dir == "." else os.path.join(destination_root, rel_dir)
            for filename in filenames:
                source = os.path.join(dir_path, filename)
                destination = os.path.join(target_dir, filename)
                is_dll = filename.lower().endswith(".dll")
                if not os.path.exists(destination):
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copy2(source, destination)
                    changed = True
                elif is_dll and not filecmp.cmp(source, destination, shallow=False):
                    shutil.copy2(source, destination)
                    changed = True
        return changed

    def driver_files_complete(self, source_root, destination_root):
        if not os.path.isdir(source_root) or not os.path.isdir(destination_root):
            return False
        for dir_path, _, filenames in os.walk(source_root):
            rel_dir = os.path.relpath(dir_path, source_root)
            target_dir = destination_root if rel_dir == "." else os.path.join(destination_root, rel_dir)
            for filename in filenames:
                source = os.path.join(dir_path, filename)
                destination = os.path.join(target_dir, filename)
                if not os.path.exists(destination):
                    return False
                if filename.lower().endswith(".dll") and not filecmp.cmp(source, destination, shallow=False):
                    return False
        return True

    def sync_installed_components(self, steamvr_driver_path, vrcfacetracking_path):
        if steamvr_driver_path is None:
            return {"updated": [], "error": None}

        app_root = os.path.dirname(os.path.abspath(__file__))
        updated = []
        try:
            for driver in ["vmt", "vrto3d"]:
                source = os.path.join(app_root, "drivers", driver)
                destination = os.path.join(steamvr_driver_path, driver)
                if self.copy_missing_files_and_changed_dlls(source, destination):
                    updated.append(driver)

            if vrcfacetracking_path is not None:
                dll_source = os.path.join(app_root, "drivers", "VRCFT-MediapipePro.dll")
                dll_destination = os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll")
                if os.path.exists(dll_destination) and self.copy_changed_file(dll_source, dll_destination):
                    updated.append("VRCFT-MediapipePro.dll")
        except (PermissionError, OSError) as exc:
            return {
                "updated": updated,
                "error": tr("Could not update installed drivers. Close SteamVR/VRCFaceTracking and reopen ExVR.") + f"\n\n{exc}",
            }

        if updated:
            print("Updated installed components:", ", ".join(updated))
        return {"updated": updated, "error": None}

    def install_checking(self):
        # Open registry key to get Steam installation path
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\Valve\Steam",
                0,
                winreg.KEY_READ,
            ) as reg_key:
                steam_path, _ = winreg.QueryValueEx(reg_key, "InstallPath")
            steamvr_driver_path = os.path.join(
                steam_path, "steamapps", "common", "SteamVR", "drivers"
            )

            check_steamvr_path = os.path.join(
                steam_path, "steamapps", "common", "SteamVR", "bin"
            )
            if not os.path.exists(check_steamvr_path):
                check_steamvr_path = None

            vrcfacetracking_path = os.path.join(
                os.getenv("APPDATA"), "VRCFaceTracking", "CustomLibs"
            )
            vrcfacetracking_module_path = os.path.join(
                vrcfacetracking_path, "VRCFT-MediapipePro.dll"
            )

            app_root = os.path.dirname(os.path.abspath(__file__))
            drivers_complete = all(
                self.driver_files_complete(
                    os.path.join(app_root, "drivers", driver),
                    os.path.join(steamvr_driver_path, driver)
                )
                for driver in ["vmt", "vrto3d"]
            )
            vrcfacetracking_complete = self.file_matches(
                os.path.join(app_root, "drivers", "VRCFT-MediapipePro.dll"),
                vrcfacetracking_module_path
            )
            if drivers_complete and vrcfacetracking_complete:
                return True, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path
            else:
                return False, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path
        except Exception as e:
            print(f"Error accessing registry or file system: {e}")
            return False, None, None, None

    def populate_priority_list(self):
        for priority_key, label in PROCESS_PRIORITY_OPTIONS:
            self.priority_selection.addItem(tr(label), priority_key)
        self.update_priority_selection()

    def update_priority_selection(self):
        priority_key = g.config["Setting"]["priority"]
        index = self.priority_selection.findData(priority_key)
        self.priority_selection.setCurrentIndex(index if index >= 0 else 0)

    def set_process_priority(self):
        priority_key = self.priority_selection.currentData()
        print(priority_key)
        # Define a mapping of priority indexes to their corresponding priority classes
        priority_classes = {
            "IDLE_PRIORITY_CLASS": 0x00000040,
            "BELOW_NORMAL_PRIORITY_CLASS": 0x00004000,
            "NORMAL_PRIORITY_CLASS": 0x00000020,  # NORMAL_PRIORITY_CLASS
            "ABOVE_NORMAL_PRIORITY_CLASS": 0x00008000,
            "HIGH_PRIORITY_CLASS": 0x00000080,
            "REALTIME_PRIORITY_CLASS": 0x00000100
        }
        # Check if the index is valid
        if priority_key not in priority_classes:
            self.display_message("Error","Invalid priority index")
            return False
        priority_class = priority_classes[priority_key]
        current_pid = os.getpid()  # Get the current process ID
        handle = windll.kernel32.OpenProcess(0x0200 | 0x0400, False, current_pid)  # Open the current process
        success = windll.kernel32.SetPriorityClass(handle, priority_class)
        windll.kernel32.CloseHandle(handle)
        print("Finished setting priority")
        g.config["Setting"]["priority"] = priority_key

    def update_model_provider_selection(self):
        self.model_provider_selection.setCurrentIndex(
            self.model_provider_selection.findText(g.config["Model"]["provider"])
        )

    def update_model_provider(self):
        provider = self.model_provider_selection.currentText()
        if provider:
            g.config["Model"]["provider"] = provider
            print(f"Model provider updated to: {provider}")

    def display_message(self,title,message,style="",icon=QMessageBox.Critical):
        msg_box = QMessageBox()
        msg_box.setIcon(icon)
        msg_box.setText(tr(message))
        msg_box.setWindowTitle(tr(title))
        msg_box.setStyleSheet(style)
        msg_box.exec_()
        return

    def install_function(self):
        try:
            self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = (
                self.install_checking()
            )
            self.steamvr_installed = check_steamvr_path is not None
            if self.steamvr_installed:
                self.steamvr_status_label.setText(tr("SteamVR Installed"))
                self.steamvr_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.steamvr_status_label.setText(tr("SteamVR Not Installed"))
                self.steamvr_status_label.setStyleSheet("color: red; font-weight: bold;")

            if steamvr_driver_path is None or check_steamvr_path is None:
                self.display_message("Error", "SteamVR is not installed or could not be found.")
                return

            if self.install_state:
                # Uninstall process
                dll_path = os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll")

                error_occurred = False
                drivers_to_remove = ["vmt", "vrto3d"]
                for driver in drivers_to_remove:
                    dir_path = os.path.join(steamvr_driver_path, driver)
                    try:
                        shutil.rmtree(dir_path)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        error_occurred = True
                    if os.path.exists(dir_path):
                        error_occurred = True
                if error_occurred:
                    self.display_message("Error", "SteamVR is running, Please close SteamVR and try again.")
                    return
                try:
                    os.remove(dll_path)
                except FileNotFoundError:
                    pass
                except PermissionError:
                    self.display_message("Error", "VRCFT is running, please close VRCFT and try again.")
                    return
                self.install_state = False
                self.install_button.setText(tr("Install Drivers"))
                self.install_button.setStyleSheet("QPushButton { background-color: blue; color: white; }")
            else:
                # Install process
                app_root = os.path.dirname(os.path.abspath(__file__))
                for driver in ["vmt", "vrto3d"]:
                    source = os.path.join(app_root, "drivers", driver)
                    destination = os.path.join(steamvr_driver_path, driver)
                    if not os.path.exists(destination):
                        shutil.copytree(source, destination)
                    else:
                        self.copy_changed_tree(source, destination)
                dll_source = os.path.join(app_root, "drivers", "VRCFT-MediapipePro.dll")
                dll_destination = os.path.join(
                    vrcfacetracking_path, "VRCFT-MediapipePro.dll"
                )
                if not os.path.exists(dll_destination):
                    os.makedirs(os.path.dirname(dll_destination), exist_ok=True)
                    shutil.copy(dll_source, dll_destination)
                else:
                    self.copy_changed_file(dll_source, dll_destination)
                self.install_state = True
                self.install_button.setText(tr("Uninstall Drivers"))
                self.install_button.setStyleSheet("")
        except (PermissionError, OSError) as exc:
            self.display_message(
                "Error",
                tr("Could not install/update drivers. Close SteamVR/VRCFaceTracking and try again.") + "\n\n" + str(exc),
            )

    def toggle_camera(self):
        self.update_checkboxes()
        self.update_sliders()
        self.update_camera_performance()
        self.update_model_provider()
        if self.video_thread and self.video_thread.isRunning():
            stop_hotkeys()
            self.toggle_button.setText(tr("Start Tracking"))
            self.toggle_button.setStyleSheet(
                "QPushButton { background-color: green; color: white; }"
            )
            self.thread_stopped()
        else:
            apply_hotkeys()
            self.toggle_button.setText(tr("Stop Tracking"))
            self.toggle_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }"
            )
            ip_camera_url = g.config["Setting"]["camera_ip"]
            selected_camera_name = self.camera_selection.currentText()
            source = (
                ip_camera_url
                if ip_camera_url != ""
                else self.get_camera_source(selected_camera_name)
            )
            self.update_camera_performance()
            self.video_thread = VideoCaptureThread(source,g.config["Setting"]["camera_width"],g.config["Setting"]["camera_height"],g.config["Setting"]["camera_fps"])
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.start()

            # controller
            self.controller_thread = ControllerApp()
            self.controller_thread.controller_connection_changed.connect(self.on_controller_connection_changed)
            self.controller_thread.start()


        self.show_frame_button.setText(tr("Show Frame"))

    def toggle_video_display(self):
        if self.video_thread:
            if self.video_thread.show_image:
                self.video_thread.show_image = False
                self.show_frame_button.setText(tr("Show Frame"))
            else:
                self.video_thread.show_image = True
                self.show_frame_button.setText(tr("Hide Frame"))
        else:
            self.show_frame_button.setText(tr("Show Frame"))
        self.image_label.setPixmap(QPixmap())

    def get_camera_source(self, selected_camera_name):
        devices = enumerate_cameras(cv2.CAP_ANY)
        for device in devices:
            if device.index > 1000:
                device.name += " (MSMF)"
            else:
                device.name += " (DSHOW)"
        for device in devices:
            if device.name == selected_camera_name:
                return device.index
        return 0

    def update_frame(self, image):
        if self.video_thread and self.video_thread.show_image:
            target_width = self.image_label.width()
            target_height = self.image_label.height()

            scaled_image = image.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))
            self.image_label.setAlignment(Qt.AlignCenter)

    def populate_camera_list(self):
        devices = enumerate_cameras(cv2.CAP_ANY)
        dshow_devices = []
        msmf_devices = []
        for device in devices:
            if device.index > 1000:
                device.name += " (MSMF)"
                msmf_devices.append(device)
            else:
                device.name += " (DSHOW)"
                dshow_devices.append(device)
        for device in dshow_devices + msmf_devices:
            self.camera_selection.addItem(device.name)

    def populate_camera_performance_list(self):
        for index, (label, _) in enumerate(CAMERA_PERFORMANCE_PRESETS, start=1):
            self.camera_performance_selection.addItem(label, index)
        preset_index = self.resolve_camera_performance_index()
        self.camera_performance_selection.setCurrentIndex(preset_index - 1)
        self.update_camera_performance()

    def populate_camera_aspect_list(self):
        self.camera_aspect_selection.addItems(["16:9", "4:3"])
        configured_aspect = g.config["Setting"].get("camera_aspect", "4:3")
        index = self.camera_aspect_selection.findText(configured_aspect)
        self.camera_aspect_selection.setCurrentIndex(index if index >= 0 else 0)

    def resolve_camera_performance_index(self):
        configured_index = g.config["Setting"].get("camera_performance")
        if configured_index is not None:
            try:
                configured_index = int(configured_index)
                if 1 <= configured_index <= len(CAMERA_PERFORMANCE_PRESETS):
                    return configured_index
            except (TypeError, ValueError):
                pass

        config_width = int(g.config["Setting"]["camera_width"])
        config_height = int(g.config["Setting"]["camera_height"])
        for index, (_, resolutions) in enumerate(CAMERA_PERFORMANCE_PRESETS, start=1):
            if (config_width, config_height) in resolutions.values():
                return index
        return 3

    def resolve_camera_aspect(self):
        selected_aspect = g.config["Setting"].get("camera_aspect", "4:3")
        if selected_aspect in ("4:3", "16:9"):
            return selected_aspect
        return "4:3"

    def update_camera_performance(self, camera_aspect=None):
        g.config["Setting"]["camera_aspect"] = self.camera_aspect_selection.currentText()
        current_performance = self.camera_performance_selection.currentData()
        if current_performance:
            index = current_performance
            aspect = camera_aspect or self.resolve_camera_aspect()
            _, resolutions = CAMERA_PERFORMANCE_PRESETS[index - 1]
            width, height = resolutions[aspect]
            g.config["Setting"]["camera_performance"] = index
            g.config["Setting"]["camera_width"] = width
            g.config["Setting"]["camera_height"] = height
            g.config["Setting"]["camera_fps"] = CAMERA_TARGET_FPS
            print(f"Camera performance updated to: {width}x{height} / {CAMERA_TARGET_FPS} FPS ({aspect})")

    def update_camera_ip(self, value):
        g.config["Setting"]["camera_ip"] = value 

    def thread_stopped(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        if self.controller_thread:
            self.controller_thread.stop()
            self.controller_thread.wait()
            self.controller_thread = None
        self.image_label.setPixmap(QPixmap())

    def closeEvent(self, event):
        self.thread_stopped()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    install_language(app, g.config["Setting"].get("language", LANGUAGE_SYSTEM))
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
