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
    QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

import cv2
from pygrabber.dshow_graph import FilterGraph
import sys, os, winreg, shutil
from utils.tracking import Tracker
from utils.actions import *
import utils.globals as g
from tracker.face.face import draw_face_landmarks
from tracker.face.tongue import draw_tongue_position
from tracker.hand.hand import draw_hand_landmarks
import win32api
import win32process
import win32con

handle = win32api.GetCurrentProcess()
win32process.SetPriorityClass(handle, win32con.REALTIME_PRIORITY_CLASS)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(QImage)
    stopped = pyqtSignal()

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.video_capture = None
        self.is_running = True
        self.show_image = False
        self.tracker = Tracker()

    def run(self):
        self.video_capture = cv2.VideoCapture(self.source)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.is_running:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if g.config["Sending"]["flip_x"]:
                    rgb_image = cv2.flip(rgb_image, 1)
                if g.config["Sending"]["flip_y"]:
                    rgb_image = cv2.flip(rgb_image, 0)

                self.tracker.process_frames(rgb_image)
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
        self.stopped.emit()

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "ExVR - Experience Virtual Reality"
        )
        self.setFixedSize(800, 1000)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.resize(800, 500)
        layout.addWidget(self.image_label)

        flip_layout = QHBoxLayout()  # Create a QHBoxLayout for new reset buttons
        self.flip_x_checkbox = QCheckBox("Flip X", self)
        self.flip_x_checkbox.clicked.connect(self.flip_x)
        self.flip_x_checkbox.setChecked(g.config["Sending"]["flip_x"])
        flip_layout.addWidget(self.flip_x_checkbox)

        self.flip_y_checkbox = QCheckBox("Flip Y", self)
        self.flip_y_checkbox.clicked.connect(self.flip_y)
        self.flip_y_checkbox.setChecked(g.config["Sending"]["flip_y"])
        flip_layout.addWidget(self.flip_y_checkbox)
        layout.addLayout(flip_layout)

        self.ip_camera_url_input = QLineEdit(self)
        self.ip_camera_url_input.setPlaceholderText("Enter IP camera URL")
        layout.addWidget(self.ip_camera_url_input)

        self.camera_selection = QComboBox(self)
        self.populate_camera_list()
        layout.addWidget(self.camera_selection)

        self.install_state, steamvr_driver_path, vrcfacetracking_path = self.install_checking()
        if steamvr_driver_path is None or vrcfacetracking_path is None:
            self.install_button = QPushButton("Please Install SteamVR", self)
            self.install_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }")
        elif self.install_state:
            self.install_button = QPushButton("Uninstall Drivers", self)
            self.install_button.setStyleSheet("")
        else:
            self.install_button = QPushButton("Install Drivers", self)
            self.install_button.setStyleSheet(
                "QPushButton { background-color: blue; color: white; }"
            )
        self.install_button.clicked.connect(self.install_function)
        layout.addWidget(self.install_button)

        self.toggle_button = QPushButton("Start Tracking", self)
        self.toggle_button.setStyleSheet(
            "QPushButton { background-color: green; color: white; }"
        )
        self.toggle_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.toggle_button)

        self.show_frame_button = QPushButton("Show Frame", self)
        self.show_frame_button.clicked.connect(self.toggle_video_display)
        layout.addWidget(self.show_frame_button)

        separator_0 = QFrame(self)
        separator_0.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_0.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_0)

        reset_layout = QHBoxLayout()  # Create a QHBoxLayout for new reset buttons
        self.reset_head = QPushButton("Reset Head", self)
        self.reset_head.clicked.connect(reset_head)
        reset_layout.addWidget(self.reset_head)
        self.reset_eyes = QPushButton("Reset Eyes", self)
        self.reset_eyes.clicked.connect(reset_eye)
        reset_layout.addWidget(self.reset_eyes)
        self.reset_l_hand = QPushButton("Reset LHand", self)
        self.reset_l_hand.clicked.connect(lambda: reset_hand(True))
        reset_layout.addWidget(self.reset_l_hand)
        self.reset_r_hand = QPushButton("Reset RHand", self)
        self.reset_r_hand.clicked.connect(lambda: reset_hand(False))
        reset_layout.addWidget(self.reset_r_hand)
        layout.addLayout(reset_layout)

        checkbox_layout = QHBoxLayout()
        self.checkbox1 = QCheckBox("Head", self)
        self.checkbox1.clicked.connect(
            lambda: self.update_config("Head", self.checkbox1.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox1)
        self.checkbox2 = QCheckBox("Face", self)
        self.checkbox2.clicked.connect(
            lambda: self.update_config("Face", self.checkbox2.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox2)
        self.checkbox3 = QCheckBox("Tongue", self)
        self.checkbox3.clicked.connect(
            lambda: self.update_config("Tongue", self.checkbox3.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox3)
        self.checkbox4 = QCheckBox("Hand", self)
        self.checkbox4.clicked.connect(
            lambda: self.update_config("Hand", self.checkbox4.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox4)
        layout.addLayout(checkbox_layout)

        checkbox_layout_1 = QHBoxLayout()
        self.block_face_checkbox = QCheckBox("Block Face", self)
        self.block_face_checkbox.clicked.connect(self.set_face_block)
        self.block_face_checkbox.setChecked(g.config["Tracking"]["Face"]["block"])
        checkbox_layout_1.addWidget(self.block_face_checkbox)

        self.only_front_checkbox = QCheckBox("Front Hand", self)
        self.only_front_checkbox.clicked.connect(self.set_hand_front)
        self.only_front_checkbox.setChecked(g.config["Tracking"]["Hand"]["only_front"])
        checkbox_layout_1.addWidget(self.only_front_checkbox)
        layout.addLayout(checkbox_layout_1)


        self.update_checkboxes()

        slider_layout = QHBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 200)
        self.slider2.setRange(0, 200)
        self.slider3.setRange(0, 100)
        self.slider1.setSingleStep(1)
        self.slider2.setSingleStep(1)
        self.slider3.setSingleStep(1)
        self.label0 = QLabel("Hand Scalar:")
        self.label1 = QLabel(f"x {g.config['Tracking']['Hand']['x_scalar']:.2f}")
        self.label2 = QLabel(f"y {g.config['Tracking']['Hand']['y_scalar']:.2f}")
        self.label3 = QLabel(f"z {g.config['Tracking']['Hand']['z_scalar']:.2f}")
        self.slider1.valueChanged.connect(lambda value: self.set_scalar(value, "x"))
        self.slider2.valueChanged.connect(lambda value: self.set_scalar(value, "y"))
        self.slider3.valueChanged.connect(lambda value: self.set_scalar(value, "z"))
        slider_layout.addWidget(self.label0)
        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.slider1)
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.slider2)
        slider_layout.addWidget(self.label3)
        slider_layout.addWidget(self.slider3)
        layout.addLayout(slider_layout)

        self.update_sliders()

        separator_1 = QFrame(self)
        separator_1.setFrameShape(
            QFrame.HLine
        )  # Set the frame to be a horizontal line
        separator_1.setFrameShadow(QFrame.Sunken)  # Give it a sunken shadow effect
        layout.addWidget(separator_1)

        config_layout = QHBoxLayout()
        self.update_config_button = QPushButton("Update Config", self)
        self.update_config_button.clicked.connect(lambda:(g.update_configs(),self.update_checkboxes(), self.update_sliders()))
        config_layout.addWidget(self.update_config_button)
        self.save_config_button = QPushButton("Save Config", self)
        self.save_config_button.clicked.connect(g.save_configs)
        config_layout.addWidget(self.save_config_button)
        layout.addLayout(config_layout)

        self.video_thread = None

    def flip_x(self, value):
        g.config["Sending"]["flip_x"] = value

    def flip_y(self, value):
        g.config["Sending"]["flip_y"] = value

    def set_hand_front(self, value):
        g.config["Tracking"]["Hand"]["only_front"] = value

    def set_face_block(self, value):
        g.config["Tracking"]["Face"]["block"] = value

    def update_checkboxes(self):
        self.flip_x_checkbox.setChecked(g.config["Sending"]["flip_x"])
        self.flip_y_checkbox.setChecked(g.config["Sending"]["flip_y"])
        self.checkbox1.setChecked(g.config["Tracking"]["Head"]["enable"])
        self.checkbox2.setChecked(g.config["Tracking"]["Face"]["enable"])
        self.checkbox3.setChecked(g.config["Tracking"]["Tongue"]["enable"])
        self.checkbox4.setChecked(g.config["Tracking"]["Hand"]["enable"])
        self.block_face_checkbox.setChecked(g.config["Tracking"]["Face"]["block"])
        self.only_front_checkbox.setChecked(g.config["Tracking"]["Hand"]["only_front"])

    def set_scalar(self, value, axis):
        scalar_value = value / 100.0
        if axis == "x":
            g.config["Tracking"]["Hand"]["x_scalar"] = scalar_value
            self.label1.setText(f"x {scalar_value:.2f}")
        elif axis == "y":
            g.config["Tracking"]["Hand"]["y_scalar"] = scalar_value
            self.label2.setText(f"y {scalar_value:.2f}")
        elif axis == "z":
            g.config["Tracking"]["Hand"]["z_scalar"] = scalar_value
            self.label3.setText(f"z {scalar_value:.2f}")

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

    def update_config(self, key, value):
        if key in g.config["Tracking"]:
            g.config["Tracking"][key]["enable"] = value

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
            vrcfacetracking_path = os.path.join(
                os.getenv("APPDATA"), "VRCFaceTracking", "CustomLibs"
            )
            vrcfacetracking_module_path = os.path.join(
                vrcfacetracking_path, "VRCFT-MediapipePro.dll"
            )

            # Check all required paths
            required_paths = [vrcfacetracking_module_path] + [
                os.path.join(steamvr_driver_path, driver)
                for driver in ["vmt", "vrto3d"]
            ]
            if all(os.path.exists(path) for path in required_paths):
                return True, steamvr_driver_path, vrcfacetracking_path
            else:
                return False, steamvr_driver_path, vrcfacetracking_path
        except Exception as e:
            print(f"Error accessing registry or file system: {e}")
            return False, None, None

    def install_function(self):
        self.install_state, steamvr_driver_path, vrcfacetracking_path = (
            self.install_checking()
        )
        if steamvr_driver_path is None or vrcfacetracking_path is None:
            self.install_button.setText("Please Install SteamVR")
            self.install_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }")
        elif self.install_state:
            # Uninstall process
            shutil.rmtree(os.path.join(steamvr_driver_path, "vmt"), ignore_errors=True)
            shutil.rmtree(
                os.path.join(steamvr_driver_path, "vrto3d"), ignore_errors=True
            )
            os.remove(os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll"))
            self.install_button.setText("Install Drivers")
            self.install_button.setStyleSheet(
                "QPushButton { background-color: blue; color: white; }"
            )
        else:
            # Install process
            for driver in ["vmt", "vrto3d"]:
                source = os.path.join("./drivers", driver)
                destination = os.path.join(steamvr_driver_path, driver)
                if not os.path.exists(destination):
                    shutil.copytree(source, destination)
            dll_source = os.path.join("./drivers", "VRCFT-MediapipePro.dll")
            dll_destination = os.path.join(
                vrcfacetracking_path, "VRCFT-MediapipePro.dll"
            )
            if not os.path.exists(dll_destination):
                os.makedirs(os.path.dirname(dll_destination), exist_ok=True)
                shutil.copy(dll_source, dll_destination)
            self.install_button.setText("Uninstall Drivers")
            self.install_button.setStyleSheet("")

    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.toggle_button.setText("Start Tracking")
            self.toggle_button.setStyleSheet(
                "QPushButton { background-color: green; color: white; }"
            )
            self.thread_stopped()
        else:
            self.toggle_button.setText("Stop Tracking")
            self.toggle_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }"
            )
            ip_camera_url = self.ip_camera_url_input.text()
            selected_camera_name = self.camera_selection.currentText()
            source = (
                ip_camera_url
                if ip_camera_url != ""
                else self.get_camera_source(selected_camera_name)
            )
            self.video_thread = VideoCaptureThread(source)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.stopped.connect(self.thread_stopped)
            self.video_thread.start()
        self.show_frame_button.setText("Show Frame")

    def toggle_video_display(self):
        if self.video_thread:
            if self.video_thread.show_image:
                self.video_thread.show_image = False
                self.show_frame_button.setText("Show Frame")
            else:
                self.video_thread.show_image = True
                self.show_frame_button.setText("Hide Frame")
        else:
            self.show_frame_button.setText("Show Frame")
        self.image_label.setPixmap(QPixmap())

    def get_camera_source(self, selected_camera_name):
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return (
            devices.index(selected_camera_name)
            if selected_camera_name in devices
            else 0
        )

    def update_frame(self, image):
        if self.video_thread:
            if self.video_thread.show_image:
                p = image.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                )
                self.image_label.setPixmap(QPixmap.fromImage(p))
                self.image_label.setAlignment(Qt.AlignCenter)

    def populate_camera_list(self):
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for device in devices:
            self.camera_selection.addItem(device)

    def thread_stopped(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        self.image_label.setPixmap(QPixmap())

    def closeEvent(self, event):
        self.thread_stopped()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
