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
    QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator

import cv2
from pygrabber.dshow_graph import FilterGraph
import sys, os, winreg, shutil
import utils.tracking
from utils.actions import *
import utils.globals as g
from utils.data import setup_data,save_data
from utils.hotkeys import stop_hotkeys, apply_hotkeys
from tracker.face.face import draw_face_landmarks
from tracker.face.tongue import draw_tongue_position
from tracker.hand.hand import draw_hand_landmarks
from ctypes import windll
from cv2_enumerate_cameras import enumerate_cameras

# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(QImage)  # 信号：图像帧准备就绪
    stopped = pyqtSignal()  # 信号：线程停止

    def __init__(self, source):
        super().__init__()
        self.source = source  # 视频源
        self.video_capture = None  # 视频捕获对象
        self.is_running = True  # 线程运行标志
        self.show_image = False  # 是否显示图像
        self.tracker = utils.tracking.Tracker()  # 初始化跟踪器

    def run(self):
        self.video_capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)  # 创建视频捕获对象
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小
        while self.is_running:
            ret, frame = self.video_capture.read()  # 读取视频帧
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                if g.config["Setting"]["flip_x"]:  # 水平翻转图像
                    rgb_image = cv2.flip(rgb_image, 1)
                if g.config["Setting"]["flip_y"]:  # 垂直翻转图像
                    rgb_image = cv2.flip(rgb_image, 0)

                self.tracker.process_frames(rgb_image)  # 处理图像帧
                if self.show_image:
                    if g.config["Tracking"]["Head"]["enable"] or g.config["Tracking"]["Face"]["enable"]:  # 处理人脸标志
                        rgb_image = draw_face_landmarks(rgb_image)
                    if g.config["Tracking"]["Tongue"]["enable"]:  # 处理舌头位置
                        rgb_image = draw_tongue_position(rgb_image)
                    if g.config["Tracking"]["Hand"]["enable"]:  # 处理手部标志
                        rgb_image = draw_hand_landmarks(rgb_image)
                    h, w, ch = rgb_image.shape  # 获取图像的高度、宽度和通道
                    bytes_per_line = ch * w  # 每行字节数
                    convert_to_Qt_format = QImage(  # 转换为Qt格式
                        rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                    )
                    self.frame_ready.emit(convert_to_Qt_format)  # 发出信号，图像帧准备就绪
        self.cleanup()  # 清理资源

    def stop(self):
        self.is_running = False  # 停止线程
        self.tracker.stop()  # 停止跟踪器

    def cleanup(self):
        if self.video_capture:
            self.video_capture.release()  # 释放视频捕获对象
        self.stopped.emit()  # 发出停止信号

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen = QApplication.screens()[0]
        screen_size = screen.size()
        self.width = int(screen_size.width() * 0.3)
        self.height = int(screen_size.height() * 0.65)
        self.half_height = int(self.height / 2)

        self.setWindowTitle(
            "ExVR - 体验虚拟现实"
        )
        # self.setFixedSize(width, height)  # 设置固定大小
        self.resize(self.width, self.height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.resize(self.width, self.half_height)
        layout.addWidget(self.image_label)

        flip_layout = QHBoxLayout()  # 创建一个 QHBoxLayout 用于新的重置按钮
        self.flip_x_checkbox = QCheckBox("翻转 X", self)
        self.flip_x_checkbox.clicked.connect(self.flip_x)
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        flip_layout.addWidget(self.flip_x_checkbox)

        self.flip_y_checkbox = QCheckBox("翻转 Y", self)
        self.flip_y_checkbox.clicked.connect(self.flip_y)
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        flip_layout.addWidget(self.flip_y_checkbox)
        layout.addLayout(flip_layout)

        self.ip_camera_url_input = QLineEdit(self)
        self.ip_camera_url_input.setPlaceholderText("请输入 IP 摄像头 URL")
        layout.addWidget(self.ip_camera_url_input)

        self.camera_selection = QComboBox(self)
        self.populate_camera_list()
        layout.addWidget(self.camera_selection)

        # 创建一个下拉框用于选择优先级
        self.priority_selection = QComboBox(self)
        # 添加优先级选项
        self.priority_selection.addItems(["IDLE_PRIORITY_CLASS", "BELOW_NORMAL_PRIORITY_CLASS", "NORMAL_PRIORITY_CLASS", "ABOVE_NORMAL_PRIORITY_CLASS", "HIGH_PRIORITY_CLASS", "REALTIME_PRIORITY_CLASS"])
        # 连接当前索引变化信号到设置进程优先级的方法
        self.priority_selection.currentIndexChanged.connect(self.set_process_priority)
        # 将下拉框添加到布局中
        layout.addWidget(self.priority_selection)
        # 设置下拉框当前选中的优先级
        self.priority_selection.setCurrentIndex(self.priority_selection.findText(g.config["Setting"]["priority"]))

        # 检查安装状态以及路径
        self.install_state, steamvr_driver_path, vrcfacetracking_path = self.install_checking()
        # 如果路径为空，提示用户安装SteamVR
        if steamvr_driver_path is None or vrcfacetracking_path is None:
            self.install_button = QPushButton("请安装SteamVR", self)
            self.install_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }")  # 设置按钮样式
        # 如果安装状态为真，显示卸载驱动按钮
        elif self.install_state:
            self.install_button = QPushButton("卸载驱动", self)
            self.install_button.setStyleSheet("")
        # 否则显示安装驱动按钮
        else:
            self.install_button = QPushButton("安装驱动", self)
            self.install_button.setStyleSheet(
                "QPushButton { background-color: blue; color: white; }"
            )
        # 连接按钮点击事件到安装功能
        self.install_button.clicked.connect(self.install_function)
        # 将安装按钮添加到布局中
        layout.addWidget(self.install_button)

        # 创建一个按钮用于开始追踪
        self.toggle_button = QPushButton("开始追踪", self)
        self.toggle_button.setStyleSheet(
            "QPushButton { background-color: green; color: white; }"
        )
        # 连接按钮点击事件到切换相机的方法
        self.toggle_button.clicked.connect(self.toggle_camera)
        # 将开始追踪按钮添加到布局中
        layout.addWidget(self.toggle_button)

        # 创建一个按钮用于显示帧
        self.show_frame_button = QPushButton("显示帧", self)
        # 连接按钮点击事件到切换视频显示的方法
        self.show_frame_button.clicked.connect(self.toggle_video_display)
        # 将显示帧按钮添加到布局中
        layout.addWidget(self.show_frame_button)

        separator_0 = QFrame(self)
        separator_0.setFrameShape(
            QFrame.HLine
        )  # 将框架设置为水平线
        separator_0.setFrameShadow(QFrame.Sunken)  # 给予一个凹陷的阴影效果
        layout.addWidget(separator_0)

        reset_layout = QHBoxLayout()  # 创建一个 QHBoxLayout 用于新的重置按钮
        self.reset_head = QPushButton("重置头部", self)
        self.reset_head.clicked.connect(reset_head)
        reset_layout.addWidget(self.reset_head)
        self.reset_eyes = QPushButton("重置眼睛", self)
        self.reset_eyes.clicked.connect(reset_eye)
        reset_layout.addWidget(self.reset_eyes)
        self.reset_l_hand = QPushButton("重置左手", self)
        self.reset_l_hand.clicked.connect(lambda: reset_hand(True))
        reset_layout.addWidget(self.reset_l_hand)
        self.reset_r_hand = QPushButton("重置右手", self)
        self.reset_r_hand.clicked.connect(lambda: reset_hand(False))
        reset_layout.addWidget(self.reset_r_hand)
        layout.addLayout(reset_layout)

        checkbox_layout = QHBoxLayout()
        self.checkbox1 = QCheckBox("头部", self)
        self.checkbox1.clicked.connect(
            lambda: self.update_config("Head", self.checkbox1.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox1)
        self.checkbox2 = QCheckBox("面部", self)
        self.checkbox2.clicked.connect(
            lambda: self.update_config("Face", self.checkbox2.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox2)
        self.checkbox3 = QCheckBox("舌头", self)
        self.checkbox3.clicked.connect(
            lambda: self.update_config("Tongue", self.checkbox3.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox3)
        self.checkbox4 = QCheckBox("手", self)
        self.checkbox4.clicked.connect(
            lambda: self.update_config("Hand", self.checkbox4.isChecked())
        )
        checkbox_layout.addWidget(self.checkbox4)
        layout.addLayout(checkbox_layout)

        checkbox_layout_1 = QHBoxLayout()
        self.block_face_checkbox = QCheckBox("阻止面部", self)
        self.block_face_checkbox.clicked.connect(self.set_face_block)
        self.block_face_checkbox.setChecked(g.config["Tracking"]["Face"]["block"])
        checkbox_layout_1.addWidget(self.block_face_checkbox)

        self.only_front_checkbox = QCheckBox("前手", self)
        self.only_front_checkbox.clicked.connect(self.set_hand_front)
        self.only_front_checkbox.setChecked(g.config["Tracking"]["Hand"]["only_front"])
        checkbox_layout_1.addWidget(self.only_front_checkbox)
        layout.addLayout(checkbox_layout_1)


        self.update_checkboxes()  # 更新复选框状态

        slider_layout = QHBoxLayout()  # 创建一个水平布局
        self.slider1 = QSlider(Qt.Horizontal)  # 创建水平滑块1
        self.slider2 = QSlider(Qt.Horizontal)  # 创建水平滑块2
        self.slider3 = QSlider(Qt.Horizontal)  # 创建水平滑块3
        self.slider1.setRange(0, 200)  # 设置滑块1的范围
        self.slider2.setRange(0, 200)  # 设置滑块2的范围
        self.slider3.setRange(0, 100)  # 设置滑块3的范围
        self.slider1.setSingleStep(1)  # 设置滑块1的单步变化值
        self.slider2.setSingleStep(1)  # 设置滑块2的单步变化值
        self.slider3.setSingleStep(1)  # 设置滑块3的单步变化值
        self.label0 = QLabel("手部缩放:")  # 创建标签0
        self.label1 = QLabel(f"x {g.config['Tracking']['Hand']['x_scalar']:.2f}")  # 创建标签1，显示x轴缩放值
        self.label2 = QLabel(f"y {g.config['Tracking']['Hand']['y_scalar']:.2f}")  # 创建标签2，显示y轴缩放值
        self.label3 = QLabel(f"z {g.config['Tracking']['Hand']['z_scalar']:.2f}")  # 创建标签3，显示z轴缩放值
        self.slider1.valueChanged.connect(lambda value: self.set_scalar(value, "x"))  # 连接滑块1的值变化信号
        self.slider2.valueChanged.connect(lambda value: self.set_scalar(value, "y"))  # 连接滑块2的值变化信号
        self.slider3.valueChanged.connect(lambda value: self.set_scalar(value, "z"))  # 连接滑块3的值变化信号
        slider_layout.addWidget(self.label0)  # 将标签0添加到布局中
        slider_layout.addWidget(self.label1)  # 将标签1添加到布局中
        slider_layout.addWidget(self.slider1)  # 将滑块1添加到布局中
        slider_layout.addWidget(self.label2)  # 将标签2添加到布局中
        slider_layout.addWidget(self.slider2)  # 将滑块2添加到布局中
        slider_layout.addWidget(self.label3)  # 将标签3添加到布局中
        slider_layout.addWidget(self.slider3)  # 将滑块3添加到布局中
        layout.addLayout(slider_layout)  # 将滑块布局添加到主布局中

        self.update_sliders()  # 更新滑块状态

        separator_1 = QFrame(self)
        separator_1.setFrameShape(
            QFrame.HLine
        )  # 设置框架为水平线
        separator_1.setFrameShadow(QFrame.Sunken)  # 给它一个凹陷的阴影效果
        layout.addWidget(separator_1)

        config_layout = QHBoxLayout()
        self.reset_hotkey_button = QPushButton("重置热键", self)
        self.reset_hotkey_button.clicked.connect(self.reset_hotkeys)
        config_layout.addWidget(self.reset_hotkey_button)
        self.stop_hotkey_button = QPushButton("停止热键", self)
        self.stop_hotkey_button.clicked.connect(stop_hotkeys)
        config_layout.addWidget(self.stop_hotkey_button)
        self.set_face_button = QPushButton("设置面孔", self)
        self.set_face_button.clicked.connect(self.face_dialog)
        config_layout.addWidget(self.set_face_button)
        self.update_config_button = QPushButton("更新配置", self)
        self.update_config_button.clicked.connect(lambda:(g.update_configs(),self.update_checkboxes(), self.update_sliders()))
        config_layout.addWidget(self.update_config_button)
        self.save_config_button = QPushButton("保存配置", self)
        self.save_config_button.clicked.connect(g.save_configs)
        config_layout.addWidget(self.save_config_button)
        layout.addLayout(config_layout)

        self.video_thread = None

    def save_data(self):
        # 保存数据的方法
        for i, (key, edits) in enumerate(self.lineEdits.items()):
            idx = i + 1  # 计算索引
            v = float(edits[0].text())  # 获取并转换值输入
            s = float(edits[1].text())  # 获取并转换偏移输入
            w = float(edits[2].text())  # 获取并转换权重输入
            max_value = float(edits[3].text())  # 获取并转换最大值输入
            e = self.checkBoxes[key].isChecked()  # 获取启用状态
            # 将数据存储到全局变量g中
            g.data["BlendShapes"][idx]["v"] = v
            g.data["BlendShapes"][idx]["s"] = s
            g.data["BlendShapes"][idx]["w"] = w
            g.data["BlendShapes"][idx]["max"] = max_value
            g.data["BlendShapes"][idx]["e"] = e
        save_data(g.data)  # 保存数据到文件
        self.dialog.close()  # 关闭对话框

    def face_dialog(self):
        # 创建面部设置的对话框
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("面部设置")  # 设置对话框标题
        self.dialog.resize(self.width, self.height)  # 设置对话框的固定大小

        layout = QVBoxLayout(self.dialog)  # 使用垂直布局

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 使滚动区域可调整大小

        # 创建一个窗口小部件以容纳表单布局
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)  # 使用网格布局以获得更好的对齐

        # 为表单添加头部标签
        form_layout.addWidget(QLabel("BlendShape"), 0, 0)
        form_layout.addWidget(QLabel("值"), 0, 1)
        form_layout.addWidget(QLabel("偏移"), 0, 2)
        form_layout.addWidget(QLabel("权重"), 0, 3)
        form_layout.addWidget(QLabel("最大值"), 0, 4)
        form_layout.addWidget(QLabel("启用"), 0, 5)

        # 存储QLineEdit和QCheckBox引用
        self.lineEdits = {}
        self.checkBoxes = {}

        # 为每个BlendShape创建输入字段
        double_validator = QDoubleValidator()  # 创建一个双精度验证器
        blendshape_data, _ = setup_data()  # 设置数据
        for i, blendshape in enumerate(blendshape_data["BlendShapes"][1:], start=1):  # 从第一行开始
            key = blendshape["k"]  # 获取BlendShape键
            v_edit = QLineEdit(str(round(blendshape["v"], 2)))  # 创建值输入框
            v_edit.setValidator(double_validator)  # 设置验证器
            s_edit = QLineEdit(str(round(blendshape["s"], 2)))  # 创建偏移输入框
            s_edit.setValidator(double_validator)  # 设置验证器
            w_edit = QLineEdit(str(round(blendshape["w"], 2)))  # 创建权重输入框
            w_edit.setValidator(double_validator)  # 设置验证器
            max_edit = QLineEdit(str(round(blendshape["max"], 2)))  # 创建最大值输入框
            max_edit.setValidator(double_validator)  # 设置验证器
            e_check = QCheckBox()  # 创建复选框
            e_check.setChecked(blendshape["e"])  # 设置复选框的初始状态

            # 保存QLineEdit和QCheckBox的引用
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check

            # 将小部件添加到网格布局中
            form_layout.addWidget(QLabel(key), i, 0)  # BlendShape键在第0列
            form_layout.addWidget(v_edit, i, 1)  # v_edit在第1列
            form_layout.addWidget(s_edit, i, 2)  # s_edit在第2列
            form_layout.addWidget(w_edit, i, 3)  # w_edit在第3列
            form_layout.addWidget(max_edit, i, 4)  # max_edit在第4列
            form_layout.addWidget(e_check, i, 5)  # e_check在第5列

        # 将表单布局添加到滚动区域
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)  # 将滚动区域添加到对话框布局中

        # 添加一个保存配置按钮
        self.save_config_button = QPushButton("保存配置", self.dialog)  # 创建保存按钮
        self.save_config_button.clicked.connect(self.save_data)  # 连接按钮点击事件
        layout.addWidget(self.save_config_button)  # 将按钮添加到布局中

        self.dialog.exec_()  # 显示对话框并等待用户输入

    def flip_x(self, value):
        # 设置是否翻转X轴
        g.config["Setting"]["flip_x"] = value

    def flip_y(self, value):
        # 设置是否翻转Y轴
        g.config["Setting"]["flip_y"] = value

    def set_hand_front(self, value):
        # 设置手部追踪是否只在前面
        g.config["Tracking"]["Hand"]["only_front"] = value

    def set_face_block(self, value):
        # 设置面部追踪阻止标志
        g.config["Tracking"]["Face"]["block"] = value

    def update_checkboxes(self):
        # 更新复选框状态
        self.flip_x_checkbox.setChecked(g.config["Setting"]["flip_x"])
        self.flip_y_checkbox.setChecked(g.config["Setting"]["flip_y"])
        self.checkbox1.setChecked(g.config["Tracking"]["Head"]["enable"])
        self.checkbox2.setChecked(g.config["Tracking"]["Face"]["enable"])
        self.checkbox3.setChecked(g.config["Tracking"]["Tongue"]["enable"])
        self.checkbox4.setChecked(g.config["Tracking"]["Hand"]["enable"])
        self.block_face_checkbox.setChecked(g.config["Tracking"]["Face"]["block"])
        self.only_front_checkbox.setChecked(g.config["Tracking"]["Hand"]["only_front"])

    def set_scalar(self, value, axis):
        # 设置手部追踪的缩放因子
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
        # 更新滑块的值
        x_scalar = g.config["Tracking"]["Hand"]["x_scalar"]
        y_scalar = g.config["Tracking"]["Hand"]["y_scalar"]
        z_scalar = g.config["Tracking"]["Hand"]["z_scalar"]
        self.slider1.setValue(int(x_scalar * 100))
        self.slider2.setValue(int(y_scalar * 100))
        self.slider3.setValue(int(z_scalar * 100))
        self.label1.setText(f"x {x_scalar:.2f}")
        self.label2.setText(f"y {y_scalar:.2f}")
        self.label3.setText(f"z {z_scalar:.2f}")

    def reset_hotkeys(self):
        # 重置热键
        stop_hotkeys()
        apply_hotkeys()
        if self.video_thread is None:
            stop_hotkeys()

    def update_config(self, key, value):
        # 更新配置项
        if key in g.config["Tracking"]:
            g.config["Tracking"][key]["enable"] = value

    def install_checking(self):
        # 检查安装路径
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

            # 检查所有必需路径
            required_paths = [vrcfacetracking_module_path] + [
                os.path.join(steamvr_driver_path, driver)
                for driver in ["vmt", "vrto3d"]
            ]
            if all(os.path.exists(path) for path in required_paths):
                return True, steamvr_driver_path, vrcfacetracking_path
            else:
                return False, steamvr_driver_path, vrcfacetracking_path
        except Exception as e:
            print(f"访问注册表或文件系统时出错: {e}")
            return False, None, None

    def set_process_priority(self):
        priority_key = self.priority_selection.currentText()
        print(priority_key)
        # 定义优先级索引与对应优先级类别的映射
        priority_classes = {
            "IDLE_PRIORITY_CLASS": 0x00000040,
            "BELOW_NORMAL_PRIORITY_CLASS": 0x00004000,
            "NORMAL_PRIORITY_CLASS": 0x00000020,  # 普通优先级
            "ABOVE_NORMAL_PRIORITY_CLASS": 0x00008000,
            "HIGH_PRIORITY_CLASS": 0x00000080,
            "REALTIME_PRIORITY_CLASS": 0x00000100
        }
        # 检查索引是否有效
        if priority_key not in priority_classes:
            self.display_message("错误","无效的优先级索引")
            return False
        priority_class = priority_classes[priority_key]
        current_pid = os.getpid()  # 获取当前进程ID
        handle = windll.kernel32.OpenProcess(0x0200 | 0x0400, False, current_pid)  # 打开当前进程
        success = windll.kernel32.SetPriorityClass(handle, priority_class)
        windll.kernel32.CloseHandle(handle)
        print("完成设置优先级")

    def display_message(self,title,message,style=""):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(style)
        msg_box.exec_()
        return

    def install_function(self):
        self.install_state, steamvr_driver_path, vrcfacetracking_path = (
            self.install_checking()
        )
        if steamvr_driver_path is None or vrcfacetracking_path is None:
            self.install_button.setText("请安装 SteamVR")
            self.install_button.setStyleSheet(
                "QPushButton { background-color: red; color: white; }")
        elif self.install_state:
            # 卸载过程
            dll_path = os.path.join(vrcfacetracking_path, "VRCFT-MediapipePro.dll")
            try:
                os.remove(dll_path)
            except PermissionError:
                self.display_message("错误", "VRCFT 正在运行，请关闭 VRCFT 后重试。")
                return
            shutil.rmtree(os.path.join(steamvr_driver_path, "vmt"), ignore_errors=True)
            shutil.rmtree(os.path.join(steamvr_driver_path, "vrto3d"), ignore_errors=True)
            self.install_button.setText("安装驱动程序")
            self.install_button.setStyleSheet("QPushButton { background-color: blue; color: white; }")
        else:
            # 安装过程
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
            self.install_button.setText("卸载驱动程序")
            self.install_button.setStyleSheet("")

    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.toggle_button.setText("开始跟踪")
            self.toggle_button.setStyleSheet(
                "QPushButton { background-color: green; color: white; }"
            )
            self.thread_stopped()
        else:
            self.toggle_button.setText("停止跟踪")
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
        self.show_frame_button.setText("显示帧")

    def toggle_video_display(self):
        # 切换视频显示状态
        if self.video_thread:
            # 如果视频线程存在
            if self.video_thread.show_image:
                self.video_thread.show_image = False
                self.show_frame_button.setText("显示帧")
            else:
                self.video_thread.show_image = True
                self.show_frame_button.setText("隐藏帧")
        else:
            self.show_frame_button.setText("显示帧")
        self.image_label.setPixmap(QPixmap())

    def get_camera_source(self, selected_camera_name):
        # 获取摄像头源
        devices = enumerate_cameras(cv2.CAP_ANY)
        for device in devices:
            # 为设备添加字符串后缀
            if device.index > 1000:
                device.name += " (MSMF)"
            else:
                device.name += " (DSHOW)"
        for device in devices:
            # 如果设备名称匹配，返回设备索引
            if device.name == selected_camera_name:
                return device.index
        return 0

    def update_frame(self, image):
        # 更新帧图像
        if self.video_thread:
            if self.video_thread.show_image:
                # 缩放图像以适应标签
                p = image.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                )
                self.image_label.setPixmap(QPixmap.fromImage(p))
                self.image_label.setAlignment(Qt.AlignCenter)

    def populate_camera_list(self):
        # 填充摄像头列表
        devices = enumerate_cameras(cv2.CAP_ANY)
        for device in devices:
            # 为设备添加字符串后缀
            if device.index > 1000:
                device.name += " (MSMF)"
            else:
                device.name += " (DSHOW)"
        for device in devices:
            # 将设备名称添加到选择框
            self.camera_selection.addItem(device.name)

    def thread_stopped(self):
    # 线程停止处理
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        self.image_label.setPixmap(QPixmap())

    def closeEvent(self, event):
        # 窗口关闭事件处理
        self.thread_stopped()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
