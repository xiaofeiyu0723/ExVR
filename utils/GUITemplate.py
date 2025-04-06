from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QMessageBox, QLabel


class CustomRow(QWidget):
    def __init__(self, name: str, left_click_handler, right_click_handler):
        super().__init__()
        layout = QHBoxLayout()
        self.button_name = name
        self.btn_left = QPushButton("链接" + name)
        self.btn_left.setFixedSize(150, 40)

        self.text = QLabel()
        self.text.setText("NONE")
        self.text.setFixedHeight(40)
        font = QFont()
        font.setPointSize(8)
        self.text.setFont(font)
        self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_right = QPushButton("X")
        self.btn_right.setFixedSize(40, 40)

        layout.addWidget(self.btn_left)
        layout.addWidget(self.text)
        layout.addWidget(self.btn_right)

        self.setLayout(layout)

        self.btn_left.clicked.connect(left_click_handler)
        self.btn_right.clicked.connect(right_click_handler)
        self.setFixedHeight(50)

    def SetText(self, t: str):
        self.text.setText(t)
