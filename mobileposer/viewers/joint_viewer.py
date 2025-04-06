import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class JointViewer:
    def __init__(self, data):
        self.data = data

    def view(self):
        print(f"Starting to view joints: {self.data.shape}")
        print("=" * 20)

        app = QApplication(sys.argv)
        window = MainWindow(self.data)
        window.show()
        sys.exit(app.exec_())
