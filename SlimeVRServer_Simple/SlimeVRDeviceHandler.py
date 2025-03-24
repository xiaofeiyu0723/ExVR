import binascii
import threading
import time
from queue import Queue


import utils.globals as g
from SlimeVRServer_Simple.UDP.UDPProtocolParser import parse_packet

lock = threading.Lock()
from SlimeVRServer_Simple.filter import AccelerometerProcessor


class SlimeVRDeviceHandler(threading.Thread):
    def __init__(self, addr, controller_type: str, socket_server):

        super().__init__()
        self.heart_beat = None
        self.addr = addr
        self.queue = Queue(maxsize=100)
        self.running = True
        self.controller_type = controller_type
        self.server = socket_server
        g.slimeVR_device_enable[controller_type] = True
        self.accelerometer_processor = None

    def run(self):
        print(f"[Device {self.controller_type}] Processing thread start")
        self.heart_beat = threading.Thread(
            target=self.HeartBeatThread, daemon=True
        )
        self.heart_beat.start()
        if (self.controller_type == "slimeVR_controller_left"):
            self.accelerometer_processor = AccelerometerProcessor("LeftSlimeVRController")
        else:
            self.accelerometer_processor = AccelerometerProcessor("RightSlimeVRController")
        while self.running:
            try:
                data = self.queue.get(timeout=5)
                if data is None:
                    break
                parse_packet(data, self.server, self.addr, self.accelerometer_processor)
            except Exception as e:
                if "empty" not in str(e):
                    print(f"[Device {self.addr}] Error: {e}")
        print(f"[Device {self.addr}] Handling thread exit")

    def stop(self):
        g.slimeVR_device_enable[self.controller_type] = False
        self.running = False

    def HeartBeatThread(self):
        while self.running:
            with lock:
                data_prefix = bytes.fromhex("0000000a0000000000000000")
                crc = binascii.crc32(data_prefix) & 0xFFFFFFFF
                response = data_prefix + crc.to_bytes(4, byteorder="little")
                self.server.sendto(response, self.addr)
                time.sleep(1)
        print("Exit a heartbeat thread")
