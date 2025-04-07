import binascii
import threading
import time
import utils.globals as g

lock = threading.Lock()
controller_code={"Left":0,"Right":1}
class SlimeVRDeviceHandler():
    def __init__(self, addr, controller_type: str, socket_server):

        super().__init__()
        self.heart_beat = None
        self.addr = addr
        self.acc=None
        self.raw=None
        self.running=True

        self.controller_type = controller_type
        self.server = socket_server

        self.heartbeat_thread = threading.Thread(target=self.HeartBeatThread)
        self.heartbeat_thread.start()

        g.combo.append(controller_code[self.controller_type])
        self.accelerometer_processor = None
    def __del__(self):
        self.running=False
        g.combo.remove(controller_code[self.controller_type])

    def put_data(self,id,data):
        if(id==4):
            self.acc=data
        elif(id==17):
            self.raw=data
        else:
            pass

    def check_is_full(self):
        if(self.acc is not None and self.raw is not None):
            return True

    def HeartBeatThread(self):
        while self.running:
            with lock:
                data_prefix = bytes.fromhex("0000000a0000000000000000")
                crc = binascii.crc32(data_prefix) & 0xFFFFFFFF
                response = data_prefix + crc.to_bytes(4, byteorder="little")
                self.server.sendto(response, self.addr)
                time.sleep(1)
        print("Exit a heartbeat thread")
