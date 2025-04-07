import threading
import socket
from cgitb import handler

from pubsub import pub
from SlimeVRServer_Simple.SlimeVRDeviceHandler import SlimeVRDeviceHandler
import utils.globals as g
from SlimeVRServer_Simple.UDP.UDPProtocolParser import parse_packet
from SlimeVRServer_Simple.DataBaseController import DataBaseController

lock = threading.Lock()


class SlimeVRServer(threading.Thread):

    def __init__(self):

        super().__init__()

        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.device_handlers = \
            {
                "Left": None,
                "Right": None
            }

        self.data_base_controller = DataBaseController(self.device_handlers)
        self.data_base_controller.start()
    def run(self):
        try:
            self.server_socket.bind(('0.0.0.0', g.config["SlimeVRServer"]["server_port"]))
            # print(f"Server started, listening {g.config["SlimeVRServer"]["server_port"]}")
            self.server_socket.settimeout(None)
            while self.running:
                data, addr = self.server_socket.recvfrom(1024)
                package_id,parse_data= parse_packet(data, self.server_socket, addr)
                if parse_data is None:
                    continue
                for i in self.device_handlers.values():
                    if i is not None and i.addr == addr:
                        i.put_data(package_id,parse_data)
            # print("termination")
        except Exception as e:
            print(e)
            print("\nServer down...")

    #Connection thread
    def link_slimeVR_device(self, controller_type: str):
        if self.device_handlers[controller_type]:
            return
        data_thread = threading.Thread(
            target=self.create_new_device_link, args=(controller_type,), daemon=True
        )
        data_thread.start()

    def remove_device(self, Type: str):
        if self.device_handlers[Type] is not None:
            del self.device_handlers[Type]
            self.device_handlers[Type] = None

    def stop(self):
        self.running = False
        self.data_base_controller.stopped.set()
        if self.server_socket:
            self.server_socket.close()

    def create_new_device_link(self, controller_type: str):
        try:
            while self.running:
                data, addr = self.server_socket.recvfrom(1024)

                with lock:
                    flag = False
                    for i in self.device_handlers.values():
                        if i is not None and i.addr == addr:
                            flag = True
                            break
                    if flag:
                        continue

                    parse_packet(data,self.server_socket,addr)
                    handler = SlimeVRDeviceHandler(addr, controller_type, self.server_socket)

                    self.device_handlers[controller_type] = handler
                    pub.sendMessage(f"{controller_type}_find_device", t=addr[0])
                    break
            print("Exit loop thread terminates")
        except Exception as e:
            pub.sendMessage(f"{controller_type}_find_device", t="NONE")

if __name__ == "__main__":

    slime=SlimeVRServer()
    slime.start()
    slime.link_slimeVR_device("Left")