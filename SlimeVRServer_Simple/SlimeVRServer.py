import threading
import socket
from pubsub import pub
from SlimeVRServer_Simple.SlimeVRDeviceHandler import SlimeVRDeviceHandler
import utils.globals as g

lock = threading.Lock()


class SlimeVRServer(threading.Thread):

    def __init__(self):

        super().__init__()

        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.device_handlers = \
            {
                "slimeVR_controller_left": None,
                "slimeVR_controller_right": None
            }

    def run(self):
        try:
            self.server_socket.bind(('0.0.0.0', g.config["SlimeVRServer"]["server_port"]))
            print(f"Server started, listening {g.config["SlimeVRServer"]["server_port"]}")
            self.server_socket.settimeout(10)
            while self.running:
                data, addr = self.server_socket.recvfrom(1024)
                for i in self.device_handlers.values():
                    if i is not None and i.addr == addr:
                        i.queue.put(data)
            print("termination")
        except Exception as e:
            print(e)
            print("\nServer down...")
            with lock:
                for addr, handler in self.device_handlers.items():
                    handler.stop()
                    handler.join()

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
            self.device_handlers[Type].stop()
            self.device_handlers[Type] = None

    def stop(self):
        self.running = False
        for addr, handler in self.device_handlers.items():
            if handler is not None:
                handler.stop()
                handler.join()
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
                    # Create new threads and queues
                    handler = SlimeVRDeviceHandler(addr, controller_type, self.server_socket)
                    handler.start()
                    handler.queue.put(data)

                    self.device_handlers[controller_type] = handler
                    print(self.device_handlers[controller_type])
                    g.slimeVR_device_enable[controller_type] = True
                    pub.sendMessage(f"{controller_type}_find_device", t=addr[0])
                    break
            print("Exit loop thread terminates")
        except Exception as e:
            pub.sendMessage(f"{controller_type}_find_device", t="NONE")
