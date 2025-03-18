import asyncio
import json
from flask import Flask, render_template, request
from dataclasses import dataclass
import os
import threading
from PyQt5.QtCore import QThread
import time
from werkzeug.serving import make_server
import socket
from scipy.spatial.transform import Rotation as R
import utils.globals as g
import websockets
import ssl

def update_controller_data(hand_name, hand_position, wrist_rot, finger_states):
    """更新手部的位置信息、旋转信息和手指状态"""
    if g.config["Smoothing"]["enable"]:
        index_offset = 92 if hand_name == "Left" else 98
        finger_offset = 104 if hand_name == "Left" else 109

        g.latest_data[index_offset] = hand_position[0]
        g.latest_data[index_offset + 1] = hand_position[1]
        g.latest_data[index_offset + 2] = hand_position[2]

        g.latest_data[index_offset + 3] = wrist_rot[0]
        g.latest_data[index_offset + 4] = wrist_rot[1]
        g.latest_data[index_offset + 5] = wrist_rot[2]

        for i, finger_state in enumerate(finger_states):
            g.latest_data[finger_offset + i] = finger_state
    else:
        for i, val in enumerate(wrist_rot):
            g.data[f"{hand_name}ControllerRotation"][i]["v"] = val

        for i, val in enumerate(hand_position):
            g.data[f"{hand_name}ControllerPosition"][i]["v"] = val

        for i, finger_state in enumerate(finger_states):
            g.data[f"{hand_name}ControllerFinger"][i]["v"] = finger_state

@dataclass
class ControllerState:
    w: float = 0
    x: float = 0
    y: float = 0
    z: float = 0
    joystick: tuple = (0, 0)
    joystickClicked: bool = False
    buttons: dict = None


class ControllerApp(QThread):
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))
        self.controllers = {
            "Left": ControllerState(
                buttons={"system": False, "X": False, "Y": False, "upperTrigger": False, "lowerTrigger": False}),
            "Right": ControllerState(
                buttons={"system": False, "B": False, "A": False, "upperTrigger": False, "lowerTrigger": False})
        }
        self.setup_routes()

        # WebSocket server variables
        self.websocket_clients = set()
        self.websocket_server = None
        self.websocket_thread = None
        self.websocket_loop = None  # 新增事件循环引用

        self.server_ip=self.get_server_ip()
        self.server_port=8888
        self.websocket_port=8889

        print(f"https://{self.server_ip}:{self.server_port}")

    def setup_routes(self):
        self.app.add_url_rule('/', 'home', self.home)
        self.app.add_url_rule('/left', 'left_controller', self.left_controller)
        self.app.add_url_rule('/right', 'right_controller', self.right_controller)

    # def get_server_ip(self):
    #     """获取本机在局域网中的IP地址"""
    #     server_ip=request.host.split(':')[0]  # 提取IP
    #     print(server_ip)
    #     return server_ip

    def get_server_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # doesn't even have to be reachable
            s.connect(('10.254.254.254', 1))
            server_ip = s.getsockname()[0]
        except Exception:
            server_ip = '127.0.0.1'
        finally:
            s.close()
        print(server_ip)
        return server_ip

    def home(self):
        server_ip = self.get_server_ip()
        return render_template('index.html', server_ip=server_ip, server_port=self.websocket_port, send_interval=50)

    def left_controller(self):
        server_ip = self.get_server_ip()
        return render_template('controller.html', hand='Left', server_ip=server_ip, server_port=self.websocket_port, send_interval=50)

    def right_controller(self):
        server_ip = self.get_server_ip()
        return render_template('controller.html', hand='Right', server_ip=server_ip, server_port=self.websocket_port, send_interval=50)

    async def websocket_handler(self, websocket, path):
        """处理WebSocket连接和消息"""
        self.websocket_clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                hand = data.get('hand')
                controller = self.controllers.get(hand)

                if not controller:
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid controller"}))
                    continue

                # 更新控制器数据
                quaternion = data.get('quaternion', {})
                controller.w = quaternion.get('w', 0)
                controller.x = quaternion.get('x', 0)
                controller.y = quaternion.get('y', 0)
                controller.z = quaternion.get('z', 0)
                controller.joystick = tuple(data.get('joystick', (0, 0)))
                controller.joystickClicked = data.get('joystickClicked', False)

                for btn in controller.buttons.keys():
                    if btn in data.get('buttons', {}):
                        controller.buttons[btn] = data['buttons'][btn]

                self.update_controller(hand, controller)
                # await websocket.send(json.dumps({"status": "success"}))
        finally:
            self.websocket_clients.remove(websocket)

    def update_controller(self, hand, controller):
        """更新控制器状态到全局数据"""
        if hand == "Left" and g.config["Tracking"]["LeftController"]["enable"]:
            rotation = R.from_quat([controller.x, controller.y, controller.z, controller.w])
            conversion_rot = R.from_euler('y', -90, degrees=True)
            rotation = rotation * conversion_rot
            euler_angles = rotation.as_euler('xyz', degrees=True)
            wrist_rot = [-euler_angles[0], euler_angles[1], -euler_angles[2]]
            update_controller_data(hand, [0.0, 0.0, 0.0], wrist_rot, [1, 1, 1, 1, 1])
        if hand == "Right" and g.config["Tracking"]["RightController"]["enable"]:
            rotation = R.from_quat([controller.x, controller.y, controller.z, controller.w])
            conversion_rot = R.from_euler('y', 90, degrees=True)
            rotation = rotation * conversion_rot
            euler_angles = rotation.as_euler('xyz', degrees=True)
            wrist_rot = [-euler_angles[0], euler_angles[1], -euler_angles[2]]
            update_controller_data(hand, [0.0, 0.0, 0.0], wrist_rot, [1, 1, 1, 1, 1])

    async def start_websocket_server(self):
        """启动WebSocket服务器"""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile="./templates/ssl/cert.pem", keyfile="./templates/ssl/key.pem")  # 替换为你的证书和密钥文件路径
        self.websocket_server = await websockets.serve(
            self.websocket_handler, "0.0.0.0", 8889,ssl=ssl_context
        )
        return self.websocket_server

    def run_websocket_server(self):
        self.websocket_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.websocket_loop)
        self.websocket_server = self.websocket_loop.run_until_complete(
            self.start_websocket_server()
        )
        self.websocket_loop.run_forever()

    def run(self):
        ssl_context = ("./templates/ssl/cert.pem", "./templates/ssl/key.pem")
        self.server = make_server('0.0.0.0', self.server_port, self.app, ssl_context=ssl_context)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        # 启动WebSocket服务器
        self.websocket_thread = threading.Thread(target=self.run_websocket_server)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()

    def stop(self):
        self.requestInterruption()
        if hasattr(self, 'server'):
            self.server.shutdown()
        if hasattr(self, 'server_thread'):
            self.server_thread.join()
        if self.websocket_server and self.websocket_loop:
            async def shutdown():
                for client in self.websocket_clients:
                    await client.close()
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.websocket_loop.stop()
            asyncio.run_coroutine_threadsafe(shutdown(), self.websocket_loop)
        if self.websocket_thread:
            self.websocket_thread.join()

if __name__ == '__main__':
    controller_app = ControllerApp()
    controller_app.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller_app.stop()
        controller_app.wait()
