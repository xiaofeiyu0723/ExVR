from dataclasses import dataclass
from pythonosc import udp_client
import utils.globals as g

def setup_controller():
    controller = GloveControllerSender(osc_ip=g.config["Sending"]["address"], osc_port=39570)
    return controller

# Define a simple Transform class to hold position and rotation data
@dataclass
class Transform:
    position: tuple  # (x, y, z)
    rotation: tuple  # (x, y, z, w)
    finger: tuple  # (0,1,2,3,4)


# GloveControllerSender equivalent in Python
class GloveControllerSender:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 8000):
        # Initialize OSC client
        self.client = udp_client.SimpleUDPClient(osc_ip, osc_port)

        self.left_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0))
        self.right_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0))
        self.vmt_init()

    def send_hand(self, is_left_hand, target: Transform):
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            5 if is_left_hand else 6,  # enable
            0.0,  # timeoffset
            target.position[0],
            target.position[1],
            target.position[2],  # -0.25,
            target.rotation[0],
            target.rotation[1],
            target.rotation[2],
            target.rotation[3],
            "HMD",  # serial
        ]
        self.client.send_message("/VMT/Follow/Driver", message)

    def send_finger(self, is_left_hand, target: Transform):
        for i, value in enumerate(target.finger):
            index = i + 1
            message_0 = [
                1 if is_left_hand else 2,  # lefthand ? 1 : 2
                int(index),
                value,
                0,
                0,
            ]
            self.client.send_message("/VMT/Skeleton/Scalar", message_0)
        message_1 = [1 if is_left_hand else 2, 0.0]  # lefthand ? 1 : 2
        self.client.send_message("/VMT/Skeleton/Apply", message_1)
        # if sum(target.finger)==0:
        # self.send_trigger(is_left_hand,1.0)
        # else:
        #     self.send_trigger(is_left_hand,0.0)

    def send_trigger(self, is_left_hand, index, status=0.0):
        message = [1 if is_left_hand else 2, index, 0.0, status]  # lefthand ? 1 : 2
        self.client.send_message("/VMT/Input/Trigger", message)

    def send_joystick(self, is_left_hand, index, status_0=0.0, status_1=0.0):
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            index,
            0.0,
            status_0,
            status_1,
        ]
        self.client.send_message("/VMT/Input/Joystick", message)

    def vmt_init(self):
        # 设置房间矩阵
        self.client.send_message(
            "/VMT/SetRoomMatrix",
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0],
        )

    def update(self):
        self.send_hand(True, self.left_hand)
        self.send_hand(False, self.right_hand)
        self.send_finger(True, self.left_hand)
        self.send_finger(False, self.right_hand)