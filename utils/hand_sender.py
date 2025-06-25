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
    follow: bool
    enable: bool
    force_enable: bool
    change_flag: bool

# GloveControllerSender equivalent in Python
class GloveControllerSender:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 39570):
        # Initialize OSC client
        self.client = udp_client.SimpleUDPClient(osc_ip, osc_port)

        self.left_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0),False,False,False,True)
        self.right_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0),False,False,False,True)
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

        if target.follow:
            self.client.send_message("/VMT/Follow/Driver", message)
        else:
            self.client.send_message("/VMT/Joint/Driver", message)

    def disable_hand(self,is_left_hand):
        message = [1 if is_left_hand else 2, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.client.send_message("/VMT/Room/Unity", message)

    def send_finger(self, is_left_hand, target: Transform):
        for i, value in enumerate(target.finger):
            index = i + 1
            message_0 = [
                1 if is_left_hand else 2,  # lefthand ? 1 : 2
                int(index),
                float(value),
                0,
                0,
            ]
            self.client.send_message("/VMT/Skeleton/Scalar", message_0)
        message_1 = [1 if is_left_hand else 2, 0.0]  # lefthand ? 1 : 2
        self.client.send_message("/VMT/Skeleton/Apply", message_1)

    def send_button(self,is_left_hand, index,status):
        message = [1 if is_left_hand else 2, int(index), 0.0, int(status)]
        self.client.send_message("/VMT/Input/Button", message)

    def send_trigger(self, is_left_hand, index, status=0.0):
        message = [1 if is_left_hand else 2, int(index), 0.0, float(status)]
        self.client.send_message("/VMT/Input/Trigger", message)

    def send_joystick(self, is_left_hand, index, status_0=0.0, status_1=0.0):
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            int(index),
            0.0,
            float(status_0),
            float(status_1),
        ]
        self.client.send_message("/VMT/Input/Joystick", message)

    def send_joystick_click(self,is_left_hand, index, status):
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            int(index),
            0.0,
            int(status)
        ]
        self.client.send_message("/VMT/Input/Joystick/Click", message)

    def vmt_init(self):
        self.client.send_message(
            "/VMT/SetRoomMatrix",
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0],
        )

    def update(self):
        if not self.left_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"] and not self.left_hand.force_enable:
            g.controller.send_trigger(True, 0, 0)
            self.disable_hand(True)
        else:
            self.send_hand(True, self.left_hand)
            self.send_finger(True, self.left_hand)

        if not self.right_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"] and not self.right_hand.force_enable:
            g.controller.send_trigger(False, 0, 0)
            self.disable_hand(False)
        else:
            self.send_hand(False, self.right_hand)
            self.send_finger(False, self.right_hand)
