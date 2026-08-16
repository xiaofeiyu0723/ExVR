import socket
import struct
from dataclasses import dataclass
import utils.globals as g


def _osc_pad(b: bytes) -> bytes:
    """OSC 字符串补零，与 pythonosc write_string 逐字节一致：
    diff = 4 - (len % 4)，len 恰为 4 的倍数时也补 4 个 0（pythonosc 原样行为）。"""
    return b + b"\0" * (4 - (len(b) % 4))


def _encode_osc(address: str, args) -> bytes:
    """轻量 OSC 编码：与 pythonosc 的 i/f/s 编码逐字节一致，但省掉逐条构造开销。
    实测 pythonosc send_message ~20us/条，本实现 ~5us/条；send 线程每轮 20+ 条 OSC。"""
    if not isinstance(args, (list, tuple)):
        args = [args]  # pythonosc 兼容：标量包装为单元素列表，避免标量 TypeError
    out = bytearray(_osc_pad(address.encode("utf-8")))
    typetag = bytearray(b",")
    payload = bytearray()
    for arg in args:
        if isinstance(arg, int):
            typetag += b"i"
            payload += struct.pack(">i", arg)
        elif isinstance(arg, float):
            typetag += b"f"
            payload += struct.pack(">f", arg)
        elif isinstance(arg, str):
            typetag += b"s"
            payload += _osc_pad(arg.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported OSC arg type: {type(arg)}")
    out += _osc_pad(bytes(typetag))
    out += payload
    return bytes(out)


class _OscSocket:
    """SimpleUDPClient 的替代：相同 send_message 接口，内部走原生 UDP + struct 编码"""

    def __init__(self, ip: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)

    def send_message(self, address: str, value):
        self._sock.sendto(_encode_osc(address, value), self._addr)

    def close(self):
        self._sock.close()


def setup_controller():
    controller = GloveControllerSender(osc_ip=g.config["Sending"]["address"], osc_port=39570)
    return controller

# Define a simple Transform class to hold position and rotation data
@dataclass
class Transform:
    position: tuple  # (x, y, z)
    rotation: tuple  # (x, y, z, w)
    finger: tuple  # (0,1,2,3,4)
    splay: tuple  # (0,1,2,3,4), normalized -1..1
    follow: bool
    enable: bool
    force_enable: bool
    change_flag: bool
    device_index: int
    enable_type: int

# GloveControllerSender equivalent in Python
class GloveControllerSender:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 39570):
        # Initialize OSC client
        self.client = _OscSocket(osc_ip, osc_port)

        self.left_hand = self.create_transform(1, 5)
        self.right_hand = self.create_transform(2, 6)
        self.left_controller = self.create_transform(3, 5)
        self.right_controller = self.create_transform(4, 6)
        self.last_left_hand = self.create_transform(1, 5)
        self.last_right_hand = self.create_transform(2, 6)
        self.partial_handoff_frames = {True: 0, False: 0}
        self.partial_handoff_hold_frames = 4
        self.vmt_init()

    def create_transform(self, device_index: int, enable_type: int):
        return Transform(
            (0, 0, 0),
            (0, 0, 0, 1),
            (1.0, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            False,
            False,
            False,
            False,
            device_index,
            enable_type,
        )

    def get_target(self, is_left_hand, use_controller=False):
        if use_controller:
            return self.left_controller if is_left_hand else self.right_controller
        return self.left_hand if is_left_hand else self.right_hand

    def get_last_hand(self, is_left_hand):
        return self.last_left_hand if is_left_hand else self.last_right_hand

    def copy_transform_data(self, target: Transform, source: Transform):
        target.position = source.position
        target.rotation = source.rotation
        target.finger = source.finger
        target.splay = source.splay

    def send_hand(self, target: Transform):
        message = [
            target.device_index,
            target.enable_type,
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

        # if target.follow:
        #     self.client.send_message("/VMT/Follow/Driver", message)
        # else:
        #     self.client.send_message("/VMT/Joint/Driver", message)
        self.client.send_message("/VMT/Joint/Driver", message)

    def disable_hand(self, target: Transform):
        message = [target.device_index, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.client.send_message("/VMT/Raw/Driver", message)

    def send_finger(self, target: Transform):
        for i, value in enumerate(target.finger):
            index = i + 1
            message_0 = [
                target.device_index,
                int(index),
                float(value),
                0,
                0,
            ]
            self.client.send_message("/VMT/Skeleton/Scalar", message_0)
        for i, value in enumerate(target.splay):
            index = i + 1
            self.client.send_message(
                "/VMT/Skeleton/Splay",
                [target.device_index, int(index), float(value)],
            )
        message_1 = [target.device_index, 0.0]
        self.client.send_message("/VMT/Skeleton/Apply", message_1)

    def send_button(self, is_left_hand, index, status, use_controller=False):
        target = self.get_target(is_left_hand, use_controller)
        message = [target.device_index, int(index), 0.0, int(status)]
        self.client.send_message("/VMT/Input/Button", message)

    def send_trigger(self, is_left_hand, index, status=0.0, use_controller=False):
        target = self.get_target(is_left_hand, use_controller)
        message = [target.device_index, int(index), 0.0, float(status)]
        self.client.send_message("/VMT/Input/Trigger", message)

    def send_joystick(self, is_left_hand, index, status_0=0.0, status_1=0.0, use_controller=False):
        target = self.get_target(is_left_hand, use_controller)
        message = [
            target.device_index,
            int(index),
            0.0,
            float(status_0),
            float(status_1),
        ]
        self.client.send_message("/VMT/Input/Joystick", message)

    def send_joystick_click(self, is_left_hand, index, status, use_controller=False):
        target = self.get_target(is_left_hand, use_controller)
        message = [
            target.device_index,
            int(index),
            0.0,
            int(status)
        ]
        self.client.send_message("/VMT/Input/Joystick/Click", message)

    def release_controller_inputs(self, is_left_hand):
        self.send_trigger(is_left_hand, 0, 0, use_controller=True)
        self.send_trigger(is_left_hand, 2, 0, use_controller=True)
        self.send_button(is_left_hand, 0, 0, use_controller=True)
        self.send_button(is_left_hand, 1, 0, use_controller=True)
        self.send_button(is_left_hand, 3, 0, use_controller=True)
        self.send_joystick(is_left_hand, 1, 0.0, 0.0, use_controller=True)
        self.send_joystick_click(is_left_hand, 1, 0, use_controller=True)

    def partial_handoff_owns_controller(self, is_left_hand):
        target = self.get_target(is_left_hand, use_controller=True)
        return self.partial_handoff_frames[is_left_hand] > 0 and not target.enable and not target.force_enable

    def cancel_partial_handoff(self, is_left_hand):
        if self.partial_handoff_owns_controller(is_left_hand):
            self.disable_hand(self.get_target(is_left_hand, use_controller=True))
        self.partial_handoff_frames[is_left_hand] = 0

    def handoff_full_hand_to_partial_disconnect(self, is_left_hand, source: Transform):
        partial_target = self.get_target(is_left_hand, use_controller=True)
        if partial_target.enable or partial_target.force_enable:
            return

        self.copy_transform_data(partial_target, source)
        self.send_hand(partial_target)
        self.send_finger(partial_target)
        self.partial_handoff_frames[is_left_hand] = self.partial_handoff_hold_frames

    def update_partial_handoff(self, is_left_hand):
        if not self.partial_handoff_owns_controller(is_left_hand):
            self.partial_handoff_frames[is_left_hand] = 0
            return False

        partial_target = self.get_target(is_left_hand, use_controller=True)
        self.send_hand(partial_target)
        self.send_finger(partial_target)
        self.partial_handoff_frames[is_left_hand] -= 1
        if self.partial_handoff_frames[is_left_hand] <= 0:
            self.disable_hand(partial_target)
        return True

    def vmt_init(self):
        self.client.send_message(
            "/VMT/SetRoomMatrix",
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0],
        )

    def update_target(self, is_left_hand, target: Transform, use_controller=False, disable_when_down=True):
        if not target.enable and disable_when_down and not target.force_enable:
            if use_controller:
                self.release_controller_inputs(is_left_hand)
                self.disable_hand(target)
            else:
                self.send_trigger(is_left_hand, 0, 0)
                self.send_trigger(is_left_hand, 2, 0)
                self.disable_hand(target)
                if target.change_flag:
                    self.handoff_full_hand_to_partial_disconnect(is_left_hand, self.get_last_hand(is_left_hand))
                    target.change_flag = False
        else:
            if not use_controller:
                self.cancel_partial_handoff(is_left_hand)
                self.copy_transform_data(self.get_last_hand(is_left_hand), target)
                target.change_flag = True
            self.send_hand(target)
            self.send_finger(target)

    def update(self):
        left_hand_disable = g.config["Tracking"]["Hand"]["enable_hand_down"] or self.left_controller.enable
        right_hand_disable = g.config["Tracking"]["Hand"]["enable_hand_down"] or self.right_controller.enable
        self.update_target(True, self.left_hand, disable_when_down=left_hand_disable)
        self.update_target(False, self.right_hand, disable_when_down=right_hand_disable)
        if not self.update_partial_handoff(True):
            self.update_target(True, self.left_controller, use_controller=True)
        if not self.update_partial_handoff(False):
            self.update_target(False, self.right_controller, use_controller=True)
