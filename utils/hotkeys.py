import keyboard
from pynput import mouse
from utils.actions import *
from utils.json_manager import load_json

mouse_listener = None

def toggle_hotkeys():
    g.config["Hotkey"]["enable"] = not g.config["Hotkey"]["enable"]
    print("Hotkey:",g.config["Hotkey"]["enable"])
    if g.config["Hotkey"]["enable"]:
        apply_hotkeys()
    else:
        stop_hotkeys()

actions = {
    "toggle_hotkeys": toggle_hotkeys,
    "reset_eye": reset_eye,
    "disable_eye_yaw": disable_eye_yaw,
    "disable_eye": disable_eye,
    "reset_head": reset_head,
    "up": up,
    "down": down,
    "left": left,
    "right": right,
    "head_pitch_up": lambda: head_pitch(True),
    "head_pitch_down": lambda: head_pitch(False),
    "grab_left": lambda: grab(True, 2),
    "grab_right": lambda: grab(False, 2),
    "trigger_left": (
        lambda _: trigger_press(True, 0),
        lambda _: trigger_release(True, 0),
    ),
    "trigger_right": (
        lambda _: trigger_press(False, 0),
        lambda _: trigger_release(False, 0),
    ),
    "joystick_up_right": lambda: joystick_up(False, 1),
    "joystick_down_right": lambda: joystick_down(False, 1),
    "joystick_middle_right": lambda: joystick_middle(False, 1),
    "joystick_middle_right_delay": lambda: joystick_middle_delay(False, 1),
    "enable_fingers_left": lambda: enable_fingers(True),
    "enable_fingers_right": lambda: enable_fingers(False),
    "set_finger_0_left": lambda: set_finger(True, 0),
    "set_finger_1_left": lambda: set_finger(True, 1),
    "set_finger_2_left": lambda: set_finger(True, 2),
    "set_finger_3_left": lambda: set_finger(True, 3),
    "set_finger_4_left": lambda: set_finger(True, 4),
    "set_finger_0_right": lambda: set_finger(False, 0),
    "set_finger_1_right": lambda: set_finger(False, 1),
    "set_finger_2_right": lambda: set_finger(False, 2),
    "set_finger_3_right": lambda: set_finger(False, 3),
    "set_finger_4_right": lambda: set_finger(False, 4),
    "set_hand_track": lambda: set_hand_track(),
    "enable_hand": enable_hand,
    "reset_hand_left": lambda: reset_hand(True),
    "reset_hand_right": lambda: reset_hand(False),
    "enable_tongue": enable_tongue,
    "set_tongue": set_tongue
}


def setup_hotkeys():
    hotkey_config = load_json("./hotkeys.json")
    return hotkey_config

def set_hand_track():  # toggle modde
    g.config["Tracking"]["Hand"]["hand_link_head"] = not g.config["Tracking"]["Hand"]["hand_link_head"]

def apply_hotkeys():
    global mouse_listener
    keyboard.unhook_all()
    if mouse_listener is not None:
        mouse_listener.stop()

    mouse_actions = {}
    for item in g.hotkey_config.get("Hotkeys"):
        key = item.get("key")
        mouse_button = item.get("mouse")
        action = item.get("action")
        if action and key:
            if action in actions:
                if isinstance(actions[action], tuple):
                    if len(actions[action]) == 2:
                        keyboard.on_press_key(key, actions[action][0])
                        keyboard.on_release_key(key, actions[action][1])
                else:
                    keyboard.add_hotkey(key, actions[action])
            elif "left_fingers" in action or "right_fingers" in action:
                keyboard.add_hotkey(key, lambda a=action: set_fingers(a))
        if mouse_button and action:
            if action in actions:
                if mouse_button not in mouse_actions:
                    mouse_actions[mouse_button] = []
                mouse_actions[mouse_button].append(actions[action])
    pressed_buttons = set()

    def on_click(x, y, button, pressed):
        button_str = None
        if pressed:
            pressed_buttons.add(button)
        if (
            mouse.Button.left in pressed_buttons
            and mouse.Button.middle in pressed_buttons
        ):
            button_str = "left+middle"
        elif (
            mouse.Button.right in pressed_buttons
            and mouse.Button.middle in pressed_buttons
        ):
            button_str = "right+middle"
        elif button == mouse.Button.left:
            button_str = "left"
        elif button == mouse.Button.right:
            button_str = "right"
        elif button == mouse.Button.middle:
            button_str = "middle"
        print(button_str,pressed)
        if not pressed:
            pressed_buttons.discard(button)
        if button_str in mouse_actions:
            action_list = mouse_actions[button_str]
            if action_list:
                for action in action_list:
                    if isinstance(action, tuple):
                        if pressed:
                            action[0](None)  # 执行按下操作
                        else:
                            action[1](None)  # 执行释放操作
                    else:
                        if pressed:
                            action()

    def on_scroll(x, y, dx, dy):
        if dy > 0:
            action_list = mouse_actions.get("scroll_up")
            if action_list:
                for action in action_list:
                    if isinstance(action, tuple):
                        print("wrong action")
                    elif action:
                        action()
        elif dy < 0:
            action_list = mouse_actions.get("scroll_down")
            if action_list:
                for action in action_list:
                    if isinstance(action, tuple):
                        print("wrong action")
                    elif action:
                        action()

    if mouse_actions:
        mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        mouse_listener.start()
    print("Start Hotkey")

def stop_hotkeys():
    global mouse_listener
    if mouse_listener is not None:
        mouse_listener.stop()
        mouse_listener = None
    keyboard.unhook_all()
    for item in g.hotkey_config["Hotkeys"]:
        if item["action"] == "toggle_hotkeys":
            keyboard.add_hotkey(item["key"], toggle_hotkeys)
    print("Stop Hotkey")