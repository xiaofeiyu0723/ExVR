import keyboard
from pynput import mouse
from utils.actions import *
from utils.json_manager import load_json
import screeninfo
import os
import psutil
import win32gui, win32process

mouse_listener = None
monitor = None
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
    "squat": squat,
    "prone": prone,
    "head_pitch_up": lambda: head_pitch(True),
    "head_pitch_down": lambda: head_pitch(False),
    "head_yaw_left": lambda: head_yaw(True),
    "head_yaw_right": lambda: head_yaw(False),
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
    "toggle_hand_tracking_mode": lambda: toggle_hand_tracking_mode(),
    "enable_hand": enable_hand,
    "reset_hand_left": lambda: reset_hand(True),
    "reset_hand_right": lambda: reset_hand(False),
    "enable_tongue": enable_tongue,
    "set_tongue": set_tongue
}


def setup_hotkeys():
    hotkey_config = load_json("./settings/hotkeys.json")
    return hotkey_config

def apply_hotkeys():
    global mouse_listener
    keyboard.unhook_all()
    if mouse_listener is not None:
        mouse_listener.stop()

    # find better way to do please
    def hook(func):
        def wrapper(*args, **kwargs):
            if g.config['Setting']["only_ingame"] and not is_in_game():
                return# print("not in game")
            return func(*args, **kwargs)
        return wrapper

    mouse_actions = {}
    for item in g.hotkey_config.get("Hotkeys"):
        key = item.get("key")
        mouse_button = item.get("mouse")
        action = item.get("action")
        if action and key:
            if action in actions:
                if isinstance(actions[action], tuple):
                    if len(actions[action]) == 2:
                        keyboard.on_press_key(key, hook(actions[action][0]))
                        keyboard.on_release_key(key, hook(actions[action][1]))
                else:
                    keyboard.add_hotkey(key, hook(actions[action]))
            elif "left_fingers" in action or "right_fingers" in action:
                keyboard.add_hotkey(key, lambda a=action: set_fingers(a))
        if mouse_button and action:
            if action in actions:
                if mouse_button not in mouse_actions:
                    mouse_actions[mouse_button] = []
                if isinstance(actions[action], tuple):
                    # press/release button
                    h = actions[action]
                    mouse_actions[mouse_button].append((hook(h[0]), hook(h[1])))
                else:
                    mouse_actions[mouse_button].append(hook(actions[action]))
    pressed_buttons = set()
    def on_click(x, y, button, pressed):
        if g.config['Setting']["only_ingame"] and not is_in_game():
            return

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
        # print(button_str,pressed)
        if not pressed:
            pressed_buttons.discard(button)
        if button_str in mouse_actions:
            action_list = mouse_actions[button_str]
            if action_list:
                for action in action_list:
                    if isinstance(action, tuple):
                        if pressed:
                            action[0](None)
                        else:
                            action[1](None)
                    else:
                        if pressed:
                            action()

    def on_scroll(x, y, dx, dy):
        if g.config['Setting']["only_ingame"] and not is_in_game():
            return

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

    def get_current_monitor(x,y):
        monitors = screeninfo.get_monitors()
        for m in monitors:
            if m.x <= x <= m.x + m.width and m.y <= y <= m.y + m.height:
                return m
        return None

    def on_move(x, y):
        global monitor

        if g.config['Mouse']["enable"]:
            if g.config['Setting']["only_ingame"] and not is_in_game():
                # this ensure that it will not keep rotating
                bound = g.config["Mouse"]["bound_threshold"] - 0.01 # i have no idea why it need -0.01
                g.latest_data[117] = max(-bound, min(bound, g.latest_data[117]))
                g.latest_data[118] = max(-bound, min(bound, g.latest_data[118]))
                g.data["MousePosition"][0]["v"] = max(-bound, min(bound, 
                                                    g.data["MousePosition"][0]["v"]))
                g.data["MousePosition"][1]["v"] = max(-bound, min(bound, 
                                                    g.data["MousePosition"][1]["v"]))
                return
            if monitor is None:
                monitor = get_current_monitor(x, y)
            x_normalized = (x / monitor.width - 0.5)
            y_normalized = -(y / monitor.height - 0.5)
            if g.config["Smoothing"]["enable"]:
                g.latest_data[117] = x_normalized
                g.latest_data[118] = y_normalized
            else:
                g.data["MousePosition"][0]["v"] = x_normalized
                g.data["MousePosition"][1]["v"] = y_normalized


    if mouse_actions:
        mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll, on_move=on_move)
        mouse_listener.start()
    print("Start Hotkey")

def stop_hotkeys():
    global mouse_listener,monitor
    if mouse_listener is not None:
        mouse_listener.stop()
        mouse_listener = None
        monitor = None
    keyboard.unhook_all()
    for item in g.hotkey_config["Hotkeys"]:
        if item["action"] == "toggle_hotkeys":
            keyboard.add_hotkey(item["key"], toggle_hotkeys)
    print("Stop Hotkey")

# check title first then program name
def is_in_game():
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd)

    if title == g.config['Setting']["only_ingame_game"]:
        return True

    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    try:
        program_name = os.path.basename(psutil.Process(pid).exe())
        if program_name == g.config['Setting']["only_ingame_game"]:
            return True
    except:
        pass

    return False