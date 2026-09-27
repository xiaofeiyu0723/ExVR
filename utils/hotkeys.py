from pynput import mouse, keyboard as pynput_keyboard
from utils.actions import *
from utils.json_manager import load_json
import screeninfo
import os
import sys
import psutil
if sys.platform == "win32":
    import keyboard
    import win32gui, win32process

mouse_listener = None
monitor = None

# Linux keyboard hotkey engine (pynput-based). The `keyboard` library cannot
# be used here without root: it hard-codes an euid==0 gate and requires
# console access (dumpkeys). pynput listens through Xwayland, which covers
# SteamOS games (they run via Proton/Xwayland) and ExVR itself.
_kb_listener = None
_kb_chords = []  # [(key-name tuple, callback or (press, release) pair)]
_kb_held = []    # currently held key names in press order
_kb_active_tuples = []  # chords whose press callback fired, awaiting release

_PYNPUT_SPECIAL = {
    pynput_keyboard.Key.up: "up",
    pynput_keyboard.Key.down: "down",
    pynput_keyboard.Key.left: "left",
    pynput_keyboard.Key.right: "right",
    pynput_keyboard.Key.ctrl: "ctrl",
    pynput_keyboard.Key.ctrl_l: "ctrl",
    pynput_keyboard.Key.ctrl_r: "ctrl",
    pynput_keyboard.Key.alt: "alt",
    pynput_keyboard.Key.alt_l: "alt",
    pynput_keyboard.Key.alt_r: "alt",
    pynput_keyboard.Key.alt_gr: "altgr",
    pynput_keyboard.Key.shift: "shift",
    pynput_keyboard.Key.shift_l: "shift",
    pynput_keyboard.Key.shift_r: "shift",
    pynput_keyboard.Key.space: "space",
    pynput_keyboard.Key.enter: "enter",
    pynput_keyboard.Key.tab: "tab",
    pynput_keyboard.Key.esc: "esc",
    pynput_keyboard.Key.backspace: "backspace",
    pynput_keyboard.Key.delete: "delete",
    pynput_keyboard.Key.insert: "insert",
    pynput_keyboard.Key.home: "home",
    pynput_keyboard.Key.end: "end",
    pynput_keyboard.Key.page_up: "pageup",
    pynput_keyboard.Key.page_down: "pagedown",
    pynput_keyboard.Key.caps_lock: "capslock",
    pynput_keyboard.Key.num_lock: "numlock",
    pynput_keyboard.Key.cmd: "windows",
}
for _i in range(1, 13):
    _PYNPUT_SPECIAL[getattr(pynput_keyboard.Key, "f%d" % _i)] = "f%d" % _i

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
    "head_yaw_left": lambda: head_yaw(False),
    "head_yaw_right": lambda: head_yaw(True),
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
    hotkey_config = load_json("settings/hotkeys.json")
    return hotkey_config

def stop_mouse_listener():
    global mouse_listener
    if mouse_listener is not None:
        try:
            mouse_listener.stop()
        except Exception as exc:
            # pynput can raise while tearing down its X record context; the
            # listener thread is a daemon either way.
            print(f"mouse listener stop warning: {exc}")
        mouse_listener = None

def _pynput_key_name(key):
    name = _PYNPUT_SPECIAL.get(key)
    if name is not None:
        return name
    if isinstance(key, pynput_keyboard.KeyCode) and key.char:
        return key.char.lower()
    return None

def _parse_chord(spec):
    parts = tuple(part.strip().lower() for part in str(spec).split("+"))
    if not all(parts):
        return None
    return parts

def _register_chord(key_spec, cb):
    if sys.platform == "win32":
        if isinstance(cb, tuple):
            keyboard.on_press_key(key_spec, cb[0])
            keyboard.on_release_key(key_spec, cb[1])
        else:
            keyboard.add_hotkey(key_spec, cb)
        return
    chord = _parse_chord(key_spec)
    if chord:
        _kb_chords.append((chord, cb))

def _on_kb_press(key):
    name = _pynput_key_name(key)
    if name is None:
        return
    if name not in _kb_held:
        _kb_held.append(name)
    # fire every chord whose key sequence matches the held-stack suffix; the
    # keyboard library behaves the same way, which the config relies on
    # (e.g. "[" is both the trigger key and the prefix of "[+f1")
    for chord, cb in _kb_chords:
        if len(chord) <= len(_kb_held) and tuple(_kb_held[-len(chord):]) == chord:
            try:
                if isinstance(cb, tuple):
                    cb[0](None)
                    if chord not in _kb_active_tuples:
                        _kb_active_tuples.append(chord)
                else:
                    cb()
            except Exception as exc:
                print(f"hotkey action error: {exc}")

def _on_kb_release(key):
    name = _pynput_key_name(key)
    if name is None:
        return
    while name in _kb_held:
        _kb_held.remove(name)
    # fire the release callback of any (press, release) chord that is no
    # longer fully held, so multi-key bindings can't leave input stuck down
    for chord in list(_kb_active_tuples):
        if all(part in _kb_held for part in chord):
            continue
        _kb_active_tuples.remove(chord)
        for registered, cb in _kb_chords:
            if registered == chord and isinstance(cb, tuple):
                try:
                    cb[1](None)
                except Exception as exc:
                    print(f"hotkey action error: {exc}")
                break

def start_keyboard_listener():
    global _kb_listener
    if _kb_listener is not None:
        return
    try:
        _kb_listener = pynput_keyboard.Listener(on_press=_on_kb_press, on_release=_on_kb_release)
        _kb_listener.start()
    except Exception as exc:
        print(f"Linux: keyboard listener unavailable ({exc}); keyboard hotkeys disabled.")
        _kb_listener = None

def stop_keyboard_listener():
    global _kb_listener
    if _kb_listener is not None:
        try:
            _kb_listener.stop()
        except Exception as exc:
            print(f"keyboard listener stop warning: {exc}")
        _kb_listener = None
    _kb_held.clear()
    _kb_active_tuples.clear()

def apply_hotkeys():
    global mouse_listener
    if sys.platform == "win32":
        keyboard.unhook_all()
    stop_mouse_listener()
    if sys.platform != "win32":
        stop_keyboard_listener()
        _kb_chords.clear()

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
                        _register_chord(key, (hook(actions[action][0]), hook(actions[action][1])))
                else:
                    _register_chord(key, hook(actions[action]))
            elif "left_fingers" in action or "right_fingers" in action:
                _register_chord(key, lambda a=action: set_fingers(a))
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
        try:
            monitors = screeninfo.get_monitors()
        except Exception:
            return None
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
            if monitor is None:
                return
            x_normalized = (x / monitor.width - 0.5)
            y_normalized = -(y / monitor.height - 0.5)
            if g.config["Smoothing"]["enable"]:
                g.latest_data[117] = x_normalized
                g.latest_data[118] = y_normalized
            else:
                g.data["MousePosition"][0]["v"] = x_normalized
                g.data["MousePosition"][1]["v"] = y_normalized


    if mouse_actions:
        try:
            mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll, on_move=on_move)
            mouse_listener.start()
        except Exception as exc:
            print(f"Linux: mouse listener unavailable ({exc}); mouse hotkeys disabled.")
            mouse_listener = None
    if sys.platform != "win32":
        start_keyboard_listener()
    print("Start Hotkey")

def stop_hotkeys():
    global monitor
    if sys.platform == "win32":
        stop_mouse_listener()
        monitor = None
        keyboard.unhook_all()
        for item in g.hotkey_config["Hotkeys"]:
            if item["action"] == "toggle_hotkeys":
                keyboard.add_hotkey(item["key"], toggle_hotkeys)
    else:
        stop_mouse_listener()
        stop_keyboard_listener()
        monitor = None
        _kb_chords.clear()
        # keep the toggle hotkey alive so tracking can be re-enabled
        for item in g.hotkey_config["Hotkeys"]:
            if item["action"] == "toggle_hotkeys" and item.get("key"):
                _register_chord(item["key"], toggle_hotkeys)
        if _kb_chords:
            start_keyboard_listener()
    print("Stop Hotkey")

def linux_foreground_window_info():
    """Return (window_title, process_name) of the active X11 window.

    Uses python-xlib, which pynput already depends on. Returns (None, None)
    when there is no active X11 window (Wayland-native focus) or the query
    fails.
    """
    try:
        from Xlib import X, display as xdisplay
        d = xdisplay.Display()
        try:
            root = d.screen().root
            active = root.get_full_property(
                d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
            )
            if not active or not active.value or not active.value[0]:
                return None, None
            window = d.create_resource_object("window", int(active.value[0]))
            name_prop = window.get_full_property(
                d.intern_atom("_NET_WM_NAME"), X.AnyPropertyType
            )
            if name_prop is not None and name_prop.value:
                title = name_prop.value[0].decode("utf-8", "ignore")
            else:
                title = window.get_wm_name() or ""
            pid_prop = window.get_full_property(
                d.intern_atom("_NET_WM_PID"), X.AnyPropertyType
            )
            process_name = None
            if pid_prop and pid_prop.value:
                try:
                    process_name = os.path.basename(
                        psutil.Process(int(pid_prop.value[0])).exe()
                    )
                except Exception:
                    pass
            return title, process_name
        finally:
            d.close()
    except Exception:
        return None, None


# check title first then program name
def is_in_game():
    if sys.platform != "win32":
        target = g.config['Setting']["only_ingame_game"]
        if not target:
            return False
        title, process_name = linux_foreground_window_info()
        if title is None and process_name is None:
            # No active X11 window (Wayland-native focus or query failed):
            # assume the game is running so hotkeys keep working.
            return True
        return title == target or process_name == target
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
