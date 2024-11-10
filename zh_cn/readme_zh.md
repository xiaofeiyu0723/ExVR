---
tags: [Import-34b4]
title: ExVR꞉ 体验虚拟现实
created: '2024-11-10T11:45:01.211Z'
modified: '2024-11-10T12:12:33.019Z'
---

# ExVR: 体验虚拟现实

这是一个为没有 VR 设备的用户但想要体验 VR 的 PC 用户准备的项目。

## 用法

TODO

## 快捷键

| **按键**         | **动作**                      |
|-----------------|---------------------------------|
| \\+`             | 切换快捷键                      |
| `ctrl+'`        | 重置头部                         |
| `ctrl+;`        | 重置眼部                         |
| `ctrl+;+1`      | 禁用眼部偏航                     |
| `ctrl+;+2`      | 禁用全部眼球运动                 |
| `ctrl+[`        | 校准手部位置（左）                |
| `ctrl+]`        | 校准手部位置（右）                |
| [+`             | 启用所有手指（左）                |
| `[+1`           | 手指 0 设置在左手上              |
| `[+2`           | 手指 1 设置在左手上              |
| `[+3`           | 手指 2 设置在左手上              |
| `[+4`           | 手指 3 设置在左手上              |
| `[+5`           | 手指 4 设置在左手上              |
| ]+`             | 启用所有手指（右）                |
| `]+1`           | 手指 0 设置在左手上              |
| `]+2`           | 手指 1 设置在左手上              |
| `]+3`           | 手指 2 设置在左手上              |
| `]+4`           | 手指 3 设置在左手上              |
| `]+5`           | 手指 4 设置在左手上              |
| `ctrl+up`       | 上升                            |
| `ctrl+down`     | 下降                            |
| `ctrl+left`     | 向左移动                         |
| `ctrl+right`    | 向右移动                         |
| `up`            | 抬头                            |
| `down`          | 低头                            |
| `,` (<)         | 头部向左偏移                     |
| `.` (>)         | 头部向右偏移                     |
| `=`             | 左手握持键                       |
| `-`             | 右手握持键                       |
| `[`             | 左手扳机键                       |
| `]`             | 右手扳机键                       |
| `left` (mouse)  | 左手扳机键                       |
| `right` (mouse) | 右手扳机键                       |
| `left` (mouse)  | 左手握持键                       |
| `right` (mouse) | 右手握持键                       |
| `scroll_up`     | 移动摇杆向上                     |
| `scroll_down`   | 移动摇杆向下                     |
| `middle` (mouse) | Activate joystick middle right(看不懂思密达)  |


## 配置

### config.json

`config.json` 文件, 被存放在本地root根目录 (`./`) 中， 修改用户配置的参数，这些参数包括有摄像机、IP设置、平滑和跟踪参数。

#### 常规设置  

| 参数        | 描述                                      |
|------------|-------------------------------------------|
| Camera     | 指定输入的摄像机索引                        |
| IP         | 定义连接设置的IP地址                        |
| Smoothing  | 启用 / 禁用 平滑移动                        |

#### Tracking Settings  

| **Component**  | **Parameter**                     | **Description**                                    |
|----------------|------------------------------------|----------------------------------------------------|
| **Head**       | enable                            | Activates head tracking (`true` or `false`).       |
|                | x_scalar, y_scalar, z_scalar       | Adjust sensitivity for head position in each axis. |
|                | x_rotation_scalar, y_rotation_scalar, z_rotation_scalar | Adjust sensitivity for head rotation. |
| **Face**       | enable                            | Activates face tracking.                          |
| **Tongue**     | enable                            | Activates tongue tracking.                        |
|                | tongue_confidence                 | Minimum confidence for tongue detection.          |
|                | tongue_threshold                  | Threshold to recognize tongue movements.          |
|                | tongue_x_scalar, tongue_y_scalar  | Adjust sensitivity for tongue movements.          |
|                | mouth_close_threshold             | Threshold to detect a closed mouth.               |
| **Hand**       | enable                            | Activates hand tracking.                          |
|                | x_scalar, y_scalar, z_scalar       | Adjust sensitivity for hand position in each axis. |
|                | hand_confidence                   | Minimum confidence for hand detection.            |
|                | hand_delta_threshold              | Minimum movement required for detection.          |
|                | hand_shifting_threshold           | Minimum shifting required for detection.          |
|                | enable_hand_auto_reset            | Automatically resets hand position (`true`/`false`). |
|                | hand_detection_upper_threshold    | Upper threshold for hand detection.               |
|                | hand_detection_lower_threshold    | Lower threshold for hand detection.               |
|                | hand_count_threshold              | Minimum number of hands required for detection.   |
|                | only_front                        | Limits tracking to front-facing hands.            |
| **Finger**     | enable                            | Activates finger tracking.                        |
|                | finger_confidence                 | Minimum confidence for finger detection.          |
|                | finger_threshold                  | Sensitivity threshold for finger movements.       |

#### Model Settings  

| **Model**      | **Parameter**                      | **Description**                                    |
|----------------|------------------------------------|----------------------------------------------------|
| **Face Model** | min_face_detection_confidence      | Minimum confidence required for face detection.    |
|                | min_face_presence_confidence       | Minimum confidence for detecting face presence.    |
|                | min_tracking_confidence            | Minimum confidence for maintaining face tracking. |
| **Hand Model** | min_hand_detection_confidence      | Minimum confidence required for hand detection.    |
|                | min_hand_presence_confidence       | Minimum confidence for detecting hand presence.    |
|                | min_tracking_confidence            | Minimum confidence for maintaining hand tracking. |

### data.json

The `data.json` file, located in the root (`./`) directory, contains the initial settings for the virtual experience, including position, rotation, and blend shapes.

- **Position**: Defines the 3D coordinates for the head position.
- **Rotation**: Specifies the head rotation around the axes.
- **BlendShapes**: Contains various facial expressions.
- **LeftHandPosition** / **RightHandPosition**: Specifies the positions of the hands.
- **LeftHandRotation** / **RightHandRotation**: Defines the rotation of the hands.
- **LeftHandFinger** / **RightHandFinger**: Controls the movement of each finger.

Each entry contains the following:
- **k**: Key (the name of the property).
- **v**: Default value for the property.
- **s**: Offset value used for adjustment.
- **e**: Enable flag (`true` or `false`) to activate or deactivate the setting.

### smoothing.json

The `smoothing.json` file, located in the root (`./`) directory, contains parameters for smoothing various movements and blend shapes.

- **OtherBlendShapes**: Controls smoothing for general facial blend shapes.  
- **EyeBlink**: Adjusts the responsiveness of eye blinking.  
- **EyeLook**: Smooths eye movement, including gaze direction.  
- **TongueOut**: Controls the tongue-out animation smoothing.  
- **TongueMove**: Smooths left and right tongue movements.  
- **HeadPosition**: Manages smoothing for head position adjustments.  
- **HeadRotation**: Smooths head rotation movements.  
- **LeftHandPosition** / **RightHandPosition**: Adjusts hand position smoothing.  
- **LeftHandRotation** / **RightHandRotation**: Manages the smoothness of hand rotations.  
- **LeftHandFinger** / **RightHandFinger**: Controls smoothing for individual finger movements.

If you are not developing the application, avoid modifying the following fields:
- **key**: Identifies the target property.
- **is_rotation**: Indicates if the property is a rotation.
- **indices**: Specifies the relevant data indices.
- **shifting**: Indices shifting.

You **can modify** the following settings to fine-tune the experience:  
- **max_delta**: Maximum allowed change in value. Higher values make movements more sensitive but can introduce jitter.
- **deadzone**: The range within which small movements are ignored. A larger dead zone reduces jitter but may make movements feel less smooth.
- **dt_multiplier**: Smoothing factor. Smaller values produce smoother movements but may slow down responsiveness.

### hotkeys.json

The `hotkeys.json` file, located in the root (`./`) directory, defines all the available keyboard and mouse shortcuts for interacting. You can modify this file to customize.

## Credits

- **Tracking Module**
  - [mediapipe-vt](https://github.com/nuekaze/mediapipe-vt)
  - [Mediapipe-VR-Fullbody-Tracking](https://github.com/ju1ce/Mediapipe-VR-Fullbody-Tracking/)
- **HMD OpenVR Drivers**
  - [OpenVR-OpenTrack](https://github.com/r57zone/OpenVR-OpenTrack)
  - [VRto3D](https://github.com/oneup03/VRto3D)
- **Hand OpenVR Driver**
  - [VMT (Virtual Motion Tracker)](https://github.com/gpsnmeajp/VirtualMotionTracker/)
- **VRCFaceTracking Module**
  - [VRCFT-MediaPipe](https://github.com/Codel1417/VRCFT-MediaPipe)
  - [VRCFaceTracking-LiveLink](https://github.com/kusomaigo/VRCFaceTracking-LiveLink/)
