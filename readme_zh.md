# ExVR: 体验虚拟现实
为无VR玩家提供更好的体验

## 语言
[简体中文](readme_zh.md) / [English](readme.md)

## 用法

TODO

## 按键设置
### 快捷键

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
| `[+1`           | 左手手指0状态切换               |
| `[+2`           | 左手手指1状态切换              |
| `[+3`           | 左手手指2状态切换              |
| `[+4`           | 左手手指3状态切换              |
| `[+5`           | 左手手指4状态切换              |
| ]+`             | 启用所有手指（右）                |
| `]+1`           | 右手手指0状态切换              |
| `]+2`           | 右手手指1状态切换              |
| `]+3`           | 右手手指2状态切换              |
| `]+4`           | 右手手指3状态切换              |
| `]+5`           | 右手手指4状态切换              |
| `ctrl+up`       | 上升                            |
| `ctrl+down`     | 下降                            |
| `ctrl+left`     | 向左移动                         |
| `ctrl+right`    | 向右移动                         |
| `up`            | 抬头                            |
| `down`          | 低头                            |
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

### VRChat自带键位

| **按键**        | **动作**                         |
|-----------------|---------------------------------|
| `wsad`          | 前后左右移动                     |
| `ESC`           | 菜单（多次连续点击切换到固定菜单） |
| `,` (<)         | 头部向左偏移                     |
| `.` (>)         | 头部向右偏移                      |

## 配置

### config.json

`config.json` 文件, 被存放在本地root根目录 (`./settings`) 中， 修改用户配置的参数，这些参数包括有摄像机、IP设置、平滑和跟踪参数

#### 常规设置  

| 参数        | 描述                                      |
|------------|-------------------------------------------|
| Camera     | 指定输入的摄像机索引                        |
| IP         | 定义连接设置的IP地址                        |
| Smoothing  | `启用` / `禁用` 平滑移动                        |

#### Tracking设置  

| **项目**  | **参数**                     | **描述**              |                                                                
|----------------|------------------------------------|------------------------------------------------------------|
| **Head**       | enable                            | 激活头部追踪 (`true` / `false`)                              |
|                | x_scalar, y_scalar, z_scalar       | 调整各个轴上头部位置的灵敏度                                 |         
|                | x_rotation_scalar, y_rotation_scalar, z_rotation_scalar | 调整头部的旋转灵敏度 |
| **Face**       | enable                            | 启用面捕                                           |
| **Tongue**     | enable                            | 启用舌头动态捕捉                        |
|                | tougue_confidence                 | 舌头动态捕捉的置信度阈值 |
|                | tongue_threshold                  | 识别舌头运动的阈值 |
|                | tongue_x_scalar, tongue_y_scalar  | 调整舌头动态捕捉的灵敏度 |
| **Mouth**      | enable                            | 启用嘴部动态捕捉                        |
|                | mouth_close_threshold             | 嘴部闭合阈值 |
| **Hand**       | enable                            | 启用手部追踪 (`true` / `false`)                              |
|                | x_scalar, y_scalar, z_scalar       | 调整各个轴上手部位置的灵敏度                                 |
|                | hand_confidence                   | 手部追踪的置信度阈值 |
|                | hand_delta_threshold              | 误识别检测的变化量阈值 |
|                | hand_shifting_threshold           | 误识别检测的偏移阈值 |
|                | enable_hand_auto_reset            | 自动重置手部位置 (`true` / `false`) |
|                | hand_detection_upper_threshold    | 手部检测的上限阈值 |
|                | hand_detection_lower_threshold    | 手部检测的下限阈值 |
|                | hand_count_threshold              | 手部的计数阈值 |
|                | only_front                        | 仅允许手在前方移动  |
| **Finger**     | enable                            | 启用手指追踪 (`true` / `false`)                              |
|                | finger_confidence                 | 手指检测的最低置信度 |
|                | finger_threshold                  | 手指状态（张开/收紧）的阈值 |



#### 模型设置 

| **模块** | **参数**                     | **描述**              |                                                          
|-----------|-------------------------------|----------|
| **Face Modle** | min_face_detection_confidence | 面部检测所需的最低置信度 |
|             | min_face_presence_confidence  | 检测面部存在的最低置信度 |
|             | min_tracking_confidence       | 维持面部跟踪的最低置信度 |
| **Hand Model** | min_hand_detection_confidence | 手部检测所需的最低置信度 |
|             | min_hand_presence_confidence  | 检测手部存在的最低置信度 |
|             | min_tracking_confidence       | 维持手部跟踪的最低置信度 |  



### data.json

文件`data.json`位于根目录（`./settings`）中，包含虚拟体验的初始设置，包括位置、旋转和形态键

- **Position**：定义头部的3D坐标
- **Rotation**：定义头部绕各个轴的旋转
- **BlendShapes**：包含各种面部表情
- **LeftHandPosition** / **RightHandPosition**：定义左右手的位置
- **LeftHandRotation** / **RightHandRotation**：定义左右手的旋转
- **LeftHandFinger** / **RightHandFinger**：控制每个手指的运动

每个条目包含以下内容：
- **k**：键（属性的名称）
- **v**：属性的默认值
- **s**：调整值的偏移量
- **e**：启用标志（`true` / `false`）以激活或禁用设置


### smoothing.json

文件`smoothing.json`位于根目录（`./settings`）中，用于设置平滑各种运动和形态键

- **OtherBlendShapes**：控制一般面部形态键的平滑度
- **EyeBlink**：平滑眼皮移动
- **EyeLook**：平滑眼睛移动
- **TongueOut**：平滑舌头伸出移动
- **TongueMove**：平滑舌头的左右移动
- **HeadPosition**：平滑头部位置移动
- **HeadRotation**：平滑头部旋转移动
- **LeftHandPosition** / **RightHandPosition**：平滑手部位置
- **LeftHandRotation** / **RightHandRotation**：平滑手部旋转
- **LeftHandFinger** / **RightHandFinger**：平滑手指运动

如果您不是对该应用程序进行二次开发，请避免修改以下参数：
- **key**：键
- **is_rotation**：属性是否为旋转
- **indices**：数据索引
- **shifting**：索引的偏移量

你**可以修改**以下参数以获得对自身更好的体验：  
- **max_delta**：允许的最大变化量。较高的值使动作更敏感，但可能引入抖动
- **deadzone**：忽略小幅运动的死区。较大的死区减少抖动，但可能使动作感觉不够流畅
- **dt_multiplier**：平滑因子。较小的值产生更平滑的运动，但可能降低响应速度


### hotkeys.json

文件`hotkeys.json`被存放在根目录(`./settings`)中，定义了所有可用的键盘和鼠标快捷键,你可以修改这个文件来自定义

## 参考项目

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
