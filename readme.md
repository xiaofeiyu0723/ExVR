# ExVR: Experience Virtual Reality

This is a project for PC users who want to experience VR without a headset.

## Usage

TODO

## Hotkeys

| **Key**                | **Action**                      |
|------------------------|---------------------------------|
| `ctrl+;`               | Reset eyes                      |
| `ctrl+;+1`             | Disable eye yaw                 |
| `ctrl+;+2`             | Disable all eye movements       |
| `ctrl+'`               | Reset head                      |
| `ctrl+[`               | Reset left hand position        |
| `ctrl+]`               | Reset right hand position       |
| `[+\``                  | Enable all fingers (left)       |
| `[+1`                  | Set finger 0 on left hand      |
| `[+2`                  | Set finger 1 on left hand      |
| `[+3`                  | Set finger 2 on left hand      |
| `[+4`                  | Set finger 3 on left hand      |
| `[+5`                  | Set finger 4 on left hand      |
| `]+\``                  | Enable all fingers (right)      |
| `]+1`                  | Set finger 0 on right hand     |
| `]+2`                  | Set finger 1 on right hand     |
| `]+3`                  | Set finger 2 on right hand     |
| `]+4`                  | Set finger 3 on right hand     |
| `]+5`                  | Set finger 4 on right hand     |
| `ctrl+up`              | Move up                         |
| `ctrl+down`            | Move down                       |
| `ctrl+left`            | Move left                       |
| `ctrl+right`           | Move right                      |
| `ctrl+=`               | Head pitch up                   |
| `ctrl+-`               | Head pitch down                 |
| `=`                    | Grab with left hand             |
| `-`                    | Grab with right hand            |
| `[`                    | Trigger with left hand          |
| `]`                    | Trigger with right hand         |
| `left` (mouse)         | Trigger with left hand          |
| `right` (mouse)        | Trigger with right hand         |
| `middle+left` (mouse)  | Grab with left hand            |
| `middle+right` (mouse) | Grab with right hand           |
| `scroll_up`            | Move joystick up                |
| `scroll_down`          | Move joystick down              |
| `middle` (mouse)       | Activate joystick middle right   |


## Configuration

### config.json

The `config.json` file, located in the root (`./`) directory, allows users to configure various components, including camera, IP settings, smoothing, and tracking parameters.

#### General Settings  

| Key        | Description                                      |
|------------|--------------------------------------------------|
| Camera     | Specifies the camera index for input.            |
| IP         | Defines the IP address for connection settings.   |
| Smoothing  | Enables or disables smoothing for movements.      |

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
