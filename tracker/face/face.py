import mediapipe as mp
import numpy as np
import math
from tracker.face.tongue import mouth_roi_on_image, detect_tongue
import utils.globals as g
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from scipy.spatial.transform import Rotation as R

def draw_face_landmarks(rgb_image):
    face_landmarks_list = g.face_landmarks

    if face_landmarks_list is None:
        return rgb_image

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])
        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=rgb_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return rgb_image

def is_hand_in_face():

    if not g.config["Tracking"]["Face"]["block"] or not g.face_landmarks or not g.hand_landmarks:
        return 0.0

    face_x = [f.x for f in g.face_landmarks[0]]
    face_y = [f.y for f in g.face_landmarks[0]]

    min_x, max_x = min(face_x), max(face_x)
    min_y, max_y = min(face_y), max(face_y)
    face_area = (max_x - min_x) * (max_y - min_y)

    blocked_area = 0.0

    # target_indices = [0, 1, 2, 5, 9, 13, 17, 6, 10, 14, 18]
    target_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for hand in g.hand_landmarks:
        hand_points = [(h.x, h.y) for i, h in enumerate(hand.landmark) if
                       i in target_indices and min_x <= h.x <= max_x and min_y <= h.y <= max_y]
        if hand_points:
            hand_min_x = min([p[0] for p in hand_points])
            hand_max_x = max([p[0] for p in hand_points])
            hand_min_y = min([p[1] for p in hand_points])
            hand_max_y = max([p[1] for p in hand_points])
            blocked_area += (hand_max_x - hand_min_x) * (hand_max_y - hand_min_y)

    if blocked_area > 0:
        normalized_area = blocked_area / face_area
        return normalized_area
    return 0.0

head_position_prev=None
head_position=None
def face_pred_handling(detection_result, output_image, timestamp_ms, tongue_model):
    global head_position_prev,head_position
    # For each face detected
    for idx in range(len(detection_result.face_landmarks)):
        g.face_landmarks = detection_result.face_landmarks

        # Block handling
        coverage_ratio = is_hand_in_face()

        # Head Image Position
        head_image_position_x = detection_result.face_landmarks[0][4].x
        head_image_position_y = detection_result.face_landmarks[0][4].y
        head_image_position_z = detection_result.face_landmarks[0][4].z

        if not coverage_ratio > g.config["Tracking"]["Face"]["face_block_threshold"]:
            for i in range(len(detection_result.face_blendshapes[0])):
                if g.config["Smoothing"]["enable"]:
                    g.latest_data[i] = detection_result.face_blendshapes[0][i].score
                else:
                    g.data["BlendShapes"][i]["v"] = detection_result.face_blendshapes[0][i].score
        # Eye with shifting
        EyeYawLeft = (-g.data["BlendShapes"][15]["v"] + g.data["BlendShapes"][13]["v"] -g.data["BlendShapes"][14]["v"] + g.data["BlendShapes"][16]["v"])/2
        EyeYawRight = EyeYawLeft
        EyePitchLeft = np.clip((g.data["BlendShapes"][11]["v"] - 0.5) + (g.data["BlendShapes"][12]["v"] - 0.5),-1,0.1)
        EyePitchRight=EyePitchLeft

        # Tongue detection
        if g.config["Tracking"]["Tongue"]["enable"]:
            mouth_image = mouth_roi_on_image(
                output_image.numpy_view(), detection_result.face_landmarks[0]
            )
            # print(233)
            # cv2.imwrite("test.png",mouth_image)
            tongue_out, tongue_x, tongue_y = detect_tongue(
                mouth_image, tongue_model, g.data
            )
        else:
            tongue_out, tongue_x, tongue_y= None, None, None

        # Head Position
        mat = np.array(detection_result.facial_transformation_matrixes[0])
        position_x = -mat[0][3] * g.config["Tracking"]["Head"]["x_scalar"]
        position_y = -mat[2][3] * g.config["Tracking"]["Head"]["z_scalar"]
        position_z = mat[1][3] * g.config["Tracking"]["Head"]["y_scalar"]
        head_position_temp=np.array([position_x, position_y,position_z])
        if head_position_prev is None:
            head_position_prev = head_position_temp.copy()
            head_position = head_position_temp.copy()
        else:
            head_position_diff = head_position_temp - head_position_prev
            head_position_prev = head_position_temp.copy()
            yaw_calibration = g.data["Rotation"][0]["s"]
            # pitch_calibration = g.data["Rotation"][1]["s"]
            # roll_calibration = g.data["Rotation"][2]["s"]
            calibration_rot = R.from_euler("z", -yaw_calibration, degrees=True)
            calibration_matrix = calibration_rot.as_matrix()
            calibrated_diff = calibration_matrix @ head_position_diff
            head_position += calibrated_diff

        rotation_yaw = (
            -np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
                * 180
                / math.pi
            * g.config["Tracking"]["Head"]["yaw_rotation_scalar"]
        )
        rotation_pitch = (
            -np.arctan2(mat[2, 1], mat[2, 2])
            * 180
            / math.pi
            * g.config["Tracking"]["Head"]["pitch_rotation_scalar"]
        )
        rotation_roll = (
            -np.arctan2(mat[1, 0], mat[0, 0])
            * 180
            / math.pi
            * g.config["Tracking"]["Head"]["roll_rotation_scalar"]
        )
        head_rotation=np.array([rotation_yaw, rotation_pitch,rotation_roll])

        # Update g.latest_data or data directly
        if g.config["Smoothing"]["enable"]:
            if not coverage_ratio > g.config["Tracking"]["Face"]["face_block_threshold"]:
                # Head Blendshape
                g.latest_data[53] = 0.0
                g.latest_data[54] = 0.0
                g.latest_data[55] = 0.0

                # Eye with shifting
                g.latest_data[56] = EyeYawLeft
                g.latest_data[57] = EyePitchLeft
                g.latest_data[58] = 0.0
                g.latest_data[59] = EyeYawRight
                g.latest_data[60] = EyePitchRight
                g.latest_data[61] = 0.0

                # Tongue
                if g.config["Tracking"]["Tongue"]["enable"] and tongue_out is not None and tongue_x is not None and tongue_y is not None:
                    g.latest_data[52] = tongue_out
                    g.latest_data[62] = tongue_x
                    g.latest_data[63] = tongue_y

            if g.config["Tracking"]["Head"]["enable"]:
                if not coverage_ratio > g.config["Tracking"]["Face"]["position_block_threshold"]:
                    # Head Position
                    g.latest_data[64] = head_position[0]
                    g.latest_data[65] = head_position[1]
                    g.latest_data[66] = head_position[2]

                if not coverage_ratio > g.config["Tracking"]["Face"]["rotation_block_threshold"]:
                    # Head Rotation
                    g.latest_data[67] = head_rotation[0]
                    g.latest_data[68] = head_rotation[1]
                    g.latest_data[69] = head_rotation[2]
            if not coverage_ratio > g.config["Tracking"]["Face"]["position_block_threshold"]:
                g.latest_data[114] = head_image_position_x
                g.latest_data[115] = head_image_position_y
                g.latest_data[116] = head_image_position_z

        else:
            if not coverage_ratio > g.config["Tracking"]["Face"]["face_block_threshold"]:
                # Head Blendshape
                g.data["BlendShapes"][53]["v"] = 0.0
                g.data["BlendShapes"][54]["v"] = 0.0
                g.data["BlendShapes"][55]["v"] = 0.0

                # Eye with shifting
                g.data["BlendShapes"][56]["v"] = EyeYawLeft
                g.data["BlendShapes"][57]["v"] = EyePitchLeft
                g.data["BlendShapes"][58]["v"] = 0.0
                g.data["BlendShapes"][59]["v"] = EyeYawRight
                g.data["BlendShapes"][60]["v"] = EyePitchRight
                g.data["BlendShapes"][61]["v"] = 0.0

                # Tongue
                if g.config["Tracking"]["Tongue"]["enable"] and tongue_out is not None and tongue_x is not None and tongue_y is not None:
                    g.data["BlendShapes"][52]["v"] = tongue_out
                    g.data["BlendShapes"][62]["v"] = tongue_x
                    g.data["BlendShapes"][63]["v"] = tongue_y
            if g.config["Tracking"]["Head"]["enable"]:
                if not coverage_ratio > g.config["Tracking"]["Face"]["position_block_threshold"]:
                    # Head Position
                    g.data["Position"][0]["v"] = head_position[0]
                    g.data["Position"][1]["v"] = head_position[1]
                    g.data["Position"][2]["v"] = head_position[2]
                if not coverage_ratio > g.config["Tracking"]["Face"]["rotation_block_threshold"]:
                    # Head Rotation
                    g.data["Rotation"][0]["v"] = head_rotation[0]
                    g.data["Rotation"][1]["v"] = head_rotation[1]
                    g.data["Rotation"][2]["v"] = head_rotation[2]
            if not coverage_ratio > g.config["Tracking"]["Face"]["position_block_threshold"]:
                g.data["HeadImagePosition"][0]["v"] = head_image_position_x
                g.data["HeadImagePosition"][1]["v"] = head_image_position_y
                g.data["HeadImagePosition"][2]["v"] = head_image_position_z




def initialize_face(tongue_model):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model = BaseOptions(model_asset_path="./model/face_landmarker.task")
    options = FaceLandmarkerOptions(
        base_options=model,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=g.config["Model"]["Face"]["min_face_detection_confidence"],
        min_face_presence_confidence=g.config["Model"]["Face"]["min_face_presence_confidence"],
        min_tracking_confidence=g.config["Model"]["Face"]["min_tracking_confidence"],
        num_faces=1,
        # running_mode=VisionRunningMode.IMAGE,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=lambda detection_result, output_image, timestamp_ms: face_pred_handling(
            detection_result, output_image, timestamp_ms, tongue_model),
    )
    return FaceLandmarker.create_from_options(options)
