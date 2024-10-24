import mediapipe as mp
import numpy as np
import math
from tracker.face.tongue import mouth_roi_on_image, detect_tongue
import utils.globals as g
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

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
    if not g.face_landmarks or not g.hand_landmarks:
        return False, 0
    # 获取面部landmarks的边界
    face_x = [f.x for f in g.face_landmarks[0]]
    face_y = [f.y for f in g.face_landmarks[0]]

    min_x, max_x = min(face_x), max(face_x)
    min_y, max_y = min(face_y), max(face_y)

    block_point_count = 0
    for hand in g.hand_landmarks:
        for h in hand:
            if min_x <= h.x <= max_x and min_y <= h.y <= max_y:
                block_point_count +=1
    if block_point_count !=0:
        return True, block_point_count
    return False, 0
def pred_callback(detection_result, output_image, timestamp_ms, tongue_model):
    # For each face detected
    for idx in range(len(detection_result.face_landmarks)):
        # Block handling
        block_flag, block_num = is_hand_in_face()
        if g.config["Tracking"]["Face"]["block"] and block_flag and block_num > g.config["Tracking"]["Face"]["block_threshold"]:
            continue
        # Handle blendshapes
        for i in range(len(detection_result.face_blendshapes[0])):
            if g.config["Smoothing"]["enable"]:
                g.latest_data[i] = detection_result.face_blendshapes[0][i].score
            else:
                g.data["BlendShapes"][i]["v"] = detection_result.face_blendshapes[0][i].score

        g.face_landmarks = detection_result.face_landmarks
        # Head Image Position
        g.head_pos[0] = detection_result.face_landmarks[0][4].x
        g.head_pos[1] = detection_result.face_landmarks[0][4].y
        g.head_pos[2] = detection_result.face_landmarks[0][4].z

        # Eye with shifting
        EyeYawLeft = -g.data["BlendShapes"][15]["v"] + g.data["BlendShapes"][13]["v"]
        EyeYawRight = -g.data["BlendShapes"][14]["v"] + g.data["BlendShapes"][16]["v"]
        EyePitchLeft = (g.data["BlendShapes"][11]["v"] - 0.5) * 2
        EyePitchRight = (g.data["BlendShapes"][12]["v"] - 0.5) * 2
        
        # Tongue detection
        if g.config["Tracking"]["Tongue"]["enable"]:
            mouth_image = mouth_roi_on_image(
                output_image.numpy_view(), detection_result.face_landmarks[0]
            )
            tongue_out, tongue_x, tongue_y = detect_tongue(
                mouth_image, tongue_model, g.data
            )
        else:
            tongue_out, tongue_x, tongue_y= None, None, None

        # Head Position
        mat = np.array(detection_result.facial_transformation_matrixes[0])
        position_x = -mat[0][3] * g.config["Tracking"]["Head"]["x_scalar"]
        position_y = mat[1][3] * g.config["Tracking"]["Head"]["y_scalar"]
        position_z = -mat[2][3] * g.config["Tracking"]["Head"]["z_scalar"]

        # Head Rotation
        rotation_x = (
            -(
                np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
                * 180
                / math.pi
            )
            * g.config["Tracking"]["Head"]["x_rotation_scalar"]
        )
        rotation_y = (
            -np.arctan2(mat[1, 0], mat[0, 0])
            * 180
            / math.pi
            * g.config["Tracking"]["Head"]["y_rotation_scalar"]
        )
        rotation_z = (
            -np.arctan2(mat[2, 1], mat[2, 2])
            * 180
            / math.pi
            * g.config["Tracking"]["Head"]["z_rotation_scalar"]
        )

        # Update g.latest_data or data directly
        if g.config["Smoothing"]["enable"]:
            # Update g.latest_data
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

            # Head Position
            g.latest_data[64] = position_x
            g.latest_data[65] = position_z
            g.latest_data[66] = position_y

            # Head Rotation
            g.latest_data[67] = rotation_x
            g.latest_data[68] = rotation_z
            g.latest_data[69] = rotation_y
        else:
            # Update data directly
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

            # Head Position
            g.data["Position"][0]["v"] = position_x
            g.data["Position"][1]["v"] = position_z
            g.data["Position"][2]["v"] = position_y

            # Head Rotation
            g.data["Rotation"][0]["v"] = rotation_x
            g.data["Rotation"][1]["v"] = rotation_z
            g.data["Rotation"][2]["v"] = rotation_y


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
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=lambda detection_result, output_image, timestamp_ms: pred_callback(
            detection_result, output_image, timestamp_ms, tongue_model),
    )
    return FaceLandmarker.create_from_options(options)
