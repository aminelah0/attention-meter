# Import libraries
from mediapipe.python.solutions import face_detection, face_mesh
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from attention.img_proc.img_process import resize_image
import face_recognition
from attention.params import *
from attention.utils.utilities import distance_coord


def detect_face(image_rgb: np.ndarray) -> list[dict]:
    ''' Takes an image array, detects the faces and returns a list of bbox coordinates for each face detected'''
    with face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_det:
        results = face_det.process(image_rgb)
        coord_set = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_rgb.shape
                x1 = int(bbox.xmin * w) - int(0.05 * int(bbox.xmin * w))
                y1 = int(bbox.ymin * h) - int(0.05 * int(bbox.ymin * h))
                x2 = int((bbox.xmin + bbox.width) * w) + int(0.05 * int(bbox.xmin * w))
                y2 = int((bbox.ymin + bbox.height) * h) + int(0.05 * int(bbox.ymin * h))
                coordinates = {'x1':x1,'y1':y1,f'x2':x2,'y2':y2}
                coord_set.append(coordinates)

    return coord_set


def find_landmarks(face: np.ndarray) -> NormalizedLandmarkList:
    '''Takes the image of a face and return all landmarks of the face. Returns None if no landmark detected'''
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        min_detection_confidence=0.5,
        refine_landmarks=True,
        min_tracking_confidence=0.5) as face_meshe:

        results = face_meshe.process(face)

        face_mp_landmarks = None
        if results.multi_face_landmarks:
            face_mp_landmarks = results.multi_face_landmarks[0]

        return face_mp_landmarks


def detect_eye_directions(face_landmarks: list[tuple], threshold: float = 0.63) -> dict:
    '''Determines if each eye is looking straight or sideways --> left eye is considered to look sideways if it looks to the left, otherwise it is said to look straight'''
    eye_directions = dict()

    #LEFT EYE
    left_iris_lm = face_landmarks[LEFT_IRIS_CENTER[0]]
    left_eye_inside_lm = face_landmarks[LEFT_EYE_EDGES[0]]
    left_eye_outside_lm = face_landmarks[LEFT_EYE_EDGES[1]]
    left_eye_length = distance_coord(left_eye_outside_lm, left_eye_inside_lm, axis='x')
    left_iris2inside = distance_coord(left_iris_lm, left_eye_inside_lm, axis='x')
    left_ratio =  round(left_iris2inside/ left_eye_length, 2)

    eye_directions['left'] = ('sideway' if left_ratio > threshold else 'straight',
                              left_ratio)

    #RIGHT EYE
    right_iris_lm = face_landmarks[RIGHT_IRIS_CENTER[0]]
    right_eye_inside_lm = face_landmarks[RIGHT_EYE_EDGES[1]]
    right_eye_outside_lm = face_landmarks[RIGHT_EYE_EDGES[0]]
    right_eye_length = distance_coord(right_eye_outside_lm, right_eye_inside_lm, axis='x')
    right_iris2inside = distance_coord(right_iris_lm, right_eye_inside_lm, axis='x')
    right_ratio =  round(right_iris2inside/ right_eye_length, 2)

    eye_directions['right'] = ('sideway' if right_ratio > threshold else 'straight',
                               right_ratio)

    return eye_directions


def detect_head_direction(face_landmarks: list[tuple], left_threshold: float = 0.35, right_threshold: float = 0.35) -> tuple:
    '''Determines the direction of the head based on the distance eye edge/ nose'''

    nose_lm = face_landmarks[NOSE[0]]
    left_eye_outside_lm = face_landmarks[LEFT_EYE_EDGES[1]]
    right_eye_outside_lm = face_landmarks[RIGHT_EYE_EDGES[0]]
    left_eye2nose = distance_coord(left_eye_outside_lm, nose_lm, axis='xy')
    right_eye2nose = distance_coord(right_eye_outside_lm, nose_lm, axis='xy')
    left_right_ratio =  round(left_eye2nose/ right_eye2nose, 2)

    if left_right_ratio < 1 - left_threshold:
        direction = 'head left'
    elif left_right_ratio > 1 + right_threshold:
        direction = 'head right'
    else:
        direction = 'head centered'

    head_direction = (direction, left_right_ratio)

    return head_direction


def detect_head_inclination(face_landmarks: list[tuple], down_threshold: float = 1.73, up_threshold: float = 0.8) -> dict:
    '''Determines subject's head is positioned up or down, or none (i.e. straight)'''
    forhead= face_landmarks[FOREHEAD_MIDDLE[0]]
    bottomlip = face_landmarks[BOTTOM_LIP[0]]
    nose = face_landmarks[NOSE[0]]
    forhead_nose_vector = distance_coord(forhead, nose, axis='y')
    bottomlip_nose_vector = distance_coord(nose, bottomlip, axis='y')
    forhead_bottomlip_ratio = forhead_nose_vector / bottomlip_nose_vector
    if forhead_bottomlip_ratio > down_threshold:
        direction_up = 'head down'
    elif forhead_bottomlip_ratio < up_threshold:
        direction_up = 'head up'
    else:
        direction_up = 'head level'

    head_inclination = (direction_up, round(forhead_bottomlip_ratio, 2))

    return head_inclination


def is_attentive(eye_directions: dict, head_direction: tuple, head_inclination: tuple) -> tuple[bool, str]:
    '''Determines if a face is attentive based on the eyes direction and head direction/inclination
    Returns a tuple (attention_bool, driver of attention):
    -- "HD": head down
    -- "EM": Eyes & head Mismatch'''
    left_eye_direction = eye_directions['left'][0]
    right_eye_direction = eye_directions['right'][0]

    if head_inclination[0] == 'head down':
        return False, 'HD'
    elif left_eye_direction == 'straight' and right_eye_direction == 'straight' and head_direction[0] == 'head centered':
        return True, ''
    elif left_eye_direction == 'sideways' and head_direction[0] == 'head right':
        return True, ''
    elif right_eye_direction == 'sideways' and head_direction[0] == 'head left':
        return True,''
    else:
        return False, 'EM'


def train_faces(known_faces: list[np.ndarray], known_names: list[str]) -> dict:
    '''Takes lists of known faces (np array) and corresponding names and returns a dictionary {name: encoding}'''
    known_encodings = {}
    for i in range(len(known_faces)):
        img_enc = face_recognition.face_encodings(known_faces[i])[0]
        known_encodings[known_names[i]] = img_enc

    return known_encodings


def recognize_face(face: np.ndarray, known_encodings: dict, threshold: float = 0.61) -> dict:
    '''Takes a face and returns prediction for the person in a dictionary: {"Detected_person": str, "distance": value}'''
    list_known_encodings = list(known_encodings.values())
    list_known_names = list(known_encodings.keys())

    # Resize image if too big to save some processing power
    if face.shape[0] > 500:
        face = resize_image(face, 500)

    try:
        face_encoding = face_recognition.face_encodings(face)
    except:
        #Re-try to encode face with upscaled image
        try:
            face_resized = resize_image(face, 400)
            face_encoding = face_recognition.face_encodings(face_resized)
        except:
            face_encoding = None
            return ("No Face", np.nan)

    if face_encoding:
        img_enc = face_encoding[0]
        distance = list(face_recognition.face_distance(list_known_encodings,img_enc))

        min_distance = np.amin(distance)
        min_index = distance.index(min_distance)
        prediction_recognition = list_known_names[min_index] if min_distance < threshold else np.nan
        prediction_distance = round(min_distance, 2)

        return (prediction_recognition, prediction_distance)

    return (np.nan, np.nan)
