import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils
from attention.utils.img_vid_utils import resize_image
from attention.img_proc.landmark_manip import resize_landmarks
from attention.params import *


def crop_faces(image_rgb: np.ndarray, bbox_list: list[dict]) -> list[np.ndarray]:
    '''Takes the coordinates of the faces detected on an image (bbox coordinates) and returns the cropped faces'''
    faces = []
    for bbox in bbox_list:
        x1 = max(bbox["x1"], 0)
        y1 = max(bbox["y1"], 0)
        x2 = max(bbox["x2"], 0)
        y2 = max(bbox["y2"], 0)
        if abs(x1 - x2) * abs(y1 - y2) > 0:
            face = image_rgb[y1:y2, x1:x2]
            faces.append(face)
    return faces


def annotate_bboxes(image_rgb: np.ndarray, bbox_list):
    '''Takes the original image and returns a copy of the image with the drawing of the bboxes'''
    image_annotated, ratio = resize_image(image_rgb, 1080)
    for bbox in bbox_list:
        x1 = int(bbox["x1"] * ratio)
        y1 = int(bbox["y1"] * ratio)
        x2 = int(bbox["x2"] * ratio)
        y2 = int(bbox["y2"] * ratio)
        cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_annotated


def annotate_mesh(face: np.ndarray, mp_landmarks: NormalizedLandmarkList):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks'''
    face_annotated, ratio = resize_image(face, 500)
    drawing_spec = drawing_utils.DrawingSpec(thickness=0, circle_radius=0, color=(0,255,0))
    drawing_utils.draw_landmarks(face_annotated, mp_landmarks, landmark_drawing_spec=drawing_spec)
    return face_annotated


def annotate_landmarks(face: np.ndarray, face_landmarks: list[tuple], landmark_idx: list[int]):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks listed with (x, y) coordinates'''
    face_annotated, ratio = resize_image(face, 500)
    landmark_focus = [face_landmarks[idx] for idx in landmark_idx]
    landmark_focus_resized = resize_landmarks(landmark_focus, ratio)
    for landmark in landmark_focus_resized:
        cv2.circle(face_annotated, landmark, radius=2, color=(0, 255, 0), thickness=-1)
    return face_annotated


def annotate_attention(face: np.ndarray, face_landmarks: list[tuple],
                       prediction_lefteye_direction: str, score_lefteye_direction: float, prediction_righteye_direction: str, score_righteye_direction: float,
                       prediction_lefteye_inclination: str, score_lefteye_inclination: float, prediction_righteye_inclination: str, score_righteye_inclination: float,
                       prediction_head_direction: str, score_head_direction: float,
                       prediction_head_inclination: str, score_head_inclination: float,
                       prediction_attention: str):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks listed with (x, y) coordinates + predictions for iris, head and overall attention'''
    face_annotated, ratio = resize_image(face, 500)
    face_landmarks_resized = resize_landmarks(face_landmarks, ratio)
    h, w = face_annotated.shape[:2]
    eye_height = max(face_landmarks_resized[LEFT_IRIS_CENTER[0], 1], face_landmarks_resized[RIGHT_IRIS_CENTER[0], 1])

    #LEFT EYE DIRECTION
    landmark_idx_lefteye_direction = LEFT_EYE_EDGES  + LEFT_IRIS_CENTER
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_lefteye_direction)
    cv2.putText(face_annotated, f'{prediction_lefteye_direction}',
                (w // 2 + 50, eye_height + 30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_lefteye_direction:.2f}',
                (w // 2 + 50, eye_height + 50),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    #RIGHT EYE DIRECTION
    landmark_idxright_direction = RIGHT_EYE_EDGES  + RIGHT_IRIS_CENTER
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idxright_direction)
    cv2.putText(face_annotated, f'{prediction_righteye_direction}',
                (50, eye_height + 30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_righteye_direction:.2f}',
                (50, eye_height + 50),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    #LEFT EYE INCLINATION
    landmark_idx_lefteye_inclination = LEFT_EYE_INT
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_lefteye_inclination)
    cv2.putText(face_annotated, f'{prediction_lefteye_inclination}',
                (w // 2 + 50, eye_height + 80),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_lefteye_inclination:.2f}',
                (w // 2 + 50, eye_height + 100),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    #RIGHT EYE INCLINATION
    landmark_idxright_inclination = RIGHT_EYE_INT
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idxright_inclination)
    cv2.putText(face_annotated, f'{prediction_righteye_inclination}',
                (50, eye_height + 80),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_righteye_inclination:.2f}',
                (50, eye_height + 100),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    #HEAD DIRECTION
    landmark_idx_nose = NOSE
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_nose)
    cv2.putText(face_annotated, f'{prediction_head_direction}',
                (w // 4 - 70, eye_height + 150),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_head_direction:.2f}',
                (w // 4 - 70, eye_height + 170),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    ##HEAD INCLINATION
    landmark_idx_bottom_lip_forehead = BOTTOM_LIP + FOREHEAD_MIDDLE
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_bottom_lip_forehead)
    cv2.putText(face_annotated, f'{prediction_head_inclination}',
                (w // 4 - 70, eye_height + 220),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))
    cv2.putText(face_annotated, f'{score_head_inclination}',
                (w // 4 - 70, eye_height + 240),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    ##OVERALL ATTENTION
    cv2.putText(face_annotated, f'{prediction_attention}',
                (w // 2 - 70, eye_height + 280),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.9, color = (255,45,0))


    return face_annotated


def annotate_recognition(face: np.ndarray, prediction_name: str, distance: float):
    '''Takes a face image and returns a copy of the image with annotation of the face recognition prediction and corresponding score'''
    face_annotated, ratio = resize_image(face, 500)
    h, w = face_annotated.shape[:2]

    cv2.putText(face_annotated, f'{prediction_name}',
            (w // 2 - 50 , (3 * h) // 4),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color =(0, 255, 0))

    cv2.putText(face_annotated, f'distance: {distance:.2f}',
                (w // 2 - 100 , (3 * h) // 4 + 30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color =(0, 255, 0))
    return face_annotated


def annotate_summary(image_summary: np.ndarray, ratio,
                     bbox_face: dict,
                     attention: bool, attention_driver: str,
                     recognition: bool):
    '''Generates final output with bboxes on all detected faces with indication of their attention and recognition:
    -- attention: green box if attentive, yellow box if inattentive
    -- recognition: thick line if recognized, thin if not recognized'''

    x1 = int(bbox_face["x1"] * ratio)
    y1 = int(bbox_face["y1"] * ratio)
    x2 = int(bbox_face["x2"] * ratio)
    y2 = int(bbox_face["y2"] * ratio)

    color = (0, 255, 0) if attention else (255, 0, 0)
    thickness = 4 if recognition else 2
    cv2.rectangle(image_summary, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image_summary, attention_driver,
                (x1, y1 - 10),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color =(192, 0, 0))
    return image_summary
