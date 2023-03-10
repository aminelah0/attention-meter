import os
import cv2
import numpy as np
import mediapipe.framework.formats.landmark_pb2 as mp_landmark
from mediapipe.python.solutions import drawing_utils
from attention.params import *
from attention.utils.utilities import resize_landmarks


def read_image(image_path: str) -> np.ndarray:
    '''Creates an image array (RGB) from an image path'''
    image_array = cv2.imread(image_path)
    #Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_image(image_rgb: np.ndarray, image_name: str, output_path: str) -> np.ndarray:
    '''Saves an image in BGR in the specified output path'''
    assert image_name.find('.') != -1, "Image name requires image extension"

    image_rgb_copy = image_rgb.copy()
    #Convert the RGB image to BGR
    image_bgr = cv2.cvtColor(image_rgb_copy, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(os.path.join(output_path, image_name), image_bgr)


def resize_image(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    '''Takes an image and resizes it as per the desired width, keeping the width/ height ratio of original image
    Returns a tuple containing the resized_image as well as the resize_ratio'''
    image_copy = image.copy()
    (h,w) = image.shape[:2]
    ratio = width/w
    height = int(h * ratio)
    image_resized = cv2.resize(image_copy, (width, height))
    return image_resized, ratio


def crop_faces(image_rgb: np.ndarray, bbox_list: list[dict]) -> list[np.ndarray]:
    '''Takes the coordinates of the faces detected on an image (bbox coordinates) and returns the cropped faces'''
    faces = []
    for bbox in bbox_list:
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]
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


def annotate_mesh(face: np.ndarray, mp_landmarks: mp_landmark.NormalizedLandmarkList):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks'''
    face_annotated, ratio = resize_image(face, 500)
    drawing_spec = drawing_utils.DrawingSpec(thickness=0, circle_radius=0, color=(0,255,0))
    drawing_utils.draw_landmarks(face_annotated, mp_landmarks, landmark_drawing_spec=drawing_spec)
    return face_annotated


def annotate_landmarks(face: np.ndarray, face_landmarks: list[tuple], landmark_idx: list[int]):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks listed with (x, y) coordinates'''
    face_annotated, ratio = resize_image(face, 500)
    landmark_focus = [face_landmarks[idx] for idx in landmark_idx]
    for landmark in landmark_focus:
        landmark_resized = (int(landmark[0] * ratio), int(landmark[1] * ratio))
        cv2.circle(face_annotated, landmark_resized, radius=2, color=(0, 255, 0), thickness=-1)
    return face_annotated


def annotate_iris_attention(face: np.ndarray, face_landmarks: list[tuple], prediction_left: str, score_left: float, prediction_right: str, score_right: float, prediction_attention: str):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks listed with (x, y) coordinates'''
    face_annotated, ratio = resize_image(face, 500)
    face_landmarks_resized = resize_landmarks(face_landmarks, ratio)
    h, w = face_annotated.shape[:2]
    eye_height = face_landmarks_resized[LEFT_IRIS_CENTER[0], 1]
    print(eye_height)

    #LEFT EYE PREDICTION
    landmark_idx_left = LEFT_EYE_EDGES  + LEFT_IRIS_CENTER
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_left)
    cv2.putText(face_annotated, f'{prediction_left}',
                (w // 2 + 50, eye_height + 50),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_left:.2f}',
                (w // 2 + 50, eye_height + 70),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    #RIGHT EYE PREDICTION
    landmark_idx_right = RIGHT_EYE_EDGES  + RIGHT_IRIS_CENTER
    face_annotated = annotate_landmarks(face_annotated, face_landmarks_resized, landmark_idx_right)
    cv2.putText(face_annotated, f'{prediction_right}',
                (50, eye_height + 50),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,225,0))
    cv2.putText(face_annotated, f'{score_right:.2f}',
                (50, eye_height + 70),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))

    ##OVERALL ATTENTION
    cv2.putText(face_annotated, f'{prediction_attention}',
                (w // 2 - 70, eye_height + 100),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color = (0,255,0))


    return face_annotated


def annotate_recognition(face: np.ndarray, prediction_name: str, distance: float):
    '''Takes a face image and returns a copy of the image with annotation of the face recognition prediction and corresponding score'''
    face_annotated, ratio = resize_image(face, 500)
    h, w = face_annotated.shape[:2]

    cv2.putText(face_annotated, f'{prediction_name}',
            (w // 2 - 50 , (3 * w) // 4),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color =(0, 255, 0))

    cv2.putText(face_annotated, f'distance: {distance:.2f}',
                (w // 2 - 100 , (3 * w) // 4 + 30),
                fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.7, color =(0, 255, 0))
    return face_annotated
