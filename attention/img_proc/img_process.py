import cv2
import numpy as np
import mediapipe.framework.formats.landmark_pb2 as mp_landmark
from mediapipe.python.solutions import drawing_utils


def cv2_process(image_path: str) -> np.ndarray:
    '''Creates an image array (RGB) from an image path'''

    image_array = cv2.imread(image_path)

    #Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    return image_rgb


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


def output_image_bboxes(image_rgb: np.ndarray, bbox_list):
    '''Takes the original image and returns a copy of the image with the drawing of the bboxes'''
    image_annoted = image_rgb.copy()
    for bbox in bbox_list:
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]
        cv2.rectangle(image_annoted, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_annoted


def output_face_landmarks(face: np.ndarray, landmarks: mp_landmark.NormalizedLandmarkList):
    '''Takes a face image and returns a copy of the image with the drawing of the landmarks'''
    image_annoted = face.copy()
    drawing_utils.draw_landmarks(image_annoted, landmarks)
    return image_annoted
