import numpy as np
import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark


def convert_landmarks(face: np.ndarray, mp_landmarks: NormalizedLandmarkList) -> list[tuple]:
    '''Takes the resulting landmarks of a face from the face_mesh process
    and converts them to a list of (x,y) coordinates in the form of a tuple'''
    img_width = face.shape[1]
    img_height = face.shape[0]
    landmark_list = []

    for landmark in mp_landmarks.landmark:
        landmark_list.append((int(landmark.x * img_width), int(landmark.y * img_height)))

    return landmark_list
