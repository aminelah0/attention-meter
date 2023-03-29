import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


def convert_landmarks(face: np.ndarray, mp_landmarks: NormalizedLandmarkList) -> list[tuple]:
    '''Takes the resulting landmarks of a face from the mediapipe face_mesh process of class NormalizedLandmarkList
    and converts them to a list of (x,y) coordinates in the form of a tuple'''
    img_width = face.shape[1]
    img_height = face.shape[0]
    landmark_list = []

    for landmark in mp_landmarks.landmark:
        landmark_list.append((int(landmark.x * img_width), int(landmark.y * img_height)))

    return landmark_list


def resize_landmarks(landmark_list: list[tuple], ratio):
    '''Takes original list of landmarks (x,y tuples) and resize them as per ratio'''
    landmark_list_resized = ((np.array(landmark_list) + 1) * ratio - 1).astype(int)

    return landmark_list_resized


def distance_coord(coord1: tuple, coord2: tuple, axis: str ='x') -> int:
    '''Calculates the distance between two points along the specified axis - if xy calculates the euclidian distance'''
    x = abs(coord1[0] - coord2[0])
    y = abs(coord1[1] - coord2[1])
    if axis == 'x':
        return x
    elif axis =='y':
        return y
    else:
        return np.sqrt(x**2 + y**2)
