import os
import numpy as np
import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark


def load_image_paths(folder_path: str) -> dict[str]:
    '''Retrieves all images in the specified folder and returns a dict in the form:
    -- file_name: name of the file without extension
    -- file_path'''
    image_paths = dict()
    image_extension = ['jpeg', 'jpg', 'png']
    for file in os.listdir(folder_path):
        if file.find('.') != -1:
            file_extension = file.split('.')[1]
            if file_extension in image_extension:
                file_path = os.path.join(folder_path,file)
                file_name = file.split('.')[0]
                image_paths[file_name] = file_path
    return image_paths


def convert_landmarks(face: np.ndarray, mp_landmarks: NormalizedLandmarkList) -> list[tuple]:
    '''Takes the resulting landmarks of a face from the face_mesh process
    and converts them to a list of (x,y) coordinates in the form of a tuple'''
    img_width = face.shape[1]
    img_height = face.shape[0]
    landmark_list = []

    for landmark in mp_landmarks.landmark:
        landmark_list.append((int(landmark.x * img_width), int(landmark.y * img_height)))

    return landmark_list


def resize_landmarks(landmark_list: list[tuple], ratio):
    '''Takes original list of landmarks (x,y tuples) and resize them as per ratio'''
    landmark_list_resized = (np.array(landmark_list) * ratio).astype(int)

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
