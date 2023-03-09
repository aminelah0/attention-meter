# Import libraries
from mediapipe.python.solutions import face_detection, face_mesh
import numpy as np
import mediapipe.framework.formats.landmark_pb2 as mp_landmark


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


def find_landmarks(face: np.ndarray) -> mp_landmark.NormalizedLandmarkList:
    '''Takes the image of a face and return all landmarks of the face. Returns None if no landmark detected'''
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        min_detection_confidence=0.5,
        refine_landmarks=True,
        min_tracking_confidence=0.5) as face_meshe:

        results = face_meshe.process(face)

        face_landmarks = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

        return face_landmarks
