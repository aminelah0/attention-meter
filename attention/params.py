import os


#### DIRECTORY STRUCTURE ####
PARAMS_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.join(PARAMS_DIRECTORY, os.pardir, "attention_data")


#### LANDMARKS INDICES - MEDIAPIPE FACE_MESH ####
NOSE = [1]

LEFT_EYE_EDGES = [362, 263]
RIGHT_EYE_EDGES = [33, 133]

RIGHT_EYE_INT = [159, 145]
LEFT_EYE_INT = [386, 374]

LEFT_IRIS_CONTOUR = [474, 475, 476, 477]
RIGHT_IRIS_CONTOUR = [469, 470, 471, 472]

LEFT_IRIS_CENTER = [473]
RIGHT_IRIS_CENTER = [468]

FOREHEAD_MIDDLE = [9]
BOTTOM_LIP = [17]


#### ZOOM ON ORIGINAL IMAGE TO ENHANCE DETECTION - see split_image method ####
N_SPLIT_W = 6
N_SPLIT_H = 3
OVERLAP_W = 0.05
OVERLAP_H = 0.05


#### ATTENTION THRESHOLDS - see face_models module ####
# specified directly in the generate_output method within main_local module


#### OUTPUT DATAFRAME SKELELTON ####
DF_COLUMNS = ['frame',
    'timestamp',
    'face_idx',
    'recognition_prediction',
    'recognition_distance',
    'attentive',
    'left_prediction',
    'left_score',
    'right_prediction',
    'right_score',
    'head_direction_prediction',
    'head_direction_score',
    'head_inclination_prediction',
    'head_inclination_score']
