import os
import pandas as pd
import numpy as np
import pickle
from attention.img_proc.img_annot import crop_faces, annotate_bboxes, annotate_mesh, annotate_recognition, annotate_attention, annotate_summary
from attention.img_proc.landmark_manip import convert_landmarks
from attention.img_proc.img_split import split_image, reconstruct_coord, bbox_merge
from attention.models.face_models import train_faces, detect_face, find_landmarks, detect_eye_directions, detect_eye_inclinations, detect_head_direction, detect_head_inclination, is_attentive, recognize_face
from attention.utils.img_vid_utils import extract_video_frames, load_image_paths, read_image, save_image, resize_image
from attention.utils.dataframe_utils import process_audience_df
from attention.params import *


def video2frames(period_sec: float, start_sec: float = 0, end_sec: float = None):
    """Loads the specified video in the input_video path and extracts then saves frames periodically:
    -- period_sec: period between 2 frame captures
    -- start_sec: second at which first frame is captured - optional: 0 by default
    -- end_sec: second at which last frame is captured - optional: complete video by default"""

    video_folder_path = os.path.join(DATA_DIRECTORY, "00_inputs", "02_video")
    video_name = os.listdir(video_folder_path)[0]
    video_path = os.path.join(video_folder_path, video_name)
    frames = extract_video_frames(video_path,
                                period_sec=period_sec,
                                start_sec=start_sec, end_sec=end_sec)

    # Saving the frames
    frames_folder_path = os.path.join(DATA_DIRECTORY, "00_inputs", "01_frames")
    for timestamp, frame in frames.items():
        frame_name = video_name.split('.')[0] + f'_ds{int(timestamp * 10):05}'
        save_image(frame, frame_name + '.png', frames_folder_path)

    print('Frames extracted from original video and saved to input frames path')


def rename_frames(period_sec: float):
    """In case photos were taken of the audience (for example every second) instead of a video, the name of the photo needs to contain the timestamp.
    This method takes all the photos within the input_frames directory alphabetically sorted and renames them to include the timestamp:
    -- period_sec: period between 2 frame captures"""

    frames_folder_path = os.path.join(DATA_DIRECTORY, "00_inputs", "01_frames")

    files = sorted([f for f in os.listdir(frames_folder_path) if not f.startswith(".")], key=str.lower)

    for idx, filename in enumerate(files):
        os.rename(os.path.join(frames_folder_path, filename), os.path.join(frames_folder_path, f'frame_ds{int(idx * period_sec * 10):05}.png'))

    print('Original frames renamed to include timestamp value')


def train_recognition():
    '''Loads the known face images from the input_known_faces directory and encodes them into a dictionary
    Saves a pickle in the input directory'''
    known_folder_path = os.path.join(DATA_DIRECTORY, "00_inputs", "99_known_faces")
    known_paths = load_image_paths(known_folder_path)
    known_names = list(known_paths.keys())
    known_faces = [read_image(image_path) for image_path in known_paths.values()]
    known_encodings = train_faces(known_faces, known_names)

    recognition_pickle_path = os.path.join(DATA_DIRECTORY, "00_inputs", "known_faces.pkl")
    with open(recognition_pickle_path, 'wb') as file:
        pickle.dump(known_encodings, file)
        print('Known faces encoding succeeded and pickle file saved')

    return known_encodings



def generate_outputs(save_summary:bool = True, save_detection:bool = True, save_faces:bool = True, save_mesh:bool = True, save_attention:bool = True, save_recognition:bool = True):
    '''Loads the original audience images from the input frames directory and:
    -- detect the faces in the group photo
        --> save_detection parameter to save a copy of the original image with a bbox frame on the faces in the output detection directory
        --> save_faces parameter to save a crop of the detected faces in the output face crops directory
    -- for each face:
        -- generates the face mesh --> save_mesh parameter to save a copy of the cropped face with a drawn mesh in the output face_mesh directory
        -- calculates attention through distances between landmarks and specified ratios --> save attention parameter to save annotated images of the faces with all attention-related calculations and outputs
        -- recognizes the face based on the known faces on which the recognition model has been trained --> save_recognition parameter to save a copy of the face with the name of the person recognized
        -- saves the computed face information into a dataframe
    '''

    frames_folder_path = os.path.join(DATA_DIRECTORY, "00_inputs", "01_frames")
    image_paths = load_image_paths(frames_folder_path)

    #Loading encoded faces pickle for recognition
    recognition_pickle_path = os.path.join(DATA_DIRECTORY, "00_inputs", "known_faces.pkl")
    with open(recognition_pickle_path, 'rb') as file:
        known_encodings = pickle.load(file)

    #Generating output dataframe
    attention_df = pd.DataFrame(columns=DF_COLUMNS)

    # Running the program
    for image_name, image_path in image_paths.items():

        # Loading image
        image = read_image(image_path)
        timestamp = int(image_name.split('_ds')[1]) if '_ds' in image_name else np.nan
        image_summary, ratio_summary = resize_image(image, 1920)
        # Splitting image
        crops = split_image(image, N_SPLIT_W, N_SPLIT_H, OVERLAP_W, OVERLAP_H)
        # Generating bboxes for each crop
        bbox_crop_list = []
        bbox_crop_list_absolute = []
        for crop in crops:
            coord_set = detect_face(crop.image)
            bbox_crop_list.append(coord_set)
            coord_set_absolute = reconstruct_coord(crop, coord_set)
            bbox_crop_list_absolute.append(coord_set_absolute)
        # Eliminating duplicates bboxes
        bbox_list = bbox_merge(bbox_crop_list_absolute, intersect_threshold=0.6)

        if save_detection:
            # Drawing the unique bboxes on the original image
            image_output = annotate_bboxes(image, bbox_list)
            # Saving the image with its bboxes
            detection_path = os.path.join(DATA_DIRECTORY, '99_outputs', '01_detection')
            save_image(image_output, image_name + '.png', detection_path)


        # Generating face crops
        faces = crop_faces(image, bbox_list)

        if save_faces:
            # Saving face crops
            face_path = os.path.join(DATA_DIRECTORY, '99_outputs', '02_face_crops')
            for face_idx, face in enumerate(faces):
                face_name = image_name + f'_{face_idx}'
                save_image(face, face_name + '.png', face_path)


        # Generating eye and iris landmarks
        for face_idx, face in enumerate(faces):

            face_name = image_name + f'_{face_idx}'
            mp_landmarks = find_landmarks(face)
            if mp_landmarks:                # Only run attention/ recognition if it detects a face

                # Converting the Mediapipe landmark to a standard system of coordinates
                landmark_list = convert_landmarks(face, mp_landmarks)

                if save_mesh:
                    # Drawing the face mesh on the face
                    face_mesh = annotate_mesh(face, mp_landmarks)
                    # Saving face with complete mesh
                    mesh_path = os.path.join(DATA_DIRECTORY, '99_outputs', '03_face_mesh')
                    save_image(face_mesh, face_name + '.png', mesh_path)


                # Detecting eye direction and attention
                eye_directions = detect_eye_directions(landmark_list, extreme_threshold = 0.63, detailed_threshold_main = 0.6, detailed_threshold_comp = 0.45)
                eye_inclinations = detect_eye_inclinations(landmark_list, threshold = 0.23)
                head_direction = detect_head_direction(landmark_list, left_threshold = 0.35, right_threshold = 0.35)
                head_inclination = detect_head_inclination(landmark_list, down_threshold = 2.3)
                attention, attention_driver = is_attentive(eye_directions, eye_inclinations, head_direction, head_inclination)

                prediction_lefteye_direction, score_lefteye_direction = eye_directions['left']
                prediction_righteye_direction, score_righteye_direction = eye_directions['right']
                prediction_lefteye_inclination, score_lefteye_inclination = eye_inclinations['left']
                prediction_righteye_inclination, score_righteye_inclination = eye_inclinations['right']
                prediction_head_direction, score_head_direction = head_direction
                prediction_head_inclination, score_head_inclination = head_inclination
                prediction_attention = 'attentive' if attention else 'inattentive'

                if save_attention:
                    # Drawing iris landmarks + annotating attention results on original image
                    face_attention = annotate_attention(face, landmark_list,
                                                            prediction_lefteye_direction, score_lefteye_direction,
                                                            prediction_righteye_direction, score_righteye_direction,
                                                            prediction_lefteye_inclination, score_lefteye_inclination,
                                                            prediction_righteye_inclination, score_righteye_inclination,
                                                            prediction_head_direction, score_head_direction,
                                                            prediction_head_inclination, score_head_inclination,
                                                            prediction_attention)
                    # Saving attention image output
                    attention_path = os.path.join(DATA_DIRECTORY, '99_outputs', '04_attention')
                    save_image(face_attention, face_name + '.png', attention_path)


                # Recognizing a face
                face_prediction = recognize_face(face, known_encodings)
                prediction_recognition, distance_recognition = face_prediction

                if save_recognition:
                    # Annotating name and distance on the face image
                    face_recognition = annotate_recognition(face, prediction_recognition, distance_recognition)
                    # Saving recognition image output
                    recognition_path = os.path.join(DATA_DIRECTORY, '99_outputs', '05_recognition')
                    save_image(face_recognition, face_name + '.png', recognition_path)


                # Generating summary image
                if save_summary:
                    # Annotating with key info (bbox, attentiveness, recognition)
                    recognition = False if pd.isna(prediction_recognition) else True
                    bbox_face = bbox_list[face_idx]
                    image_summary = annotate_summary(image_summary, ratio_summary,
                                                    bbox_face,
                                                    attention, attention_driver,
                                                    recognition)


                # Saving data in the dataframe
                attention_df.loc[len(attention_df)] = [image_name,
                                                        timestamp,
                                                        face_idx,
                                                        prediction_recognition, distance_recognition,
                                                        attention,
                                                        prediction_lefteye_direction, score_lefteye_direction,
                                                        prediction_righteye_direction, score_righteye_direction,
                                                        prediction_head_direction, score_head_direction,
                                                        prediction_head_inclination, score_head_inclination]


        # Saving summary image once all faces of the image are processed
        if save_summary:
            summary_path = os.path.join(DATA_DIRECTORY, '99_outputs', '00_summary')
            save_image(image_summary, image_name + '.png', summary_path)


    # Saving the dataframe
    output_path = os.path.join(DATA_DIRECTORY, '99_outputs', 'attention_output.csv')
    attention_df.to_csv(output_path, index=False)

    print('All frames have been processed and output dataframe generated')

    return attention_df


if __name__ == '__main__':
    # video2frames(period_sec=1)
    # rename_frames(period_sec=1)
    train_recognition()
    generate_outputs()
    pass
