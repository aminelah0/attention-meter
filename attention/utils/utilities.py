import os
import numpy as np
import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import plotly.express as px
import nbformat


def load_image_paths(folder_path: str) -> dict[str]:
    '''Retrieves all images in the specified folder and returns a dict in the form:
    -- file_name: name of the file without extension
    -- file_path'''
    image_paths = dict()
    image_extension = ['jpeg', 'jpg', 'png']
    for file in sorted(os.listdir(folder_path)):
        if file.find('.') != -1:
            file_extension = file.split('.')[1]
            if file_extension in image_extension:
                file_path = os.path.join(folder_path,file)
                file_name = file.split('.')[0]
                image_paths[file_name] = file_path
    return image_paths


def extract_video_frames(video_path: str, period_sec: float, start_sec: float = 0, end_sec: float = None) -> dict[float: np.ndarray]:
    '''Loads the video, extracts the frames at a periodic rate and stores them in a dictionary {timestamp_in_seconds: frame}
    -- period_sec: period between 2 frame captures
    -- start_sec: second at which first frame is captured - optional: 0 by default
    -- end_sec: second at which last frame is captured - optional: complete video by default'''
    frames = dict()
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames in video
    fps = video.get(cv2.CAP_PROP_FPS) # number of frames per second
    duration = int((frame_count/fps - 1) * 1000) # duration of the video in ms

    start = start_sec * 1_000
    end = end_sec * 1_000 if end_sec else duration
    period = period_sec * 1_000

    assert end > start, 'Specified start for the video capture posterior to the end'
    for period in range(start, end, period_sec * 1_000):
        video.set(cv2.CAP_PROP_POS_MSEC, period)
        frame = video.read()[1]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = round(period / 1000, 1)
        frames[timestamp] = image_rgb

    return frames



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


def reframe_dataframe(df:pd.DataFrame, video_start:int)-> pd.DataFrame:
    '''Reframes the dataframe to have axis in seconds and percentage'''
    df = df.reset_index()
    df["seconds"] = df['timestamp']/10 - video_start
    df["percentage"] = df['attentive']*100
    return df

def plot_attention_curve(df:pd.DataFrame, video_start:int)-> plotly.fig:
    df = df.groupby('timestamp')[['attentive']].mean()
    df = df.rolling(window=10).mean().dropna()
    df = reframe_dataframe(df, video_start)
    fig = px.line(df, x="seconds", y="percentage", labels = dict(percentage = "Attentiveness (in %)", seconds = "Time (in seconds)"))
    fig.update_layout(yaxis_range=[0,100])
    fig.update_layout(xaxis_range=[0,150])
    return fig
