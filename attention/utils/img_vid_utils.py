import os
import numpy as np
import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


def load_image_paths(folder_path: str) -> dict[str]:
    '''Retrieves all images in the specified folder and returns a dict in the form:
    -- file_name: name of the file without extension
    -- file_path'''
    image_paths = dict()
    image_extension = ['jpeg', 'jpg', 'png']
    for file in sorted(os.listdir(folder_path)):
        if file.find('.') != -1:
            file_extension = file.split('.')[1].lower()
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
    for period in range(start, end, int(period_sec * 1_000)):
        video.set(cv2.CAP_PROP_POS_MSEC, period)
        frame = video.read()[1]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = round(period / 1000, 1)
        frames[timestamp] = image_rgb

    return frames


def crop_split_video(video_path: str, output_path: str,
                     x1: int = 0, x2: int = None, y1: int = 0, y2: int = None,
                     video_start_sec: float = 0, video_end_sec: float = None,
                     period_sec: float = None):
    '''Loads a video and:
    -- reads it between video_start_sec and video_end_sec (if specified, otherwise from start to end)
    -- crops it in the x1,x2,y1,y2 area (if specified, otherwise no crop)
    -- insert a white frame in between each period_sec video section if specified'''

    # Open the video
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split(os.sep)[-1].split('.')[0]

    # Initialize frame counter
    cnt = 0

    # Some characteristics from the original video
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # PARAMETERS
    x2 = x2 if x2 else w_frame
    y2 = y2 if y2 else h_frame
    video_end_sec = video_end_sec if video_end_sec else frames // fps

    # Params converted to metrics
    width = x2 - x1
    height = y2 - y1
    frame_start = int(video_start_sec * fps)
    frame_end = int(video_end_sec * fps)
    frame_total = frame_end - frame_start
    frame_period = int(period_sec * fps)

    # output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, f'{video_name}.mp4'), fourcc, fps, (width, height))

    # Now we start
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * video_start_sec)) #Start the video at a certain frame

    while(cap.isOpened()):
        ret, frame = cap.read()

        cnt += 1 # Counting frames

        # End the capture when specified frame reached
        if cnt < frame_total:

            # Avoid problems when video finish
            if ret==True:

                # Croping the frame
                crop_frame = frame[y1:y2, x1:x2]

                # Percentage
                xx = cnt *100/ frame_total
                print(int(xx),'%')

                # Here you save all the video
                out.write(crop_frame)

                # Insert white frame between periods (if period sepcified)
                if period_sec:
                    if cnt % frame_period == 0:
                        white_frame = np.ones(crop_frame.shape, dtype=crop_frame.dtype) * 255
                        white_frame_annot = cv2.putText(white_frame, f'{cnt // frame_period}',
                        (width // 2, height // 2),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 2, color = (0,0,0))
                        for _ in range(2 * int(fps)):
                            out.write(white_frame_annot)


                # # Ssee the video in real time
                # cv2.imshow('frame',frame)
                # cv2.imshow('croped',crop_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
    Returns a tuple containing the resized_image as well as the resize_ratio
    Input width = 0 to keep original size'''
    image_copy = image.copy()
    if width == 0:
        return image_copy, 1
    else:
        (h,w) = image.shape[:2]
        ratio = width/w
        height = int(h * ratio)
        image_resized = cv2.resize(image_copy, (width, height))
        return image_resized, ratio
