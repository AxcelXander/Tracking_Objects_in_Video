# import libraries
import cv2
import numpy as np

#Repeatability
np.random.seed(0)

VfileName = 'video.mp4'
Height = 480
Width = 640


# load video frames from video file

def get_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames

