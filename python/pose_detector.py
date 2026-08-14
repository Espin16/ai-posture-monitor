import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from typing import Tuple, Union

MODEL_PATH = 'python/models/pose_landmarker_lite.task'

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255) 

frame_ms = 1000/60  # Approximate frame duration for a 60 FPS stream

### VISUALIZATION ###

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: vision.PoseLandmarkerResult) -> np.ndarray: # type: ignore

    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)


    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image = annotated_image,
            landmark_list = pose_landmarks,
            connections = vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec = pose_landmark_style,
            connection_drawing_spec = pose_connection_style)

    return annotated_image


### POSE DETECTION ###

# Initialise objects

class PoseLandmarkerWrapper:

    def __init__(self, model_path: str = MODEL_PATH, min_pose_detection_confidence: float = 0.8, min_pose_presence_confidence: float = 0.7, min_tracking_confidence: float = 0.8):

        base_options = mp.tasks.BaseOptions(model_asset_path = model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(base_options = base_options,
                                                        running_mode = vision.RunningMode.VIDEO,
                                                        min_pose_detection_confidence = min_pose_detection_confidence,
                                                        min_pose_presence_confidence = min_pose_presence_confidence,
                                                        min_tracking_confidence = min_tracking_confidence)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms:int) -> vision.PoseLandmarkerResult:           #type: ignore
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
