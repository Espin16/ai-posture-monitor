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

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                running_mode=VisionRunningMode.VIDEO,
                                min_pose_detection_confidence=0.7,
                                min_pose_presence_confidence=0.7,
                                min_tracking_confidence=0.7)



class PoseLandmarkerWrapper:

    def __init__(self, model_path: str = MODEL_PATH, min_pose_detection_confidence: float = 0.7, min_pose_presence_confidence: float = 0.7, min_tracking_confidence: float = 0.7):

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






# with PoseLandmarker.create_from_options(options) as landmarker:

#     def main():

#         capture = cv2.VideoCapture(0)
#         timestamp_ms = 0
#         frame_counter = 0


#         if not capture.isOpened():
#             print("Error: Could not open webcam.")
#             return

#         print("Press 'q' to quit the webcam feed.")

#         time.sleep(2)

#         while True:
#             ret, frame = capture.read()
#             timestamp_ms += frame_ms
#             frame_counter += 1

#             if not ret:
#                 print("Error: Could not read frame.")
#                 break

#             # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#             pose_landmark_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

#             image_copy = np.copy(mp_image.numpy_view())

#             annotated_image = draw_landmarks_on_image(image_copy, pose_landmark_result)

#             # if pose_landmark_result.segmentation_masks:
#             #     segmentation_mask = pose_landmark_result.segmentation_masks[0].numpy_view()
#             #     segmentation_mask = np.squeeze(segmentation_mask)

#             #     visualized_mask = (segmentation_mask * 255).astype(np.uint8)
#             #     visualized_mask = np.stack([visualized_mask]*3, axis=-1)

#             if annotated_image is not None:
#                 cv2.imshow('Webcam Feed', annotated_image)
#             else:
#                 cv2.imshow('Webcam Feed', frame)




#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         capture.release()
#         cv2.destroyAllWindows()

#     main()

    