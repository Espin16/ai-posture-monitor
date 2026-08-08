import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
# from mediapipe.tasks.python.vision import drawing_utils
# from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from typing import Tuple, Union

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255) 

frame_ms = 1000/60  # Approximate frame duration for a 60 FPS stream

### VISUALIZATION ###


# Converting valid normalized coordinates from the result to pixel coordinates
def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:

    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0) and (value < 1)

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        #TODO: Draw coordinates even if it;s outside of the image bounds
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image: np.ndarray, detection_result: vision.FaceDetectorResult) -> np.ndarray: # type: ignore
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:

        #Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        #Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, radius, color, thickness)

        #Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score,2)
        result_text = '{} ({})'.format(category_name, probability)
        text_location = (MARGIN + bbox.origin_x, MARGIN + bbox.origin_y + ROW_SIZE)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return annotated_image



### FACE DETECTION ###

# Initialise objects

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path='python/models/blaze_face_short_range.tflite'),
                              running_mode=VisionRunningMode.VIDEO,
                              min_detection_confidence=0.65)

with FaceDetector.create_from_options(options) as detector:

    def main():

        capture = cv2.VideoCapture(0)
        timestamp_ms = 0
        frame_counter = 0


        if not capture.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit the webcam feed.")

        time.sleep(2)

        while True:
            ret, frame = capture.read()
            timestamp_ms += frame_ms
            frame_counter += 1

            if not ret:
                print("Error: Could not read frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_detector_result = detector.detect_for_video(mp_image, int(timestamp_ms))

            image_copy = np.copy(mp_image.numpy_view())

            annotated_image = visualize(image_copy, face_detector_result)
            


            if annotated_image is not None:
                cv2.imshow('Webcam Feed', annotated_image)
            else:
                cv2.imshow('Webcam Feed', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    main()