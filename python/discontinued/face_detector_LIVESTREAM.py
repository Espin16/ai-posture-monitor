import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
# from mediapipe.tasks.python.vision import drawing_utils
# from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 255) 

frame_ms = 1000/60  # Approximate frame duration for a 60 FPS stream

### VISUALIZATION ###

def visualize(image: np.ndarray, detection_result: vision.FaceDetectorResult) -> np.ndarray:
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
            pass

        #Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score,2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + bbox.origin_y + ROW_SIZE)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return annotated_image



### FACE DETECTION ###

# Initialise objects

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print('face detector result: {}'.format(result))
    return result

options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path='python/models/blaze_face_short_range.tflite'),
                              running_mode=VisionRunningMode.LIVE_STREAM,
                              min_detection_confidence=0.5,
                              result_callback=print_result);

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
            detector.detect_async(mp_image, int(timestamp_ms))

            # image_copy = np.copy(mp_image.numpy_view())

            # annotated_image = visualize(image_copy, detector_result)
            # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            cv2.imshow('Webcam Feed', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


    main()