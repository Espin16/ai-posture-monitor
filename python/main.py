import cv2
import time
from face_detector import FaceDetectorWrapper, visualize
from pose_detector import PoseLandmarkerWrapper, draw_landmarks_on_image
from serial_comm import ArduinoLink

MODEL_PATH = "python/models/blaze_face_short_range.tflite"
frame_ms = 1000/60

def main():
    face_detector = FaceDetectorWrapper(model_path=MODEL_PATH)
    pose_landmarker = PoseLandmarkerWrapper()
    arduino = ArduinoLink(port="COM3")
    capture = cv2.VideoCapture(0)
    timestamp_ms = 0
    last_sent_state = None

    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the webcam feed.")
    time.sleep(2)

    while True:

        ret, frame = capture.read()
        timestamp_ms += frame_ms

        if not ret:
            print("Error: Could not read frame.")
            break

        face_detector_result = face_detector.detect(frame, int(timestamp_ms))
        annotated_image = visualize(frame, face_detector_result)

        pose_landmarker_result = pose_landmarker.detect(frame, int(timestamp_ms))
        annotated_image = draw_landmarks_on_image(annotated_image, pose_landmarker_result)

        current_state = 'g' if face_detector_result.detections else 'a'
        if current_state != last_sent_state:
            arduino.send(current_state)
            last_sent_state = current_state

        if annotated_image is not None:
            cv2.imshow('Webcam Feed', annotated_image)
        else:
            cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    arduino.send('off')
    face_detector.close()
    arduino.close()
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()