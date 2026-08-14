import cv2
import time
from face_detector import FaceDetectorWrapper, visualize
from pose_detector import PoseLandmarkerWrapper, draw_landmarks_on_image
from posture_logic import PostureEvaluator
from serial_comm import ArduinoLink

MODEL_PATH = "python/models/blaze_face_short_range.tflite"
frame_ms = 1000/60

def main():

    # Initialise scanning and processing objects
    face_detector = FaceDetectorWrapper(model_path=MODEL_PATH)
    pose_landmarker = PoseLandmarkerWrapper()
    posture_evaluator = PostureEvaluator()

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


        # Show bounding box and pose landmarks
        face_detector_result = face_detector.detect(frame, int(timestamp_ms))
        annotated_image = visualize(frame, face_detector_result)

        pose_landmarker_result = pose_landmarker.detect(frame, int(timestamp_ms))
        annotated_image = draw_landmarks_on_image(annotated_image, pose_landmarker_result)


        # Evaluate given landmarks if available
        if pose_landmarker_result.pose_landmarks:
            evaluation = posture_evaluator.evaluate(pose_landmarks=pose_landmarker_result.pose_landmarks[0])
            posture_state = evaluation["state"]

            color = (0, 200, 0) if posture_state == "good" else (0, 0, 255)
            cv2.putText(annotated_image, posture_state, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            posture_state = None


        # Encode posture state to send to Arduino
        if not face_detector_result.detections or not pose_landmarker_result:
            current_state = 'a'
        elif posture_state == "slouching":
            current_state = 's'
        else:
            current_state = 'g'


        # Send to Arduino when change is detected
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
    pose_landmarker.close()
    arduino.close()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()