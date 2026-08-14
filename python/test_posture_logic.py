import cv2
import time

from pose_detector import PoseLandmarkerWrapper
from posture_logic import PostureEvaluator

frame_ms = 1000 / 60


def main():
    pose_landmarker = PoseLandmarkerWrapper()
    evaluator = PostureEvaluator()  # loads data/calibration.json
    capture = cv2.VideoCapture(0)
    timestamp_ms = 0

    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = capture.read()
        timestamp_ms += frame_ms
        if not ret:
            print("Error: Could not read frame.")
            break

        result = pose_landmarker.detect(frame, int(timestamp_ms))

        if result.pose_landmarks:
            evaluation = evaluator.evaluate(result.pose_landmarks[0])

            # Print a compact one-line status each frame
            deviation_str = ", ".join(
                f"{k}={v:.1f}{'*' if evaluation['breaches'].get(k) else ''}"
                for k, v in evaluation["deviations"].items()
            )
            print(f"[{evaluation['state'].upper():9}] {deviation_str}")

            color = (0, 200, 0) if evaluation["state"] == "good" else (0, 0, 255)
            cv2.putText(frame, evaluation["state"], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "no pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Posture Logic Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose_landmarker.close()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()