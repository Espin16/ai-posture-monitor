import cv2
import json
import time
import math
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmark
from pose_detector import PoseLandmarkerWrapper

frame_ms = 1000/60

COUNTDOWN_SECONDS = 3
CALIBRATION_PATH = "data/calibration.json"


# Formatting for JSON file
def _landmark_to_dict(landmark):

    return {"x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,}


# Note that for the following angle calculations, we only require the .x and .y attributes

def _angle_from_vertical(p1, p2):

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    return math.degrees(math.atan2(dx,dy))

def _angle_from_horizontal(p1, p2):

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    return math.degrees(math.atan2(dy,dx))


# A helper class to obtain points derived from existing landmarks

class _Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def derive_metrics(landmarks):

    """
    Choice of parameters inspired by Piñero-Fuentes et al. (2021),
    'A Deep-Learning Based Posture Detection System for Preventing
    Telework-Related Musculoskeletal Disorders.' Their neck keypoint
    is simulated by the shoulder midpoint. This is necessary by the
    lack of relevant keypoint from MediaPipe's landmarker.
    """

    nose = landmarks[PoseLandmark.NOSE]
    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[PoseLandmark.RIGHT_ELBOW]
    left_ear = landmarks[PoseLandmark.LEFT_EAR]
    right_ear = landmarks[PoseLandmark.RIGHT_EAR]

    shoulder_midpoint = _Point((left_shoulder.x + right_shoulder.x) / 2,
                               (left_shoulder.y + right_shoulder.y) / 2,)

    ear_midpoint = _Point((left_ear.x + right_ear.x) / 2,
                          (left_ear.y + right_ear.y) / 2,)

    return {"frontal_neck_flexion_deg": _angle_from_vertical(nose, shoulder_midpoint),    # Frontal, ie parallel to the plane of view
            "shoulder_alignment_deg": _angle_from_horizontal(right_shoulder, left_shoulder),
            "right_arm_abduction_deg": _angle_from_vertical(right_shoulder, right_elbow),
            "left_arm_abduction_deg": _angle_from_vertical(left_shoulder, left_elbow),
            "head_tilt_deg": _angle_from_horizontal(right_ear, left_ear),} # Head tilt/roll isolated


def main():

    pose_landmarker = PoseLandmarkerWrapper()
    capture = cv2.VideoCapture(0)
    timestamp_ms = 0

    if not capture.isOpened():
        print("Error: Could not open camera.")
        return

    print("Sit normally to begin calibration.")

    ready = False
    start_time = None
    captured = False
    baseline_landmarks = None


    while True:

        ret, frame = capture.read()
        timestamp_ms += frame_ms

        if not ret:
            print("Error: Could not read frame.")
            break

        display_frame = frame.copy()


        # Wait for user to begin calibration
        if not ready:

            cv2.putText(display_frame, "Press SPACE to begin calibration.", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord(' '):

                ready = True
                start_time = time.time()


        else:

            result = pose_landmarker.detect(frame, int(timestamp_ms))
            remaining = max(0, COUNTDOWN_SECONDS - (time.time() - start_time))

            cv2.putText(display_frame, f"Capturing in {remaining:.1f}s...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if remaining <= 0 and not captured:

                if result.pose_landmarks:
                    baseline_landmarks = result.pose_landmarks[0]
                    captured = True
                    print("Baseline captured.")

                else:
                    print("No pose detected at capture - restarting countdown.")
                    start_time = time.time()

        cv2.imshow("Calibration", display_frame)

        if captured or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    pose_landmarker.close()
    capture.release()
    cv2.destroyAllWindows()

    if baseline_landmarks is None:
        print("No baseline captured. Exiting without saving.")
        return

    raw_landmarks = {
        "nose": _landmark_to_dict(baseline_landmarks[PoseLandmark.NOSE]),
        "left_shoulder": _landmark_to_dict(baseline_landmarks[PoseLandmark.LEFT_SHOULDER]),
        "right_shoulder": _landmark_to_dict(baseline_landmarks[PoseLandmark.RIGHT_SHOULDER]),
        "left_elbow": _landmark_to_dict(baseline_landmarks[PoseLandmark.LEFT_ELBOW]),
        "right_elbow": _landmark_to_dict(baseline_landmarks[PoseLandmark.RIGHT_ELBOW]),
        "left_ear": _landmark_to_dict(baseline_landmarks[PoseLandmark.LEFT_EAR]),
        "right_ear": _landmark_to_dict(baseline_landmarks[PoseLandmark.RIGHT_EAR]),
    }

    calibration_data = {
        "raw_landmarks": raw_landmarks,
        "derived_metrics": derive_metrics(baseline_landmarks),
        "captured_at": time.strftime("%d-%m-%Y %H:%M:%S"),
    }

    os.makedirs("data", exist_ok = True)
    with open(CALIBRATION_PATH, 'w') as f:
        json.dump(calibration_data, f, indent = 2)

    print(f"Calibration complete.")



if __name__ == "__main__":
    main()