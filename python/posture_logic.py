import json
import time

from calibration import derive_metrics, CALIBRATION_PATH
from mediapipe.tasks.python.vision import PoseLandmark

"""
For posture logic, I will be following the guidelines specified in
the work by Piñero-Fuentes et al. (2021), which I have referenced
in calibration.py. In the paper, Table 1 summarises the ranges of
various motion zones which posture quality can be classifed into.
I will use a simplified version of this in this file for now.
"""

# Deviation of metrics until which a posture is considered good
THRESHOLDS = {
    "frontal_neck_flexion_deg": 15,
    "shoulder_alignment_deg": 8,
    "right_arm_abduction_deg": 10,
    "left_arm_abduction_deg": 20,
    "head_tilt_deg": 20,
}

# The landmarks required for each metric to be calculated
VISIBILITY_REQUIREMENTS = {
    "frontal_neck_flexion_deg": [PoseLandmark.NOSE, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
    "shoulder_alignment_deg": [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER],
    "right_arm_abduction_deg": [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW],
    "left_arm_abduction_deg": [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW],
    "head_tilt_deg": [PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR],
}

MIN_VISIBILITY = 0.7    # Required visibility to take metrics into account


# The main class to evaluate posture quality
class PostureEvaluator:

    def __init__(self, calibration_path: str = CALIBRATION_PATH):

        with open(calibration_path, "r") as f:
            calibration_data = json.load(f)

        self.baseline_metrics = calibration_data["derived_metrics"]

    def evaluate(self, pose_landmarks) -> dict:

        live_metrics = derive_metrics(pose_landmarks)

        deviations = {}
        breaches = {}

        for metric, threshold in THRESHOLDS.items():

            if VISIBILITY_REQUIREMENTS.get(metric):

                visible = all(pose_landmarks[landmark].visibility >= MIN_VISIBILITY for landmark in VISIBILITY_REQUIREMENTS.get(metric))

                if not visible:
                    continue    # Skip this metric in the current frame

            deviation = abs(live_metrics[metric] - self.baseline_metrics[metric])
            deviations[metric] = deviation
            breaches[metric] = deviation > threshold

        state = "slouching" if any(breaches.values()) else "good"

        return {
            "state": state,
            "deviations": deviations,
            "breaches": breaches,
        }

