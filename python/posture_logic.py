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

MIN_VISIBILITY = 0.7    # Required visibility to take metrics into account


# The main class to evaluate posture quality
class PostureEvaluator:

    def __init__(self, calibration_path: str = CALIBRATION_PATH):

        with open(calibration_path, "r") as f:
            calibration_data = json.load(f)

        self.baseline_metrics = calibration_data["derived_metrics"]