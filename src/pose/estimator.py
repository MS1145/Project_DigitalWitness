"""
Pose estimation using MediaPipe for Digital Witness.

Wraps MediaPipe's pose solution to extract 33 body landmarks per frame.
We focus on upper body landmarks (shoulders, elbows, wrists, hips)
which are most relevant for detecting shopping behaviors.
"""
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ..config import POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE


@dataclass
class PoseLandmark:
    """Single body landmark with 3D position and visibility."""
    x: float           # Normalized x (0-1, left to right)
    y: float           # Normalized y (0-1, top to bottom)
    z: float           # Depth relative to hip midpoint
    visibility: float  # Landmark visibility confidence


@dataclass
class PoseResult:
    """Pose estimation output for a single video frame."""
    frame_number: int
    timestamp: float                                  # Frame time in seconds
    landmarks: Optional[Dict[str, PoseLandmark]]      # Named landmarks dict
    raw_landmarks: Optional[Any]                      # MediaPipe native format


# MediaPipe landmark indices we use for behavior analysis
# Full pose has 33 landmarks; we extract only upper body for efficiency
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",        # Key for hand tracking
    16: "right_wrist",       # Key for hand tracking
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",          # Body center reference
    24: "right_hip",         # Body center reference
}


class PoseEstimator:
    """
    MediaPipe pose estimation wrapper with context manager support.

    Detects human body pose in video frames and extracts landmarks
    for downstream behavior classification.
    """

    def __init__(
        self,
        min_detection_confidence: float = POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = POSE_MIN_TRACKING_CONFIDENCE,
        static_image_mode: bool = False
    ):
        """
        Initialize MediaPipe Pose solution.

        static_image_mode=False enables tracking optimization for video,
        where pose from previous frame helps locate pose in current frame.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float
    ) -> PoseResult:
        """
        Run pose detection on a single frame.

        Returns PoseResult with landmarks=None if no person detected.
        """
        # MediaPipe expects RGB; OpenCV provides BGR
        rgb_frame = frame[:, :, ::-1]

        results = self.pose.process(rgb_frame)

        # Extract landmarks if detected
        landmarks = None
        raw_landmarks = None

        if results.pose_landmarks:
            raw_landmarks = results.pose_landmarks
            landmarks = {}

            for idx, name in LANDMARK_NAMES.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = PoseLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                )

        return PoseResult(
            frame_number=frame_number,
            timestamp=frame_number / fps,
            landmarks=landmarks,
            raw_landmarks=raw_landmarks
        )

    def process_frames(
        self,
        frames: List[tuple],
        fps: float
    ) -> List[PoseResult]:
        """
        Process multiple frames.

        Args:
            frames: List of (frame_number, frame_data) tuples
            fps: Video frames per second

        Returns:
            List of PoseResult objects
        """
        results = []
        for frame_number, frame in frames:
            result = self.process_frame(frame, frame_number, fps)
            results.append(result)
        return results

    def get_hand_positions(self, pose_result: PoseResult) -> Optional[Dict[str, tuple]]:
        """
        Extract hand positions from pose result.

        Args:
            pose_result: PoseResult from process_frame

        Returns:
            Dict with 'left' and 'right' hand (x, y) positions, or None
        """
        if pose_result.landmarks is None:
            return None

        return {
            'left': (
                pose_result.landmarks['left_wrist'].x,
                pose_result.landmarks['left_wrist'].y
            ),
            'right': (
                pose_result.landmarks['right_wrist'].x,
                pose_result.landmarks['right_wrist'].y
            )
        }

    def get_body_center(self, pose_result: PoseResult) -> Optional[tuple]:
        """
        Calculate body center from hip positions.

        Args:
            pose_result: PoseResult from process_frame

        Returns:
            (x, y) tuple of body center, or None
        """
        if pose_result.landmarks is None:
            return None

        left_hip = pose_result.landmarks['left_hip']
        right_hip = pose_result.landmarks['right_hip']

        center_x = (left_hip.x + right_hip.x) / 2
        center_y = (left_hip.y + right_hip.y) / 2

        return (center_x, center_y)

    def close(self):
        """Release resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
