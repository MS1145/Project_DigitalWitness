"""
Feature extraction from pose sequences for ML classification.

Transforms raw pose landmark sequences into a fixed-size feature vector
suitable for sklearn classifiers. Uses sliding windows to capture temporal
patterns in body movement.
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .estimator import PoseResult
from ..config import SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE


@dataclass
class PoseFeatures:
    """Feature vector extracted from a temporal window of pose frames."""
    window_start_frame: int
    window_end_frame: int
    window_start_time: float
    window_end_time: float
    features: np.ndarray       # Shape: (21,) - the feature vector
    feature_names: List[str]   # Human-readable feature labels


class FeatureExtractor:
    """
    Converts pose sequences to ML feature vectors.

    Uses a sliding window approach to extract 21 features capturing:
    - Hand velocity and trajectory
    - Hand-to-body proximity (concealment indicator)
    - Arm joint angles
    - Body movement patterns
    """

    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        window_stride: int = SLIDING_WINDOW_STRIDE
    ):
        self.window_size = window_size
        self.window_stride = window_stride

        # Feature names enable model interpretability and debugging
        self.feature_names = [
            # Hand velocities (4 features)
            "left_hand_velocity_mean",
            "left_hand_velocity_max",
            "right_hand_velocity_mean",
            "right_hand_velocity_max",
            # Hand-to-body distance (4 features)
            "left_hand_body_dist_mean",
            "left_hand_body_dist_min",
            "right_hand_body_dist_mean",
            "right_hand_body_dist_min",
            # Arm angles (4 features)
            "left_elbow_angle_mean",
            "left_elbow_angle_std",
            "right_elbow_angle_mean",
            "right_elbow_angle_std",
            # Body trajectory (4 features)
            "body_displacement_x",
            "body_displacement_y",
            "body_velocity_mean",
            "body_velocity_max",
            # Hand height relative to body (4 features)
            "left_hand_height_mean",
            "left_hand_height_min",
            "right_hand_height_mean",
            "right_hand_height_min",
            # Pose detection rate (1 feature)
            "pose_detection_rate",
        ]

    def extract_from_sequence(
        self,
        pose_results: List[PoseResult]
    ) -> List[PoseFeatures]:
        """
        Extract features from a sequence of pose results using sliding windows.

        Args:
            pose_results: List of PoseResult from pose estimation

        Returns:
            List of PoseFeatures, one per window
        """
        features_list = []

        for start_idx in range(0, len(pose_results) - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size
            window = pose_results[start_idx:end_idx]

            features = self._extract_window_features(window)
            if features is not None:
                features_list.append(features)

        return features_list

    def _extract_window_features(
        self,
        window: List[PoseResult]
    ) -> Optional[PoseFeatures]:
        """Extract features from a single window of pose results."""
        if not window:
            return None

        # Collect valid poses
        valid_poses = [p for p in window if p.landmarks is not None]
        pose_detection_rate = len(valid_poses) / len(window)

        # If too few valid poses, return zeros with detection rate
        if len(valid_poses) < 3:
            features = np.zeros(len(self.feature_names))
            features[-1] = pose_detection_rate
            return PoseFeatures(
                window_start_frame=window[0].frame_number,
                window_end_frame=window[-1].frame_number,
                window_start_time=window[0].timestamp,
                window_end_time=window[-1].timestamp,
                features=features,
                feature_names=self.feature_names
            )

        # Extract position sequences
        left_hand_pos = []
        right_hand_pos = []
        body_center_pos = []
        left_elbow_angles = []
        right_elbow_angles = []
        left_hand_heights = []
        right_hand_heights = []

        for pose in valid_poses:
            lm = pose.landmarks

            # Hand positions
            left_hand_pos.append((lm['left_wrist'].x, lm['left_wrist'].y))
            right_hand_pos.append((lm['right_wrist'].x, lm['right_wrist'].y))

            # Body center
            center_x = (lm['left_hip'].x + lm['right_hip'].x) / 2
            center_y = (lm['left_hip'].y + lm['right_hip'].y) / 2
            body_center_pos.append((center_x, center_y))

            # Elbow angles
            left_angle = self._calculate_angle(
                (lm['left_shoulder'].x, lm['left_shoulder'].y),
                (lm['left_elbow'].x, lm['left_elbow'].y),
                (lm['left_wrist'].x, lm['left_wrist'].y)
            )
            right_angle = self._calculate_angle(
                (lm['right_shoulder'].x, lm['right_shoulder'].y),
                (lm['right_elbow'].x, lm['right_elbow'].y),
                (lm['right_wrist'].x, lm['right_wrist'].y)
            )
            left_elbow_angles.append(left_angle)
            right_elbow_angles.append(right_angle)

            # Hand heights relative to shoulders
            shoulder_y = (lm['left_shoulder'].y + lm['right_shoulder'].y) / 2
            left_hand_heights.append(shoulder_y - lm['left_wrist'].y)  # Positive = above shoulders
            right_hand_heights.append(shoulder_y - lm['right_wrist'].y)

        # Convert to numpy arrays
        left_hand_pos = np.array(left_hand_pos)
        right_hand_pos = np.array(right_hand_pos)
        body_center_pos = np.array(body_center_pos)

        # Calculate velocities
        left_hand_vel = self._calculate_velocities(left_hand_pos)
        right_hand_vel = self._calculate_velocities(right_hand_pos)
        body_vel = self._calculate_velocities(body_center_pos)

        # Calculate hand-to-body distances
        left_hand_body_dist = np.linalg.norm(left_hand_pos - body_center_pos, axis=1)
        right_hand_body_dist = np.linalg.norm(right_hand_pos - body_center_pos, axis=1)

        # Body displacement
        body_displacement = body_center_pos[-1] - body_center_pos[0]

        # Compile feature vector
        features = np.array([
            # Hand velocities
            np.mean(left_hand_vel) if len(left_hand_vel) > 0 else 0,
            np.max(left_hand_vel) if len(left_hand_vel) > 0 else 0,
            np.mean(right_hand_vel) if len(right_hand_vel) > 0 else 0,
            np.max(right_hand_vel) if len(right_hand_vel) > 0 else 0,
            # Hand-to-body distance
            np.mean(left_hand_body_dist),
            np.min(left_hand_body_dist),
            np.mean(right_hand_body_dist),
            np.min(right_hand_body_dist),
            # Elbow angles
            np.mean(left_elbow_angles),
            np.std(left_elbow_angles),
            np.mean(right_elbow_angles),
            np.std(right_elbow_angles),
            # Body trajectory
            body_displacement[0],
            body_displacement[1],
            np.mean(body_vel) if len(body_vel) > 0 else 0,
            np.max(body_vel) if len(body_vel) > 0 else 0,
            # Hand heights
            np.mean(left_hand_heights),
            np.min(left_hand_heights),
            np.mean(right_hand_heights),
            np.min(right_hand_heights),
            # Detection rate
            pose_detection_rate,
        ])

        return PoseFeatures(
            window_start_frame=window[0].frame_number,
            window_end_frame=window[-1].frame_number,
            window_start_time=window[0].timestamp,
            window_end_time=window[-1].timestamp,
            features=features,
            feature_names=self.feature_names
        )

    def _calculate_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute frame-to-frame velocity magnitudes from position sequence."""
        if len(positions) < 2:
            return np.array([])
        deltas = np.diff(positions, axis=0)
        velocities = np.linalg.norm(deltas, axis=1)
        return velocities

    def _calculate_angle(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        """
        Compute joint angle at p2 using vector math.

        Used for elbow angle calculation which helps detect concealment
        (bent arm bringing item close to body).
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Epsilon prevents division by zero for degenerate poses
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def normalize_features(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Normalize features using z-score normalization.

        Args:
            features: Feature array (n_samples, n_features)
            mean: Pre-computed mean (for inference)
            std: Pre-computed std (for inference)

        Returns:
            Tuple of (normalized_features, mean, std)
        """
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Prevent division by zero

        normalized = (features - mean) / std
        return normalized, mean, std
