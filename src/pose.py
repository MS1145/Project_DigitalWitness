"""
Pose estimation and behavior classification module for Digital Witness.
Handles MediaPipe pose detection, feature extraction, and ML classification.
"""
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .config import (
    POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE,
    SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE,
    BEHAVIOR_CLASSES, BEHAVIOR_MODEL_PATH
)

# MediaPipe landmark indices for upper body
LANDMARK_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb", 23: "left_hip", 24: "right_hip",
}

FEATURE_NAMES = [
    "left_hand_velocity_mean", "left_hand_velocity_max",
    "right_hand_velocity_mean", "right_hand_velocity_max",
    "left_hand_body_dist_mean", "left_hand_body_dist_min",
    "right_hand_body_dist_mean", "right_hand_body_dist_min",
    "left_elbow_angle_mean", "left_elbow_angle_std",
    "right_elbow_angle_mean", "right_elbow_angle_std",
    "body_displacement_x", "body_displacement_y",
    "body_velocity_mean", "body_velocity_max",
    "left_hand_height_mean", "left_hand_height_min",
    "right_hand_height_mean", "right_hand_height_min",
    "pose_detection_rate",
]


@dataclass
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    frame_number: int
    timestamp: float
    landmarks: Optional[Dict[str, PoseLandmark]]
    raw_landmarks: Optional[Any]


@dataclass
class PoseFeatures:
    window_start_frame: int
    window_end_frame: int
    window_start_time: float
    window_end_time: float
    features: np.ndarray
    feature_names: List[str]


@dataclass
class BehaviorEvent:
    behavior_type: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    probabilities: Dict[str, float]


class PoseEstimator:
    """MediaPipe pose estimation wrapper."""

    def __init__(self, min_detection_confidence: float = POSE_MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = POSE_MIN_TRACKING_CONFIDENCE):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> PoseResult:
        rgb_frame = frame[:, :, ::-1]
        results = self.pose.process(rgb_frame)
        landmarks = None
        raw_landmarks = None
        if results.pose_landmarks:
            raw_landmarks = results.pose_landmarks
            landmarks = {}
            for idx, name in LANDMARK_NAMES.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = PoseLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
        return PoseResult(frame_number=frame_number, timestamp=frame_number / fps,
                          landmarks=landmarks, raw_landmarks=raw_landmarks)

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FeatureExtractor:
    """Converts pose sequences to ML feature vectors."""

    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE,
                 window_stride: int = SLIDING_WINDOW_STRIDE):
        self.window_size = window_size
        self.window_stride = window_stride
        self.feature_names = FEATURE_NAMES

    def extract_from_sequence(self, pose_results: List[PoseResult]) -> List[PoseFeatures]:
        features_list = []
        for start_idx in range(0, len(pose_results) - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size
            window = pose_results[start_idx:end_idx]
            features = self._extract_window_features(window)
            if features is not None:
                features_list.append(features)
        return features_list

    def _extract_window_features(self, window: List[PoseResult]) -> Optional[PoseFeatures]:
        if not window:
            return None
        valid_poses = [p for p in window if p.landmarks is not None]
        pose_detection_rate = len(valid_poses) / len(window)
        if len(valid_poses) < 3:
            features = np.zeros(len(self.feature_names))
            features[-1] = pose_detection_rate
            return PoseFeatures(
                window_start_frame=window[0].frame_number, window_end_frame=window[-1].frame_number,
                window_start_time=window[0].timestamp, window_end_time=window[-1].timestamp,
                features=features, feature_names=self.feature_names
            )

        left_hand_pos, right_hand_pos, body_center_pos = [], [], []
        left_elbow_angles, right_elbow_angles = [], []
        left_hand_heights, right_hand_heights = [], []

        for pose in valid_poses:
            lm = pose.landmarks
            left_hand_pos.append((lm['left_wrist'].x, lm['left_wrist'].y))
            right_hand_pos.append((lm['right_wrist'].x, lm['right_wrist'].y))
            center_x = (lm['left_hip'].x + lm['right_hip'].x) / 2
            center_y = (lm['left_hip'].y + lm['right_hip'].y) / 2
            body_center_pos.append((center_x, center_y))
            left_elbow_angles.append(self._calc_angle(
                (lm['left_shoulder'].x, lm['left_shoulder'].y),
                (lm['left_elbow'].x, lm['left_elbow'].y),
                (lm['left_wrist'].x, lm['left_wrist'].y)))
            right_elbow_angles.append(self._calc_angle(
                (lm['right_shoulder'].x, lm['right_shoulder'].y),
                (lm['right_elbow'].x, lm['right_elbow'].y),
                (lm['right_wrist'].x, lm['right_wrist'].y)))
            shoulder_y = (lm['left_shoulder'].y + lm['right_shoulder'].y) / 2
            left_hand_heights.append(shoulder_y - lm['left_wrist'].y)
            right_hand_heights.append(shoulder_y - lm['right_wrist'].y)

        left_hand_pos = np.array(left_hand_pos)
        right_hand_pos = np.array(right_hand_pos)
        body_center_pos = np.array(body_center_pos)

        left_hand_vel = self._calc_velocities(left_hand_pos)
        right_hand_vel = self._calc_velocities(right_hand_pos)
        body_vel = self._calc_velocities(body_center_pos)
        left_hand_body_dist = np.linalg.norm(left_hand_pos - body_center_pos, axis=1)
        right_hand_body_dist = np.linalg.norm(right_hand_pos - body_center_pos, axis=1)
        body_displacement = body_center_pos[-1] - body_center_pos[0]

        features = np.array([
            np.mean(left_hand_vel) if len(left_hand_vel) > 0 else 0,
            np.max(left_hand_vel) if len(left_hand_vel) > 0 else 0,
            np.mean(right_hand_vel) if len(right_hand_vel) > 0 else 0,
            np.max(right_hand_vel) if len(right_hand_vel) > 0 else 0,
            np.mean(left_hand_body_dist), np.min(left_hand_body_dist),
            np.mean(right_hand_body_dist), np.min(right_hand_body_dist),
            np.mean(left_elbow_angles), np.std(left_elbow_angles),
            np.mean(right_elbow_angles), np.std(right_elbow_angles),
            body_displacement[0], body_displacement[1],
            np.mean(body_vel) if len(body_vel) > 0 else 0,
            np.max(body_vel) if len(body_vel) > 0 else 0,
            np.mean(left_hand_heights), np.min(left_hand_heights),
            np.mean(right_hand_heights), np.min(right_hand_heights),
            pose_detection_rate,
        ])
        return PoseFeatures(
            window_start_frame=window[0].frame_number, window_end_frame=window[-1].frame_number,
            window_start_time=window[0].timestamp, window_end_time=window[-1].timestamp,
            features=features, feature_names=self.feature_names
        )

    def _calc_velocities(self, positions: np.ndarray) -> np.ndarray:
        if len(positions) < 2:
            return np.array([])
        return np.linalg.norm(np.diff(positions, axis=0), axis=1)

    def _calc_angle(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


class BehaviorClassifier:
    """RandomForest-based behavior classifier."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or BEHAVIOR_MODEL_PATH
        self.classes = BEHAVIOR_CLASSES
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        if self.model_path.exists():
            self.load_model()

    def train(self, features: np.ndarray, labels: np.ndarray,
              n_estimators: int = 100, max_depth: int = 10) -> Dict[str, float]:
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(features)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, class_weight='balanced'
        )
        self.model.fit(features_normalized, labels)
        predictions = self.model.predict(features_normalized)
        return {"accuracy": np.mean(predictions == labels), "n_samples": len(labels)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(self.scaler.transform(features))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict_proba(self.scaler.transform(features))

    def classify_sequence(self, pose_features: List[PoseFeatures]) -> List[BehaviorEvent]:
        if not pose_features:
            return []
        feature_matrix = np.vstack([pf.features for pf in pose_features])
        predictions = self.predict(feature_matrix)
        probabilities = self.predict_proba(feature_matrix)
        events = []
        for pf, pred, probs in zip(pose_features, predictions, probabilities):
            events.append(BehaviorEvent(
                behavior_type=self.classes[pred],
                start_time=pf.window_start_time, end_time=pf.window_end_time,
                start_frame=pf.window_start_frame, end_frame=pf.window_end_frame,
                confidence=float(np.max(probs)),
                probabilities={cls: float(p) for cls, p in zip(self.classes, probs)}
            ))
        return events

    def save_model(self, path: Optional[Path] = None) -> None:
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'classes': self.classes}, path)

    def load_model(self, path: Optional[Path] = None) -> None:
        path = path or self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.classes = data.get('classes', BEHAVIOR_CLASSES)
