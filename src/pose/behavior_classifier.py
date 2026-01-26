"""
ML-based behavior classification for Digital Witness.
"""
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .feature_extractor import PoseFeatures
from ..config import BEHAVIOR_CLASSES, BEHAVIOR_MODEL_PATH


@dataclass
class BehaviorEvent:
    """A detected behavior event."""
    behavior_type: str  # "normal", "pickup", "concealment", "bypass"
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    probabilities: Dict[str, float]  # Probability for each class


class BehaviorClassifier:
    """ML classifier for detecting shopping behaviors."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize behavior classifier.

        Args:
            model_path: Path to saved model, or None for new model
        """
        self.model_path = model_path or BEHAVIOR_MODEL_PATH
        self.classes = BEHAVIOR_CLASSES
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        # Try to load existing model
        if self.model_path.exists():
            self.load_model()

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the behavior classifier.

        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (n_samples,) as class indices
            n_estimators: Number of trees in random forest
            max_depth: Maximum tree depth
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        # Normalize features
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(features)

        # Store normalization parameters
        self.feature_mean = self.scaler.mean_
        self.feature_std = self.scaler.scale_

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        self.model.fit(features_normalized, labels)

        # Calculate training accuracy
        predictions = self.model.predict(features_normalized)
        accuracy = np.mean(predictions == labels)

        return {
            "accuracy": accuracy,
            "n_samples": len(labels),
            "n_features": features.shape[1],
            "classes": self.classes
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict behavior classes.

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Array of predicted class indices
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        features_normalized = self.scaler.transform(features)
        return self.model.predict(features_normalized)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Array of class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        features_normalized = self.scaler.transform(features)
        return self.model.predict_proba(features_normalized)

    def classify_sequence(
        self,
        pose_features: List[PoseFeatures]
    ) -> List[BehaviorEvent]:
        """
        Classify a sequence of pose feature windows.

        Args:
            pose_features: List of PoseFeatures from feature extractor

        Returns:
            List of BehaviorEvent objects
        """
        if not pose_features:
            return []

        # Stack features
        feature_matrix = np.vstack([pf.features for pf in pose_features])

        # Get predictions and probabilities
        predictions = self.predict(feature_matrix)
        probabilities = self.predict_proba(feature_matrix)

        # Convert to BehaviorEvents
        events = []
        for i, (pf, pred, probs) in enumerate(zip(pose_features, predictions, probabilities)):
            prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
            confidence = float(np.max(probs))

            event = BehaviorEvent(
                behavior_type=self.classes[pred],
                start_time=pf.window_start_time,
                end_time=pf.window_end_time,
                start_frame=pf.window_start_frame,
                end_frame=pf.window_end_frame,
                confidence=confidence,
                probabilities=prob_dict
            )
            events.append(event)

        return events

    def merge_consecutive_events(
        self,
        events: List[BehaviorEvent],
        min_confidence: float = 0.5
    ) -> List[BehaviorEvent]:
        """
        Merge consecutive events of the same type.

        Args:
            events: List of BehaviorEvent objects
            min_confidence: Minimum confidence to include event

        Returns:
            List of merged BehaviorEvent objects
        """
        if not events:
            return []

        # Filter by confidence
        filtered = [e for e in events if e.confidence >= min_confidence]
        if not filtered:
            return []

        merged = []
        current = filtered[0]

        for event in filtered[1:]:
            if event.behavior_type == current.behavior_type:
                # Merge: extend current event
                current = BehaviorEvent(
                    behavior_type=current.behavior_type,
                    start_time=current.start_time,
                    end_time=event.end_time,
                    start_frame=current.start_frame,
                    end_frame=event.end_frame,
                    confidence=(current.confidence + event.confidence) / 2,
                    probabilities=current.probabilities  # Keep first probabilities
                )
            else:
                # Different type: save current and start new
                merged.append(current)
                current = event

        merged.append(current)
        return merged

    def get_suspicious_events(
        self,
        events: List[BehaviorEvent]
    ) -> List[BehaviorEvent]:
        """
        Filter for suspicious events (shoplifting, concealment, or bypass).

        Args:
            events: List of BehaviorEvent objects

        Returns:
            List of suspicious events only
        """
        suspicious_types = {"shoplifting", "concealment", "bypass"}
        return [e for e in events if e.behavior_type in suspicious_types]

    def save_model(self, path: Optional[Path] = None) -> None:
        """Save model and scaler to disk."""
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'classes': self.classes
        }
        joblib.dump(model_data, path)

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load model and scaler from disk."""
        path = path or self.model_path

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_mean = model_data['feature_mean']
        self.feature_std = model_data['feature_std']
        self.classes = model_data['classes']
