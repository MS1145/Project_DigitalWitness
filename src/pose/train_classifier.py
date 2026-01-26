"""
Training pipeline for behavior classifier.

For MVP, includes synthetic data generation since real labeled data is not available.
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict

from .behavior_classifier import BehaviorClassifier
from .feature_extractor import FeatureExtractor
from ..config import (
    BEHAVIOR_CLASSES,
    BEHAVIOR_MODEL_PATH,
    TRAINING_DATA_DIR,
    MODELS_DIR
)


def generate_synthetic_training_data(
    n_samples_per_class: int = 200,
    n_features: int = 21,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for MVP demonstration.

    Each behavior class has characteristic feature patterns:
    - normal: Moderate hand movement, hands away from body
    - pickup: High hand velocity, hands at shelf height
    - concealment: Hands move toward body, low hand-body distance
    - bypass: Large body displacement, moving trajectory

    Args:
        n_samples_per_class: Number of samples per class
        n_features: Number of features (must match FeatureExtractor)
        random_state: Random seed

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)

    features_list = []
    labels_list = []

    for class_idx, behavior in enumerate(BEHAVIOR_CLASSES):
        class_features = _generate_class_features(behavior, n_samples_per_class, n_features)
        features_list.append(class_features)
        labels_list.extend([class_idx] * n_samples_per_class)

    features = np.vstack(features_list)
    labels = np.array(labels_list)

    # Shuffle
    shuffle_idx = np.random.permutation(len(labels))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    return features, labels


def _generate_class_features(
    behavior: str,
    n_samples: int,
    n_features: int
) -> np.ndarray:
    """Generate synthetic features for a specific behavior class."""
    features = np.random.randn(n_samples, n_features) * 0.1  # Base noise

    # Feature indices (matching FeatureExtractor.feature_names)
    # 0-3: Hand velocities
    # 4-7: Hand-body distances
    # 8-11: Elbow angles
    # 12-15: Body trajectory
    # 16-19: Hand heights
    # 20: Pose detection rate

    if behavior == "normal":
        # Normal shopping: moderate movement, hands away from body
        features[:, 0:4] += np.random.uniform(0.02, 0.08, (n_samples, 4))  # Moderate velocity
        features[:, 4:8] += np.random.uniform(0.2, 0.4, (n_samples, 4))  # Hands away from body
        features[:, 8:12] += np.random.uniform(120, 160, (n_samples, 4))  # Relaxed arms
        features[:, 12:16] += np.random.uniform(-0.1, 0.1, (n_samples, 4))  # Minimal displacement
        features[:, 16:20] += np.random.uniform(-0.1, 0.1, (n_samples, 4))  # Hands at mid level
        features[:, 20] = np.random.uniform(0.8, 1.0, n_samples)  # High detection

    elif behavior == "pickup":
        # Product pickup: hands reach up/out, high velocity
        features[:, 0:4] += np.random.uniform(0.08, 0.2, (n_samples, 4))  # High velocity
        features[:, 4:8] += np.random.uniform(0.3, 0.6, (n_samples, 4))  # Reaching out
        features[:, 8:12] += np.random.uniform(90, 140, (n_samples, 4))  # Extended arms
        features[:, 12:16] += np.random.uniform(-0.05, 0.05, (n_samples, 4))  # Stationary
        features[:, 16:20] += np.random.uniform(0.0, 0.3, (n_samples, 4))  # Hands above shoulders
        features[:, 20] = np.random.uniform(0.85, 1.0, n_samples)

    elif behavior == "concealment":
        # Concealment: hands move to body, low hand-body distance
        features[:, 0:4] += np.random.uniform(0.05, 0.15, (n_samples, 4))  # Moderate velocity
        features[:, 4:8] += np.random.uniform(0.05, 0.15, (n_samples, 4))  # Very close to body
        features[:, 8:12] += np.random.uniform(30, 80, (n_samples, 4))  # Bent arms
        features[:, 12:16] += np.random.uniform(-0.05, 0.05, (n_samples, 4))  # Stationary
        features[:, 16:20] += np.random.uniform(-0.3, -0.1, (n_samples, 4))  # Hands low (pocket level)
        features[:, 20] = np.random.uniform(0.7, 0.95, n_samples)

    elif behavior == "bypass":
        # Checkout bypass: large body displacement, moving away
        features[:, 0:4] += np.random.uniform(0.03, 0.1, (n_samples, 4))  # Moderate velocity
        features[:, 4:8] += np.random.uniform(0.15, 0.3, (n_samples, 4))  # Normal distance
        features[:, 8:12] += np.random.uniform(100, 150, (n_samples, 4))  # Normal arms
        features[:, 12:16] += np.random.uniform(0.15, 0.4, (n_samples, 4))  # Large displacement
        features[:, 16:20] += np.random.uniform(-0.15, 0.05, (n_samples, 4))  # Hands normal
        features[:, 20] = np.random.uniform(0.75, 1.0, n_samples)

    return features


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path = BEHAVIOR_MODEL_PATH,
    n_folds: int = 5
) -> Dict:
    """
    Train and evaluate the behavior classifier.

    Args:
        features: Training features
        labels: Training labels
        output_path: Path to save trained model
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary with training results
    """
    classifier = BehaviorClassifier()

    # Train model
    train_results = classifier.train(features, labels)

    # Cross-validation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    cv_scores = cross_val_score(
        classifier.model,
        features_scaled,
        labels,
        cv=n_folds,
        scoring='accuracy'
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(output_path)

    results = {
        **train_results,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "model_path": str(output_path)
    }

    return results


def save_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path = TRAINING_DATA_DIR
) -> Path:
    """Save training data for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "features": features.tolist(),
        "labels": labels.tolist(),
        "classes": BEHAVIOR_CLASSES,
        "n_samples": len(labels),
        "n_features": features.shape[1]
    }

    output_path = output_dir / "synthetic_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    """Main training script."""
    print("=" * 60)
    print("Digital Witness - Behavior Classifier Training")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1/4] Generating synthetic training data...")
    features, labels = generate_synthetic_training_data(
        n_samples_per_class=200,
        n_features=21
    )
    print(f"  - Generated {len(labels)} samples")
    print(f"  - Feature dimensions: {features.shape}")
    print(f"  - Classes: {BEHAVIOR_CLASSES}")

    # Save training data
    print("\n[2/4] Saving training data...")
    data_path = save_training_data(features, labels)
    print(f"  - Saved to: {data_path}")

    # Train model
    print("\n[3/4] Training classifier...")
    results = train_model(features, labels)
    print(f"  - Training accuracy: {results['accuracy']:.4f}")
    print(f"  - CV accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']:.4f})")

    # Summary
    print("\n[4/4] Training complete!")
    print(f"  - Model saved to: {results['model_path']}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Samples:        {results['n_samples']}")
    print(f"  Features:       {results['n_features']}")
    print(f"  CV Accuracy:    {results['cv_accuracy_mean']:.2%}")
    print(f"  Classes:        {', '.join(results['classes'])}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
