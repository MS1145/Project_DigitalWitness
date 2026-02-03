"""
Training script for the deep learning pipeline.

Trains the LSTM classifier on extracted features from labeled videos.
YOLO and CNN use pretrained weights; only LSTM is trained from scratch.

Includes evaluation metrics: accuracy, precision, recall, F1, confusion matrix.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..config import (
    TRAINING_DATA_DIR,
    TRAINING_NORMAL_DIR,
    TRAINING_SHOPLIFTING_DIR,
    MODELS_DIR,
    LSTM_SEQUENCE_LENGTH,
    SLIDING_WINDOW_STRIDE,
    INTENT_CLASSES
)
from .cnn_feature_extractor import CNNFeatureExtractor
from .lstm_classifier import LSTMIntentClassifier


def compute_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> Dict:
    """
    Compute detailed classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Class names

    Returns:
        Dictionary with accuracy, precision, recall, f1, confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = len(classes)

    # Confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1

    # Overall accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(n_classes):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i][i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision)

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall)

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)

    # Macro-averaged metrics
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    # For binary classification, use class 1 (shoplifting) metrics
    if n_classes == 2:
        precision = precision_per_class[1]
        recall = recall_per_class[1]
        f1 = f1_per_class[1]
    else:
        precision = macro_precision
        recall = macro_recall
        f1 = macro_f1

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": {classes[i]: float(p) for i, p in enumerate(precision_per_class)},
        "per_class_recall": {classes[i]: float(r) for i, r in enumerate(recall_per_class)},
        "per_class_f1": {classes[i]: float(f) for i, f in enumerate(f1_per_class)},
        "confusion_matrix": confusion_matrix.tolist(),
        "classes": classes
    }


def extract_features_from_videos(
    video_paths: List[Path],
    cnn: CNNFeatureExtractor,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    stride: int = SLIDING_WINDOW_STRIDE,
    max_videos: Optional[int] = None
) -> List[np.ndarray]:
    """
    Extract CNN features from videos as sequences.

    Args:
        video_paths: List of video file paths
        cnn: Initialized CNN feature extractor
        sequence_length: Frames per sequence
        stride: Stride between sequences
        max_videos: Maximum videos to process (None = all)

    Returns:
        List of feature sequences (each shape: seq_len x feature_dim)
    """
    import cv2

    all_sequences = []

    if max_videos:
        video_paths = video_paths[:max_videos]

    for i, video_path in enumerate(video_paths):
        print(f"  Processing video {i+1}/{len(video_paths)}: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    Warning: Could not open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract features from all frames (no skipping for better accuracy)
        frame_features = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every frame for accurate feature extraction
            features = cnn.extract_features(
                frame, frame_num, frame_num / fps
            )
            frame_features.append(features.features)
            frame_num += 1

        cap.release()

        if not frame_features:
            continue

        # Create sequences with sliding window
        feature_array = np.array(frame_features)
        num_frames = len(frame_features)

        for start in range(0, num_frames - sequence_length + 1, stride):
            end = start + sequence_length
            sequence = feature_array[start:end]
            all_sequences.append(sequence)

    return all_sequences


def train_lstm_classifier(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    max_videos_per_class: Optional[int] = None
) -> Dict:
    """
    Train the LSTM classifier on labeled video data.

    Args:
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split ratio
        max_videos_per_class: Max videos to use per class

    Returns:
        Training results and metrics
    """
    print("=" * 60)
    print("  DEEP MODEL TRAINING")
    print("  LSTM Intent Classifier")
    print("=" * 60)
    print()

    # Initialize CNN for feature extraction
    print("[1/4] Initializing CNN feature extractor...")
    cnn = CNNFeatureExtractor()
    cnn.initialize()
    print(f"  - Backbone: {cnn.backbone_name}")
    print(f"  - Feature dim: {cnn.feature_dim}")
    print()

    # Find training videos
    print("[2/4] Loading training videos...")

    normal_videos = list(TRAINING_NORMAL_DIR.glob("*.mp4"))
    shoplifting_videos = list(TRAINING_SHOPLIFTING_DIR.glob("*.mp4"))

    print(f"  - Normal videos: {len(normal_videos)}")
    print(f"  - Shoplifting videos: {len(shoplifting_videos)}")

    if not normal_videos or not shoplifting_videos:
        print("\n  ERROR: Training videos not found!")
        print(f"  Expected videos in:")
        print(f"    - {TRAINING_NORMAL_DIR}")
        print(f"    - {TRAINING_SHOPLIFTING_DIR}")
        return {"success": False, "error": "No training videos found"}

    # Extract features
    print("\n[3/4] Extracting features from videos...")

    print("  Processing normal videos...")
    normal_sequences = extract_features_from_videos(
        normal_videos, cnn,
        max_videos=max_videos_per_class
    )
    print(f"  - Extracted {len(normal_sequences)} normal sequences")

    print("  Processing shoplifting videos...")
    shoplifting_sequences = extract_features_from_videos(
        shoplifting_videos, cnn,
        max_videos=max_videos_per_class
    )
    print(f"  - Extracted {len(shoplifting_sequences)} shoplifting sequences")

    cnn.close()

    # Prepare training data with 2-class system (matches available dataset)
    # Classes: normal=0, shoplifting=1
    # Dataset structure:
    #   - data/training/normal/      -> class 0 (normal behavior)
    #   - data/training/shoplifting/ -> class 1 (suspicious/shoplifting behavior)
    all_sequences = normal_sequences + shoplifting_sequences
    all_labels = [0] * len(normal_sequences) + [1] * len(shoplifting_sequences)

    print(f"\n  Total sequences: {len(all_sequences)}")
    print(f"  - Normal sequences (class 0): {len(normal_sequences)}")
    print(f"  - Shoplifting sequences (class 1): {len(shoplifting_sequences)}")

    if len(all_sequences) < 10:
        print("  ERROR: Not enough training data!")
        return {"success": False, "error": "Insufficient training data"}

    # Split train/val
    indices = np.arange(len(all_sequences))
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_sequences = [all_sequences[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_sequences = [all_sequences[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    print(f"  - Training samples: {len(train_sequences)}")
    print(f"  - Validation samples: {len(val_sequences)}")

    # Train LSTM with 2-class system (matches dataset)
    print("\n[4/4] Training LSTM classifier...")

    # Get input dimension from first sequence
    input_dim = train_sequences[0].shape[1]

    # Binary classification: normal vs shoplifting
    lstm = LSTMIntentClassifier(
        input_dim=input_dim,
        num_classes=2  # 2 classes: normal, shoplifting
    )
    lstm.classes = ["normal", "shoplifting"]

    results = lstm.train(
        sequences=train_sequences,
        labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Save model
    model_path = MODELS_DIR / "lstm_classifier.pt"
    lstm.save_model(model_path)
    print(f"\n  Model saved to: {model_path}")

    # Compute detailed metrics on validation set
    print("\n[5/5] Computing detailed evaluation metrics...")

    val_predictions = []
    for seq in val_sequences:
        pred = lstm.predict_sequence(seq)
        val_predictions.append(pred.class_id)

    detailed_metrics = compute_metrics(val_labels, val_predictions, lstm.classes)

    print(f"  - Accuracy:  {detailed_metrics['accuracy']:.1%}")
    print(f"  - Precision: {detailed_metrics['precision']:.1%}")
    print(f"  - Recall:    {detailed_metrics['recall']:.1%}")
    print(f"  - F1 Score:  {detailed_metrics['f1_score']:.1%}")

    # Save training info with detailed metrics
    info = {
        "training_date": datetime.now().isoformat(),
        "n_train_samples": len(train_sequences),
        "n_val_samples": len(val_sequences),
        "n_normal_videos": len(normal_videos),
        "n_shoplifting_videos": len(shoplifting_videos),
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "stride": SLIDING_WINDOW_STRIDE,
        "input_dim": input_dim,
        "epochs": epochs,
        "final_train_acc": results["final_train_acc"],
        "final_val_acc": results["final_val_acc"],
        "classes": lstm.classes,
        "metrics": {
            "accuracy": detailed_metrics["accuracy"],
            "precision": detailed_metrics["precision"],
            "recall": detailed_metrics["recall"],
            "f1_score": detailed_metrics["f1_score"]
        },
        "per_class_metrics": {
            "precision": detailed_metrics["per_class_precision"],
            "recall": detailed_metrics["per_class_recall"],
            "f1": detailed_metrics["per_class_f1"]
        },
        "confusion_matrix": detailed_metrics["confusion_matrix"]
    }

    info_path = MODELS_DIR / "lstm_classifier_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final Training Accuracy: {results['final_train_acc']:.1%}")
    print(f"  Final Validation Accuracy: {results['final_val_acc']:.1%}")
    print()
    print("  Detailed Metrics (on validation set):")
    print(f"    Precision: {detailed_metrics['precision']:.1%}")
    print(f"    Recall:    {detailed_metrics['recall']:.1%}")
    print(f"    F1 Score:  {detailed_metrics['f1_score']:.1%}")
    print()
    print("  Confusion Matrix:")
    print(f"    {lstm.classes}")
    for i, row in enumerate(detailed_metrics['confusion_matrix']):
        print(f"    {lstm.classes[i]}: {row}")
    print()

    lstm.close()

    return {
        "success": True,
        "model_path": str(model_path),
        "info": info
    }


def evaluate_model(
    test_normal_dir: Optional[Path] = None,
    test_shoplifting_dir: Optional[Path] = None,
    max_videos_per_class: Optional[int] = None
) -> Dict:
    """
    Evaluate trained model on test data.

    Args:
        test_normal_dir: Directory with normal test videos (defaults to training dir)
        test_shoplifting_dir: Directory with shoplifting test videos (defaults to training dir)
        max_videos_per_class: Max videos to evaluate per class

    Returns:
        Evaluation results with detailed metrics
    """
    print("=" * 60)
    print("  MODEL EVALUATION")
    print("  LSTM Intent Classifier")
    print("=" * 60)
    print()

    # Use training directories if not specified
    test_normal_dir = test_normal_dir or TRAINING_NORMAL_DIR
    test_shoplifting_dir = test_shoplifting_dir or TRAINING_SHOPLIFTING_DIR

    # Check if model exists
    model_path = MODELS_DIR / "lstm_classifier.pt"
    if not model_path.exists():
        print("  ERROR: No trained model found!")
        print(f"  Expected: {model_path}")
        print("  Run: python run.py --train")
        return {"success": False, "error": "Model not found"}

    # Initialize models
    print("[1/4] Loading trained LSTM model...")
    lstm = LSTMIntentClassifier()
    lstm.load_model(model_path)
    print(f"  - Classes: {lstm.classes}")
    print(f"  - Input dim: {lstm.input_dim}")

    print("\n[2/4] Initializing CNN feature extractor...")
    cnn = CNNFeatureExtractor()
    cnn.initialize()
    print(f"  - Backbone: {cnn.backbone_name}")

    # Find test videos
    print("\n[3/4] Loading test videos...")

    normal_videos = list(test_normal_dir.glob("*.mp4"))
    shoplifting_videos = list(test_shoplifting_dir.glob("*.mp4"))

    print(f"  - Normal videos: {len(normal_videos)}")
    print(f"  - Shoplifting videos: {len(shoplifting_videos)}")

    if not normal_videos and not shoplifting_videos:
        print("\n  ERROR: No test videos found!")
        return {"success": False, "error": "No test videos found"}

    # Extract features and predict
    print("\n[4/4] Evaluating model...")

    all_predictions = []
    all_labels = []

    # Process normal videos
    print("  Processing normal videos...")
    normal_sequences = extract_features_from_videos(
        normal_videos, cnn,
        max_videos=max_videos_per_class
    )
    for seq in normal_sequences:
        pred = lstm.predict_sequence(seq)
        all_predictions.append(pred.class_id)
        all_labels.append(0)  # normal = 0

    print(f"    - {len(normal_sequences)} sequences evaluated")

    # Process shoplifting videos
    print("  Processing shoplifting videos...")
    shoplifting_sequences = extract_features_from_videos(
        shoplifting_videos, cnn,
        max_videos=max_videos_per_class
    )
    for seq in shoplifting_sequences:
        pred = lstm.predict_sequence(seq)
        all_predictions.append(pred.class_id)
        all_labels.append(1)  # shoplifting = 1

    print(f"    - {len(shoplifting_sequences)} sequences evaluated")

    cnn.close()
    lstm.close()

    # Compute metrics
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    metrics = compute_metrics(all_labels, all_predictions, ["normal", "shoplifting"])

    print(f"\n  Total samples evaluated: {len(all_labels)}")
    print(f"    - Normal: {all_labels.count(0)}")
    print(f"    - Shoplifting: {all_labels.count(1)}")

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.1%}")
    print(f"    Precision: {metrics['precision']:.1%}")
    print(f"    Recall:    {metrics['recall']:.1%}")
    print(f"    F1 Score:  {metrics['f1_score']:.1%}")

    print(f"\n  Per-Class Precision:")
    for cls, val in metrics['per_class_precision'].items():
        print(f"    {cls}: {val:.1%}")

    print(f"\n  Per-Class Recall:")
    for cls, val in metrics['per_class_recall'].items():
        print(f"    {cls}: {val:.1%}")

    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Shoplifting")
    cm = metrics['confusion_matrix']
    print(f"    Normal      {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"    Shoplifting {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Update info file with evaluation metrics
    info_path = MODELS_DIR / "lstm_classifier_info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)

        info["evaluation"] = {
            "date": datetime.now().isoformat(),
            "n_samples": len(all_labels),
            "n_normal": all_labels.count(0),
            "n_shoplifting": all_labels.count(1),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            },
            "per_class_metrics": {
                "precision": metrics["per_class_precision"],
                "recall": metrics["per_class_recall"],
                "f1": metrics["per_class_f1"]
            },
            "confusion_matrix": metrics["confusion_matrix"]
        }

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n  Metrics saved to: {info_path}")

    print()

    return {
        "success": True,
        "metrics": metrics
    }


def main():
    """Entry point for training."""
    return train_lstm_classifier()


if __name__ == "__main__":
    main()
