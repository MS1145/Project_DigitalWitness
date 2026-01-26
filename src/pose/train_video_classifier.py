"""
Video-based training pipeline for behavior classifier.

This script trains the model using real video data from the Kaggle dataset,
processing each video to extract pose-based features for classification.
"""
import numpy as np
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib

# Import project modules
from .estimator import PoseEstimator
from .feature_extractor import FeatureExtractor
from ..video.loader import VideoLoader
from ..config import (
    TRAINING_DATA_DIR,
    MODELS_DIR,
    BEHAVIOR_MODEL_PATH,
    SLIDING_WINDOW_SIZE,
    SLIDING_WINDOW_STRIDE
)


# Training configuration
TRAINING_CLASSES = ["normal", "shoplifting"]
NORMAL_VIDEO_DIR = TRAINING_DATA_DIR / "normal"
SHOPLIFTING_VIDEO_DIR = TRAINING_DATA_DIR / "shoplifting"

# Processing settings
FRAME_SKIP = 2  # Process every Nth frame for speed (1 = all frames)
MAX_VIDEOS_PER_CLASS = None  # None = use all videos


@dataclass
class VideoProcessingResult:
    """Result of processing a single video."""
    video_path: str
    label: int
    label_name: str
    n_frames_processed: int
    n_poses_detected: int
    n_feature_windows: int
    features: Optional[np.ndarray]  # (n_windows, 21)
    success: bool
    error_message: str = ""


@dataclass
class TrainingMetrics:
    """Training metrics and results."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: List[List[int]]
    classification_report: str
    n_train_samples: int
    n_test_samples: int
    n_features: int
    feature_importance: Dict[str, float]


def discover_videos(video_dir: Path, max_videos: Optional[int] = None) -> List[Path]:
    """Discover all video files in a directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    videos = []

    if not video_dir.exists():
        print(f"  Warning: Directory not found: {video_dir}")
        return videos

    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
        videos.extend(video_dir.glob(f"*{ext.upper()}"))

    videos = sorted(set(videos))  # Remove duplicates and sort

    if max_videos is not None:
        videos = videos[:max_videos]

    return videos


def process_single_video(
    video_path: Path,
    label: int,
    label_name: str,
    pose_estimator: PoseEstimator,
    feature_extractor: FeatureExtractor,
    frame_skip: int = 1
) -> VideoProcessingResult:
    """
    Process a single video and extract features.

    Args:
        video_path: Path to video file
        label: Class label (0 or 1)
        label_name: Class name
        pose_estimator: PoseEstimator instance
        feature_extractor: FeatureExtractor instance
        frame_skip: Process every Nth frame

    Returns:
        VideoProcessingResult with extracted features
    """
    try:
        with VideoLoader(video_path) as loader:
            metadata = loader.metadata
            fps = metadata.fps

            # Collect pose results
            pose_results = []
            frames_processed = 0

            for frame_num, frame in loader.frames(step=frame_skip):
                # Adjust frame number for actual time calculation
                actual_frame_num = frame_num
                pose_result = pose_estimator.process_frame(frame, actual_frame_num, fps)
                pose_results.append(pose_result)
                frames_processed += 1

            if frames_processed == 0:
                return VideoProcessingResult(
                    video_path=str(video_path),
                    label=label,
                    label_name=label_name,
                    n_frames_processed=0,
                    n_poses_detected=0,
                    n_feature_windows=0,
                    features=None,
                    success=False,
                    error_message="No frames could be read"
                )

            # Count successful pose detections
            n_poses = sum(1 for p in pose_results if p.landmarks is not None)

            # Extract features using sliding windows
            pose_features = feature_extractor.extract_from_sequence(pose_results)

            if not pose_features:
                return VideoProcessingResult(
                    video_path=str(video_path),
                    label=label,
                    label_name=label_name,
                    n_frames_processed=frames_processed,
                    n_poses_detected=n_poses,
                    n_feature_windows=0,
                    features=None,
                    success=False,
                    error_message="Not enough frames for feature extraction"
                )

            # Stack features into array
            features = np.vstack([pf.features for pf in pose_features])

            return VideoProcessingResult(
                video_path=str(video_path),
                label=label,
                label_name=label_name,
                n_frames_processed=frames_processed,
                n_poses_detected=n_poses,
                n_feature_windows=len(pose_features),
                features=features,
                success=True
            )

    except Exception as e:
        return VideoProcessingResult(
            video_path=str(video_path),
            label=label,
            label_name=label_name,
            n_frames_processed=0,
            n_poses_detected=0,
            n_feature_windows=0,
            features=None,
            success=False,
            error_message=str(e)
        )


def process_all_videos(
    normal_dir: Path,
    shoplifting_dir: Path,
    frame_skip: int = FRAME_SKIP,
    max_videos: Optional[int] = MAX_VIDEOS_PER_CLASS
) -> Tuple[np.ndarray, np.ndarray, List[VideoProcessingResult]]:
    """
    Process all videos and extract features.

    Args:
        normal_dir: Directory with normal behavior videos
        shoplifting_dir: Directory with shoplifting videos
        frame_skip: Process every Nth frame
        max_videos: Maximum videos per class (None = all)

    Returns:
        Tuple of (features, labels, processing_results)
    """
    print("\n" + "=" * 60)
    print("Video Processing Pipeline")
    print("=" * 60)

    # Discover videos
    print("\n[1/4] Discovering videos...")
    normal_videos = discover_videos(normal_dir, max_videos)
    shoplifting_videos = discover_videos(shoplifting_dir, max_videos)

    print(f"  - Normal videos found: {len(normal_videos)}")
    print(f"  - Shoplifting videos found: {len(shoplifting_videos)}")

    if not normal_videos or not shoplifting_videos:
        raise ValueError("No videos found in one or both directories")

    # Initialize processors
    print("\n[2/4] Initializing pose estimator and feature extractor...")
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor(
        window_size=SLIDING_WINDOW_SIZE,
        window_stride=SLIDING_WINDOW_STRIDE
    )
    print(f"  - Window size: {SLIDING_WINDOW_SIZE} frames")
    print(f"  - Window stride: {SLIDING_WINDOW_STRIDE} frames")
    print(f"  - Frame skip: {frame_skip}")

    # Process videos
    print("\n[3/4] Processing videos...")
    all_results = []
    all_features = []
    all_labels = []

    total_videos = len(normal_videos) + len(shoplifting_videos)
    processed_count = 0

    # Process normal videos
    print("\n  Processing NORMAL videos:")
    for i, video_path in enumerate(normal_videos):
        processed_count += 1
        print(f"    [{processed_count}/{total_videos}] {video_path.name}...", end=" ", flush=True)

        result = process_single_video(
            video_path, label=0, label_name="normal",
            pose_estimator=pose_estimator,
            feature_extractor=feature_extractor,
            frame_skip=frame_skip
        )
        all_results.append(result)

        if result.success and result.features is not None:
            all_features.append(result.features)
            all_labels.extend([0] * len(result.features))
            print(f"OK ({result.n_feature_windows} windows)")
        else:
            print(f"FAILED: {result.error_message}")

    # Process shoplifting videos
    print("\n  Processing SHOPLIFTING videos:")
    for i, video_path in enumerate(shoplifting_videos):
        processed_count += 1
        print(f"    [{processed_count}/{total_videos}] {video_path.name}...", end=" ", flush=True)

        result = process_single_video(
            video_path, label=1, label_name="shoplifting",
            pose_estimator=pose_estimator,
            feature_extractor=feature_extractor,
            frame_skip=frame_skip
        )
        all_results.append(result)

        if result.success and result.features is not None:
            all_features.append(result.features)
            all_labels.extend([1] * len(result.features))
            print(f"OK ({result.n_feature_windows} windows)")
        else:
            print(f"FAILED: {result.error_message}")

    # Cleanup
    pose_estimator.close()

    # Compile results
    print("\n[4/4] Compiling features...")

    if not all_features:
        raise ValueError("No features extracted from any video")

    features = np.vstack(all_features)
    labels = np.array(all_labels)

    # Summary
    successful = sum(1 for r in all_results if r.success)
    failed = len(all_results) - successful

    print(f"\n  Processing Summary:")
    print(f"  - Total videos: {len(all_results)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Total feature windows: {len(labels)}")
    print(f"  - Normal samples: {np.sum(labels == 0)}")
    print(f"  - Shoplifting samples: {np.sum(labels == 1)}")
    print(f"  - Feature dimensions: {features.shape}")

    return features, labels, all_results


def train_and_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_folds: int = 5
) -> Tuple[RandomForestClassifier, StandardScaler, TrainingMetrics]:
    """
    Train and evaluate the classifier.

    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        test_size: Fraction for test set
        random_state: Random seed
        n_cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (trained_model, scaler, metrics)
    """
    print("\n" + "=" * 60)
    print("Model Training and Evaluation")
    print("=" * 60)

    # Split data
    print("\n[1/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")

    # Normalize features
    print("\n[2/5] Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  - Feature mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  - Feature std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # Train model
    print("\n[3/5] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print(f"  - Number of trees: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")

    # Cross-validation
    print("\n[4/5] Performing cross-validation...")
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"  - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  - CV Scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=TRAINING_CLASSES)

    print(f"\n  Test Set Results:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  Shoplifting")
    print(f"  Actual Normal    {conf_matrix[0,0]:4d}    {conf_matrix[0,1]:4d}")
    print(f"  Actual Shoplift  {conf_matrix[1,0]:4d}    {conf_matrix[1,1]:4d}")

    # Feature importance
    feature_extractor = FeatureExtractor()
    feature_names = feature_extractor.feature_names
    importance = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Top 10 Feature Importances:")
    for i, (name, imp) in enumerate(list(sorted_importance.items())[:10]):
        print(f"    {i+1}. {name}: {imp:.4f}")

    metrics = TrainingMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        cv_accuracy_mean=float(cv_scores.mean()),
        cv_accuracy_std=float(cv_scores.std()),
        confusion_matrix=conf_matrix.tolist(),
        classification_report=class_report,
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
        n_features=features.shape[1],
        feature_importance=sorted_importance
    )

    return model, scaler, metrics


def save_model_and_results(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    metrics: TrainingMetrics,
    processing_results: List[VideoProcessingResult],
    output_dir: Path = MODELS_DIR
) -> Dict[str, Path]:
    """
    Save trained model and training results.

    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        metrics: Training metrics
        processing_results: Video processing results
        output_dir: Output directory

    Returns:
        Dict of saved file paths
    """
    print("\n" + "=" * 60)
    print("Saving Model and Results")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "behavior_classifier.pkl"
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_mean': scaler.mean_,
        'feature_std': scaler.scale_,
        'classes': TRAINING_CLASSES
    }
    joblib.dump(model_data, model_path)
    print(f"  - Model saved: {model_path}")

    # Save training info
    info_path = output_dir / "behavior_classifier_info.json"
    training_info = {
        'training_date': datetime.now().isoformat(),
        'classes': TRAINING_CLASSES,
        'n_classes': len(TRAINING_CLASSES),
        'metrics': {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'cv_accuracy_mean': metrics.cv_accuracy_mean,
            'cv_accuracy_std': metrics.cv_accuracy_std,
        },
        'confusion_matrix': metrics.confusion_matrix,
        'n_train_samples': metrics.n_train_samples,
        'n_test_samples': metrics.n_test_samples,
        'n_features': metrics.n_features,
        'feature_importance': metrics.feature_importance,
        'model_params': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
        },
        'processing_config': {
            'window_size': SLIDING_WINDOW_SIZE,
            'window_stride': SLIDING_WINDOW_STRIDE,
            'frame_skip': FRAME_SKIP,
        }
    }

    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"  - Training info saved: {info_path}")

    # Save processing log
    log_path = output_dir / "video_processing_log.json"
    processing_log = {
        'total_videos': len(processing_results),
        'successful': sum(1 for r in processing_results if r.success),
        'failed': sum(1 for r in processing_results if not r.success),
        'videos': [
            {
                'path': r.video_path,
                'label': r.label_name,
                'success': r.success,
                'frames_processed': r.n_frames_processed,
                'poses_detected': r.n_poses_detected,
                'feature_windows': r.n_feature_windows,
                'error': r.error_message if not r.success else None
            }
            for r in processing_results
        ]
    }

    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=2)
    print(f"  - Processing log saved: {log_path}")

    return {
        'model': model_path,
        'info': info_path,
        'log': log_path
    }


def generate_visualizations(
    model: RandomForestClassifier,
    metrics: TrainingMetrics,
    output_dir: Path = MODELS_DIR
) -> List[Path]:
    """
    Generate training visualization plots.

    Args:
        model: Trained model
        metrics: Training metrics
        output_dir: Output directory

    Returns:
        List of saved plot paths
    """
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("  Warning: matplotlib not available, skipping visualizations")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # 1. Confusion Matrix
    print("  - Creating confusion matrix plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    conf_matrix = np.array(metrics.confusion_matrix)
    im = ax.imshow(conf_matrix, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(TRAINING_CLASSES)
    ax.set_yticklabels(TRAINING_CLASSES)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, conf_matrix[i, j],
                          ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black",
                          fontsize=14, fontweight='bold')

    plt.colorbar(im)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    saved_plots.append(cm_path)
    print(f"    Saved: {cm_path}")

    # 2. Feature Importance
    print("  - Creating feature importance plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    importance_items = list(metrics.feature_importance.items())
    names = [item[0] for item in importance_items]
    values = [item[1] for item in importance_items]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))[::-1]
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Random Forest)')
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()

    fi_path = output_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    saved_plots.append(fi_path)
    print(f"    Saved: {fi_path}")

    # 3. Metrics Summary
    print("  - Creating metrics summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Accuracy']
    metric_values = [
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1_score,
        metrics.cv_accuracy_mean
    ]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(metric_names, metric_values, color=colors)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
               f'{val:.2%}', ha='center', fontsize=11, fontweight='bold')

    # Add CV std as error bar
    cv_bar_idx = metric_names.index('CV Accuracy')
    ax.errorbar(cv_bar_idx, metrics.cv_accuracy_mean,
               yerr=metrics.cv_accuracy_std, fmt='none',
               color='black', capsize=5, capthick=2)

    plt.tight_layout()

    metrics_path = output_dir / "metrics_summary.png"
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    saved_plots.append(metrics_path)
    print(f"    Saved: {metrics_path}")

    print(f"\n  Generated {len(saved_plots)} visualization(s)")
    return saved_plots


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("  DIGITAL WITNESS - Video-Based Behavior Classifier Training")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Classes: {TRAINING_CLASSES}")

    # Check directories
    print(f"\nData directories:")
    print(f"  - Normal videos: {NORMAL_VIDEO_DIR}")
    print(f"  - Shoplifting videos: {SHOPLIFTING_VIDEO_DIR}")

    try:
        # Process videos
        features, labels, processing_results = process_all_videos(
            NORMAL_VIDEO_DIR,
            SHOPLIFTING_VIDEO_DIR,
            frame_skip=FRAME_SKIP,
            max_videos=MAX_VIDEOS_PER_CLASS
        )

        # Train and evaluate
        model, scaler, metrics = train_and_evaluate(features, labels)

        # Save model and results
        saved_paths = save_model_and_results(
            model, scaler, metrics, processing_results
        )

        # Generate visualizations
        plot_paths = generate_visualizations(model, metrics)

        # Final summary
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nFinal Results:")
        print(f"  - Test Accuracy:  {metrics.accuracy:.2%}")
        print(f"  - CV Accuracy:    {metrics.cv_accuracy_mean:.2%} (+/- {metrics.cv_accuracy_std:.2%})")
        print(f"  - F1 Score:       {metrics.f1_score:.2%}")
        print(f"\nModel saved to: {saved_paths['model']}")
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'metrics': metrics,
            'model_path': saved_paths['model'],
            'success': True
        }

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    main()
