"""
Training pipeline for Digital Witness behavior classifier.
Processes videos and trains the ML model.
"""
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

from .pose import PoseEstimator, FeatureExtractor
from .video import VideoLoader
from .config import (
    TRAINING_DATA_DIR, MODELS_DIR, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE
)

TRAINING_CLASSES = ["normal", "shoplifting"]
NORMAL_VIDEO_DIR = TRAINING_DATA_DIR / "normal"
SHOPLIFTING_VIDEO_DIR = TRAINING_DATA_DIR / "shoplifting"
FRAME_SKIP = 2


@dataclass
class VideoProcessingResult:
    video_path: str
    label: int
    label_name: str
    n_frames_processed: int
    n_poses_detected: int
    n_feature_windows: int
    features: Optional[np.ndarray]
    success: bool
    error_message: str = ""


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: List[List[int]]
    n_train_samples: int
    n_test_samples: int
    n_features: int
    feature_importance: Dict[str, float]


def discover_videos(video_dir: Path, max_videos: Optional[int] = None) -> List[Path]:
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    videos = []
    if not video_dir.exists():
        print(f"  Warning: Directory not found: {video_dir}")
        return videos
    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
        videos.extend(video_dir.glob(f"*{ext.upper()}"))
    videos = sorted(set(videos))
    return videos[:max_videos] if max_videos else videos


def process_single_video(video_path: Path, label: int, label_name: str,
                         pose_estimator: PoseEstimator, feature_extractor: FeatureExtractor,
                         frame_skip: int = 1) -> VideoProcessingResult:
    try:
        with VideoLoader(video_path) as loader:
            fps = loader.metadata.fps
            pose_results = []
            frames_processed = 0
            for frame_num, frame in loader.frames(step=frame_skip):
                pose_result = pose_estimator.process_frame(frame, frame_num, fps)
                pose_results.append(pose_result)
                frames_processed += 1
            if frames_processed == 0:
                return VideoProcessingResult(str(video_path), label, label_name, 0, 0, 0, None, False, "No frames")
            n_poses = sum(1 for p in pose_results if p.landmarks is not None)
            pose_features = feature_extractor.extract_from_sequence(pose_results)
            if not pose_features:
                return VideoProcessingResult(str(video_path), label, label_name, frames_processed, n_poses, 0, None, False, "Not enough frames")
            features = np.vstack([pf.features for pf in pose_features])
            return VideoProcessingResult(str(video_path), label, label_name, frames_processed, n_poses, len(pose_features), features, True)
    except Exception as e:
        return VideoProcessingResult(str(video_path), label, label_name, 0, 0, 0, None, False, str(e))


def process_all_videos(normal_dir: Path, shoplifting_dir: Path, frame_skip: int = FRAME_SKIP,
                       max_videos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[VideoProcessingResult]]:
    print("\n[1/4] Discovering videos...")
    normal_videos = discover_videos(normal_dir, max_videos)
    shoplifting_videos = discover_videos(shoplifting_dir, max_videos)
    print(f"  Normal: {len(normal_videos)}, Shoplifting: {len(shoplifting_videos)}")
    if not normal_videos or not shoplifting_videos:
        raise ValueError("No videos found")

    print("\n[2/4] Initializing processors...")
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor(window_size=SLIDING_WINDOW_SIZE, window_stride=SLIDING_WINDOW_STRIDE)

    print("\n[3/4] Processing videos...")
    all_results, all_features, all_labels = [], [], []
    total = len(normal_videos) + len(shoplifting_videos)
    count = 0

    for video_path in normal_videos:
        count += 1
        print(f"  [{count}/{total}] {video_path.name}...", end=" ", flush=True)
        result = process_single_video(video_path, 0, "normal", pose_estimator, feature_extractor, frame_skip)
        all_results.append(result)
        if result.success and result.features is not None:
            all_features.append(result.features)
            all_labels.extend([0] * len(result.features))
            print(f"OK ({result.n_feature_windows} windows)")
        else:
            print(f"FAILED: {result.error_message}")

    for video_path in shoplifting_videos:
        count += 1
        print(f"  [{count}/{total}] {video_path.name}...", end=" ", flush=True)
        result = process_single_video(video_path, 1, "shoplifting", pose_estimator, feature_extractor, frame_skip)
        all_results.append(result)
        if result.success and result.features is not None:
            all_features.append(result.features)
            all_labels.extend([1] * len(result.features))
            print(f"OK ({result.n_feature_windows} windows)")
        else:
            print(f"FAILED: {result.error_message}")

    pose_estimator.close()
    print("\n[4/4] Compiling features...")
    if not all_features:
        raise ValueError("No features extracted")
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    print(f"  Total: {len(labels)} samples ({np.sum(labels==0)} normal, {np.sum(labels==1)} shoplifting)")
    return features, labels, all_results


def train_and_evaluate(features: np.ndarray, labels: np.ndarray) -> Tuple[RandomForestClassifier, StandardScaler, TrainingMetrics]:
    print("\n[1/3] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    print("\n[2/3] Training model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
                                   random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    print("\n[3/3] Evaluating...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    y_pred = model.predict(X_test_scaled)

    feature_extractor = FeatureExtractor()
    importance = dict(zip(feature_extractor.feature_names, model.feature_importances_))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    metrics = TrainingMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average='weighted'),
        recall=recall_score(y_test, y_pred, average='weighted'),
        f1_score=f1_score(y_test, y_pred, average='weighted'),
        cv_accuracy_mean=float(cv_scores.mean()),
        cv_accuracy_std=float(cv_scores.std()),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
        n_features=features.shape[1],
        feature_importance=sorted_importance
    )
    print(f"\n  Accuracy: {metrics.accuracy:.2%}, CV: {metrics.cv_accuracy_mean:.2%} +/- {metrics.cv_accuracy_std:.2%}")
    return model, scaler, metrics


def save_model_and_results(model: RandomForestClassifier, scaler: StandardScaler, metrics: TrainingMetrics,
                           processing_results: List[VideoProcessingResult], output_dir: Path = MODELS_DIR) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "behavior_classifier.pkl"
    joblib.dump({'model': model, 'scaler': scaler, 'feature_mean': scaler.mean_, 'feature_std': scaler.scale_, 'classes': TRAINING_CLASSES}, model_path)
    print(f"  Model saved: {model_path}")

    info_path = output_dir / "behavior_classifier_info.json"
    training_info = {
        'training_date': datetime.now().isoformat(),
        'classes': TRAINING_CLASSES,
        'metrics': {'accuracy': metrics.accuracy, 'precision': metrics.precision, 'recall': metrics.recall,
                    'f1_score': metrics.f1_score, 'cv_accuracy_mean': metrics.cv_accuracy_mean, 'cv_accuracy_std': metrics.cv_accuracy_std},
        'confusion_matrix': metrics.confusion_matrix,
        'n_train_samples': metrics.n_train_samples,
        'n_test_samples': metrics.n_test_samples,
        'n_features': metrics.n_features,
        'feature_importance': metrics.feature_importance,
        'model_params': {'n_estimators': model.n_estimators, 'max_depth': model.max_depth},
        'processing_config': {'window_size': SLIDING_WINDOW_SIZE, 'window_stride': SLIDING_WINDOW_STRIDE, 'frame_skip': FRAME_SKIP}
    }
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"  Info saved: {info_path}")
    return {'model': model_path, 'info': info_path}


def main():
    print("\n" + "=" * 60)
    print("  DIGITAL WITNESS - Behavior Classifier Training")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        features, labels, processing_results = process_all_videos(NORMAL_VIDEO_DIR, SHOPLIFTING_VIDEO_DIR)
        model, scaler, metrics = train_and_evaluate(features, labels)
        saved_paths = save_model_and_results(model, scaler, metrics, processing_results)
        print(f"\n  TRAINING COMPLETE - Accuracy: {metrics.accuracy:.2%}")
        return {'metrics': metrics, 'model_path': saved_paths['model'], 'success': True}
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    main()
