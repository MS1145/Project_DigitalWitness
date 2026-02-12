"""
Optimized deep learning pipeline for Digital Witness.

Pipeline: YOLO26n -> Conditional Processing -> CNN -> LSTM

Optimizations:
- Frame skipping (configurable)
- Motion detection (skip static frames)
- Person gate (skip frames without people)
- Scene validation (detect non-retail footage)
- Batch processing for GPU efficiency
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..detection.yolo_detector import YOLODetector, Detection, InteractionEvent
from ..detection.tracker import ObjectTracker, TrackedObject
from .cnn_feature_extractor import CNNFeatureExtractor, FrameFeatures
from .lstm_classifier import LSTMIntentClassifier, IntentPrediction
from ..config import (
    LSTM_SEQUENCE_LENGTH,
    SLIDING_WINDOW_STRIDE,
    FRAME_SKIP,
    USE_MOTION_DETECTION,
    USE_PERSON_GATE,
    USE_SCENE_VALIDATION,
    MOTION_THRESHOLD,
    PRODUCT_CLASSES,
    CNN_BACKBONE
)


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    interactions: List[InteractionEvent]
    features: Optional[np.ndarray] = None


@dataclass
class PipelineStats:
    """Processing statistics for optimization tracking."""
    total_frames: int = 0
    processed_frames: int = 0
    skipped_no_motion: int = 0
    skipped_no_person: int = 0
    skipped_frame_skip: int = 0

    @property
    def skip_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return 1.0 - (self.processed_frames / self.total_frames)

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Processed: {self.processed_frames}/{self.total_frames} frames "
            f"({100-self.skip_rate*100:.1f}% processed)\n"
            f"  - Skipped (no motion): {self.skipped_no_motion}\n"
            f"  - Skipped (no person): {self.skipped_no_person}"
        )


@dataclass
class SceneValidationResult:
    """Result of scene validation check."""
    is_valid: bool
    is_retail_scene: bool
    has_person: bool
    has_products: bool
    detected_classes: set
    message: str

    @staticmethod
    def invalid(reason: str) -> 'SceneValidationResult':
        return SceneValidationResult(
            is_valid=False,
            is_retail_scene=False,
            has_person=False,
            has_products=False,
            detected_classes=set(),
            message=reason
        )


@dataclass
class DeepPipelineResult:
    """Complete results from deep learning pipeline."""
    video_path: str
    total_frames: int
    processed_frames: int
    duration: float
    fps: float

    # Detection results
    total_detections: int
    persons_tracked: int
    products_tracked: int
    total_interactions: int

    # Classification results
    intent_predictions: List[IntentPrediction]
    overall_intent: str
    overall_confidence: float

    # Suspicious activity
    suspicious_segments: List[Dict[str, Any]]
    interaction_timeline: List[InteractionEvent]

    # Optimization stats
    stats: Optional[PipelineStats] = None

    # Scene validation - WRONG VIDEO detection
    is_valid_scene: bool = True
    scene_validation: Optional[SceneValidationResult] = None
    is_static_video: bool = False  # No motion detected
    skip_summary: str = ""

    @property
    def is_wrong_video(self) -> bool:
        """Check if this video is flagged as wrong/invalid."""
        return not self.is_valid_scene or self.is_static_video

    @property
    def wrong_video_reason(self) -> str:
        """Get reason why video is flagged as wrong."""
        if self.is_static_video:
            return "WRONG VIDEO: Static footage - no motion detected throughout video"
        if not self.is_valid_scene and self.scene_validation:
            return self.scene_validation.message
        if not self.is_valid_scene:
            return "WRONG VIDEO: Scene validation failed"
        return ""

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "duration": self.duration,
            "fps": self.fps,
            "total_detections": self.total_detections,
            "persons_tracked": self.persons_tracked,
            "products_tracked": self.products_tracked,
            "total_interactions": self.total_interactions,
            "intent_predictions": [p.to_dict() for p in self.intent_predictions],
            "overall_intent": self.overall_intent,
            "overall_confidence": self.overall_confidence,
            "suspicious_segments": self.suspicious_segments,
            "is_valid_scene": self.is_valid_scene,
            "is_wrong_video": self.is_wrong_video,
            "wrong_video_reason": self.wrong_video_reason,
            "is_static_video": self.is_static_video,
            "skip_summary": self.skip_summary
        }


class ConditionalProcessor:
    """
    Implements conditional processing to skip unnecessary frames.

    Conditions:
    1. Motion Detection - Skip static frames (no movement = nothing to analyze)
    2. Person Gate - Only process if person detected (no person = no shoplifting)
    3. Scene Validation - Detect if video is NOT a retail store (wrong footage)

    These conditions can save 50-80% of processing time.
    """

    def __init__(self, motion_threshold: int = MOTION_THRESHOLD):
        self.motion_threshold = motion_threshold
        self.prev_frame_gray = None
        self.stats = PipelineStats()
        self.scene_validation_result: Optional[SceneValidationResult] = None
        self.frames_checked_for_scene = 0
        self.scene_check_frames = 30  # Check first N frames for scene validation

    def reset(self):
        """Reset for new video."""
        self.prev_frame_gray = None
        self.stats = PipelineStats()
        self.scene_validation_result = None
        self.frames_checked_for_scene = 0

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Check if frame has significant motion compared to previous frame.

        Static footage (security camera with no activity) = skip processing.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return True  # First frame always has "motion"

        diff = cv2.absdiff(self.prev_frame_gray, gray)
        motion_score = np.sum(diff > 25)
        self.prev_frame_gray = gray

        return motion_score > self.motion_threshold

    def has_person(self, detections: List[Detection]) -> bool:
        """Check if any person is detected."""
        return any(d.class_name == "person" for d in detections)

    def validate_scene(self, detections: List[Detection]) -> SceneValidationResult:
        """
        Validate if this looks like a retail/store scene.

        WRONG VIDEO conditions:
        1. No person detected in first N frames → Not useful footage
        2. No retail products detected → Probably not a store

        This helps catch:
        - Outdoor footage
        - Office/home footage
        - Empty store footage
        - Completely wrong video files
        """
        classes_found = {d.class_name for d in detections}
        has_person = "person" in classes_found
        has_products = bool(classes_found & PRODUCT_CLASSES)

        # Determine if valid retail scene
        if not has_person and not has_products:
            return SceneValidationResult(
                is_valid=False,
                is_retail_scene=False,
                has_person=False,
                has_products=False,
                detected_classes=classes_found,
                message="WRONG VIDEO: No person or retail products detected. This may not be store footage."
            )

        if not has_person:
            return SceneValidationResult(
                is_valid=False,
                is_retail_scene=has_products,
                has_person=False,
                has_products=has_products,
                detected_classes=classes_found,
                message="WRONG VIDEO: No person detected. Cannot analyze behavior without people."
            )

        # Person detected - this is valid footage
        if has_products:
            message = f"Valid retail scene: Person + products detected ({classes_found & PRODUCT_CLASSES})"
        else:
            message = "Valid scene: Person detected (no products visible yet)"

        return SceneValidationResult(
            is_valid=True,
            is_retail_scene=has_products,
            has_person=True,
            has_products=has_products,
            detected_classes=classes_found,
            message=message
        )

    def check_scene_validity(self, detections: List[Detection]) -> Optional[SceneValidationResult]:
        """
        Accumulate detections over first N frames to validate scene.
        Returns result only after enough frames checked.
        """
        self.frames_checked_for_scene += 1

        # Accumulate all detected classes
        if self.scene_validation_result is None:
            self.scene_validation_result = self.validate_scene(detections)
        else:
            # Update with new detections
            current = self.validate_scene(detections)
            # If we find person or products, update
            if current.has_person or current.has_products:
                self.scene_validation_result = SceneValidationResult(
                    is_valid=current.has_person,  # Valid if person found
                    is_retail_scene=self.scene_validation_result.has_products or current.has_products,
                    has_person=self.scene_validation_result.has_person or current.has_person,
                    has_products=self.scene_validation_result.has_products or current.has_products,
                    detected_classes=self.scene_validation_result.detected_classes | current.detected_classes,
                    message=current.message if current.is_valid else self.scene_validation_result.message
                )

        # Return final result after checking enough frames
        if self.frames_checked_for_scene >= self.scene_check_frames:
            return self.scene_validation_result

        return None  # Still checking

    def is_static_video(self, sample_frames: int = 10) -> bool:
        """
        Check if video appears to be completely static (no motion at all).
        Called after processing some frames.
        """
        if self.stats.total_frames < sample_frames:
            return False

        # If ALL frames were skipped due to no motion, video is static
        motion_ratio = 1.0 - (self.stats.skipped_no_motion / self.stats.total_frames)
        return motion_ratio < 0.1  # Less than 10% frames have motion

    def should_process(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        use_motion: bool = True,
        use_person_gate: bool = True
    ) -> Tuple[bool, str]:
        """
        Determine if frame should be fully processed through CNN.

        Returns:
            (should_process, reason)

        Reasons:
            - "process": Frame should be processed
            - "no_motion": Skipped - static frame, no movement
            - "no_person": Skipped - no person detected
        """
        self.stats.total_frames += 1

        # Check 1: Motion detection (cheapest - just pixel diff)
        if use_motion and not self.has_motion(frame):
            self.stats.skipped_no_motion += 1
            return False, "no_motion"

        # Check 2: Person gate (requires YOLO detections)
        if use_person_gate and detections is not None:
            if not self.has_person(detections):
                self.stats.skipped_no_person += 1
                return False, "no_person"

        self.stats.processed_frames += 1
        return True, "process"

    def get_skip_summary(self) -> str:
        """Get human-readable skip summary."""
        total = self.stats.total_frames
        if total == 0:
            return "No frames processed yet"

        lines = [
            f"Frame Processing Summary:",
            f"  Total frames checked: {total}",
            f"  Processed through CNN: {self.stats.processed_frames} ({self.stats.processed_frames/total*100:.1f}%)",
            f"  Skipped (no motion): {self.stats.skipped_no_motion} ({self.stats.skipped_no_motion/total*100:.1f}%)",
            f"  Skipped (no person): {self.stats.skipped_no_person} ({self.stats.skipped_no_person/total*100:.1f}%)",
            f"  Compute savings: {self.stats.skip_rate*100:.1f}%"
        ]
        return "\n".join(lines)


class DeepPipeline:
    """
    Optimized YOLO -> CNN -> LSTM pipeline.

    Features:
    - Conditional processing (motion, person gate)
    - Configurable frame skipping
    - Multiple CNN backbone support
    - Scene validation
    """

    def __init__(
        self,
        yolo_model: Optional[str] = None,
        cnn_backbone: str = CNN_BACKBONE,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        stride: int = SLIDING_WINDOW_STRIDE,
        frame_skip: int = FRAME_SKIP,
        use_motion_detection: bool = USE_MOTION_DETECTION,
        use_person_gate: bool = USE_PERSON_GATE,
        device: str = "auto"
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.frame_skip = frame_skip
        self.use_motion_detection = use_motion_detection
        self.use_person_gate = use_person_gate
        self.device = device

        # Initialize components
        self.detector = YOLODetector(model_path=yolo_model, device=device)
        self.tracker = ObjectTracker()
        self.cnn = CNNFeatureExtractor(backbone=cnn_backbone, device=device)
        self.lstm = LSTMIntentClassifier(
            input_dim=self.cnn.feature_dim,
            device=device
        )
        self.processor = ConditionalProcessor()

        self._initialized = False

    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return

        self.detector.initialize()
        self.cnn.initialize()

        # Load trained LSTM if available
        from ..config import MODELS_DIR
        lstm_path = MODELS_DIR / "lstm_classifier.pt"

        if lstm_path.exists():
            self.lstm.load_model(lstm_path)
        else:
            print("  Warning: No trained LSTM model found")
            self.lstm.initialize()

        self._initialized = True

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None,
        store_frame_analyses: bool = False,
        early_exit_on_invalid: bool = False
    ) -> DeepPipelineResult:
        """
        Process video through optimized pipeline.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(progress, message)
            store_frame_analyses: Store detailed per-frame analysis
            early_exit_on_invalid: Exit early if scene validation fails (wrong video)

        Returns:
            DeepPipelineResult with all outputs

        Conditions that skip processing:
        1. Static video (no motion) - flagged as "wrong video"
        2. No person detected - frames skipped
        3. Not a retail scene - flagged as "wrong video"
        """
        self.initialize()
        self.tracker.reset()
        self.detector.reset_tracking()
        self.processor.reset()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Storage
        frame_analyses = [] if store_frame_analyses else None
        all_features = []
        frame_count = 0
        scene_validated = False

        if progress_callback:
            progress_callback(0.0, "Starting optimized analysis...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # Frame skipping
            if frame_count % self.frame_skip != 0:
                continue

            # Run YOLO detection first (needed for person gate & scene validation)
            detections = self.detector.detect_with_tracking(
                frame, frame_count, timestamp
            )

            # Scene validation - check first N frames for retail scene
            if USE_SCENE_VALIDATION and not scene_validated:
                validation_result = self.processor.check_scene_validity(detections)
                if validation_result is not None:
                    scene_validated = True
                    if not validation_result.is_valid:
                        if progress_callback:
                            progress_callback(0.1, f"Scene check: {validation_result.message}")

                        # Early exit if requested and scene is invalid
                        if early_exit_on_invalid:
                            cap.release()
                            return self._create_invalid_result(
                                video_path, total_frames, duration, fps,
                                validation_result, "Scene validation failed"
                            )

            # Conditional processing check
            should_process, reason = self.processor.should_process(
                frame,
                detections=detections,
                use_motion=self.use_motion_detection,
                use_person_gate=self.use_person_gate
            )

            if not should_process:
                # Check if video appears completely static
                if self.processor.is_static_video(sample_frames=30):
                    if progress_callback:
                        progress_callback(0.1, "WRONG VIDEO: Static footage detected")
                    if early_exit_on_invalid:
                        cap.release()
                        return self._create_invalid_result(
                            video_path, total_frames, duration, fps,
                            None, "Static video - no motion detected"
                        )
                continue

            # Update tracker
            active_tracks = self.tracker.update(detections, timestamp)

            # Detect interactions
            persons = [d for d in detections if d.class_name == "person"]
            products = [d for d in detections if d.class_name != "person"]
            interactions = self.detector.detect_interactions(
                persons, products, frame_count, timestamp
            )

            if interactions:
                self.tracker.update(detections, timestamp, interactions)

            # Extract CNN features
            frame_features = self.cnn.extract_features(frame, frame_count, timestamp)
            all_features.append(frame_features.features)

            # Store frame analysis
            if store_frame_analyses:
                frame_analyses.append(FrameAnalysis(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    detections=detections,
                    tracked_objects=active_tracks,
                    interactions=interactions,
                    features=frame_features.features
                ))

            # Progress
            if progress_callback and frame_count % 100 == 0:
                progress = frame_count / total_frames
                progress_callback(progress * 0.8, f"Frame {frame_count}/{total_frames}")

        cap.release()

        # Check for static video after processing
        is_static = self.processor.is_static_video()

        if progress_callback:
            progress_callback(0.85, "Running LSTM classification...")

        # LSTM classification
        intent_predictions = self._classify_sequences(all_features, fps)

        # Generate results
        tracking_summary = self.tracker.get_tracking_summary()
        overall_intent, overall_confidence = self._aggregate_predictions(intent_predictions)
        suspicious_segments = self._find_suspicious_segments(
            intent_predictions,
            self.tracker.interaction_history
        )

        # Determine scene validity
        scene_result = self.processor.scene_validation_result
        is_valid = scene_result.is_valid if scene_result else True

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return DeepPipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=self.processor.stats.processed_frames,
            duration=duration,
            fps=fps,
            total_detections=tracking_summary["total_tracks"],
            persons_tracked=tracking_summary["persons_tracked"],
            products_tracked=tracking_summary["products_tracked"],
            total_interactions=tracking_summary["total_interactions"],
            intent_predictions=intent_predictions,
            overall_intent=overall_intent,
            overall_confidence=overall_confidence,
            suspicious_segments=suspicious_segments,
            interaction_timeline=self.tracker.interaction_history,
            stats=self.processor.stats,
            is_valid_scene=is_valid,
            scene_validation=scene_result,
            is_static_video=is_static,
            skip_summary=self.processor.get_skip_summary()
        )

    def _create_invalid_result(
        self,
        video_path: str,
        total_frames: int,
        duration: float,
        fps: float,
        scene_result: Optional[SceneValidationResult],
        reason: str
    ) -> DeepPipelineResult:
        """Create result for invalid/wrong video."""
        return DeepPipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=0,
            duration=duration,
            fps=fps,
            total_detections=0,
            persons_tracked=0,
            products_tracked=0,
            total_interactions=0,
            intent_predictions=[],
            overall_intent="invalid",
            overall_confidence=0.0,
            suspicious_segments=[],
            interaction_timeline=[],
            stats=self.processor.stats,
            is_valid_scene=False,
            scene_validation=scene_result or SceneValidationResult.invalid(reason),
            is_static_video="static" in reason.lower(),
            skip_summary=f"WRONG VIDEO: {reason}"
        )

    def _classify_sequences(
        self,
        features: List[np.ndarray],
        fps: float
    ) -> List[IntentPrediction]:
        """Classify feature sequences using LSTM."""
        if not features:
            return []

        predictions = []
        feature_array = np.array(features)
        num_frames = len(features)

        # Adjust sequence length for frame skipping
        adjusted_seq_len = max(1, self.sequence_length // self.frame_skip)
        adjusted_stride = max(1, self.stride // self.frame_skip)

        if num_frames < adjusted_seq_len:
            # Pad short videos
            padding = np.zeros((adjusted_seq_len - num_frames, feature_array.shape[1]))
            padded = np.vstack([feature_array, padding])

            pred = self.lstm.predict_sequence(
                padded,
                sequence_id="seq_0_padded",
                start_time=0.0,
                end_time=num_frames * self.frame_skip / fps
            )
            pred.confidence *= (num_frames / adjusted_seq_len)
            predictions.append(pred)
            return predictions

        # Sliding window
        for start in range(0, num_frames - adjusted_seq_len + 1, adjusted_stride):
            end = start + adjusted_seq_len
            sequence = feature_array[start:end]

            start_time = start * self.frame_skip / fps
            end_time = end * self.frame_skip / fps

            pred = self.lstm.predict_sequence(
                sequence,
                sequence_id=f"seq_{start}_{end}",
                start_time=start_time,
                end_time=end_time
            )
            predictions.append(pred)

        return predictions

    def _aggregate_predictions(
        self,
        predictions: List[IntentPrediction]
    ) -> Tuple[str, float]:
        """Aggregate predictions to overall intent."""
        if not predictions:
            return "normal", 0.0

        class_counts = {}
        class_confidences = {}

        for pred in predictions:
            cls = pred.intent_class
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in class_confidences:
                class_confidences[cls] = []
            class_confidences[cls].append(pred.confidence)

        # Weight suspicious classes higher
        suspicious_classes = {"shoplifting", "concealment", "bypass", "pickup"}
        weighted_counts = {}

        for cls, count in class_counts.items():
            weight = 2.0 if cls in suspicious_classes else 1.0
            weighted_counts[cls] = count * weight

        total = sum(weighted_counts.values())
        if total == 0:
            return "normal", 0.0

        # Check suspicious threshold
        suspicious_score = sum(
            weighted_counts.get(cls, 0) for cls in suspicious_classes
        ) / total

        if suspicious_score > 0.3:
            suspicious_preds = [p for p in predictions if p.intent_class in suspicious_classes]
            if suspicious_preds:
                best = max(suspicious_preds, key=lambda p: p.confidence)
                return best.intent_class, best.confidence

        # Return dominant class
        overall_class = max(class_counts, key=class_counts.get)
        avg_confidence = np.mean(class_confidences[overall_class])
        return overall_class, float(avg_confidence)

    def _find_suspicious_segments(
        self,
        predictions: List[IntentPrediction],
        interactions: List[InteractionEvent]
    ) -> List[Dict[str, Any]]:
        """Identify suspicious time segments."""
        suspicious = []

        for pred in predictions:
            if pred.intent_class in {"shoplifting", "concealment", "bypass"}:
                if pred.confidence > 0.5:
                    suspicious.append({
                        "type": "behavior",
                        "class": pred.intent_class,
                        "confidence": pred.confidence,
                        "start_time": pred.start_time,
                        "end_time": pred.end_time,
                        "source": "lstm"
                    })

        for interaction in interactions:
            if interaction.interaction_type in {"conceal", "pickup"}:
                if interaction.confidence > 0.6:
                    suspicious.append({
                        "type": "interaction",
                        "class": interaction.interaction_type,
                        "confidence": interaction.confidence,
                        "start_time": interaction.timestamp,
                        "end_time": interaction.timestamp + 1.0,
                        "source": "yolo"
                    })

        suspicious.sort(key=lambda x: x["start_time"])
        return suspicious

    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.detector.reset_tracking()
        self.processor.reset()

    def close(self):
        """Release resources."""
        self.detector.close()
        self.cnn.close()
        self.lstm.close()
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
