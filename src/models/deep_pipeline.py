"""
Deep learning pipeline orchestration for Digital Witness.

Coordinates the YOLO -> CNN -> LSTM flow for end-to-end
video analysis using deep learning models.

Architecture Overview:
----------------------
1. YOLO (Object Detection) - Detects persons and products in each frame
2. Tracker (ByteTrack) - Maintains object identity across frames
3. CNN (ResNet18) - Extracts 512-dim spatial features per frame
4. LSTM (Bidirectional + Attention) - Classifies temporal behavior sequences

Data Flow:
----------
Video Frame → YOLO Detection → Object Tracking → CNN Features → LSTM Classification
                    ↓                 ↓                              ↓
              Person/Product    Track IDs           Intent: normal/pickup/concealment/bypass
              Bounding Boxes    Maintained
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Import deep learning components
from ..detection.yolo_detector import YOLODetector, Detection, InteractionEvent
from ..detection.tracker import ObjectTracker, TrackedObject
from .cnn_feature_extractor import CNNFeatureExtractor, SequenceFeatures
from .lstm_classifier import LSTMIntentClassifier, IntentPrediction
from ..config import (
    LSTM_SEQUENCE_LENGTH,    # Default: 30 frames per sequence (~1 sec at 30fps)
    SLIDING_WINDOW_STRIDE    # Default: 15 frames overlap between sequences
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

    # Suspicious activity summary
    suspicious_segments: List[Dict[str, Any]]
    interaction_timeline: List[InteractionEvent]

    # Frame-level analysis (optional, for detailed review)
    frame_analyses: Optional[List[FrameAnalysis]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
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
            "interaction_timeline": [i.to_dict() for i in self.interaction_timeline]
        }


class DeepPipeline:
    """
    End-to-end deep learning pipeline: YOLO -> CNN -> LSTM.

    Orchestrates:
    1. YOLO detection of persons and products
    2. Multi-object tracking
    3. Interaction detection
    4. CNN feature extraction
    5. LSTM temporal classification
    """

    def __init__(
        self,
        yolo_model: Optional[str] = None,
        cnn_backbone: str = "resnet18",
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        stride: int = SLIDING_WINDOW_STRIDE,
        device: str = "auto"
    ):
        """
        Initialize deep pipeline.

        Args:
            yolo_model: Path to YOLO model (None for pretrained)
            cnn_backbone: CNN backbone architecture
            sequence_length: Frames per LSTM sequence
            stride: Frames between sequences
            device: Device for inference
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device

        # Initialize components (lazy)
        self.detector = YOLODetector(model_path=yolo_model, device=device)
        self.tracker = ObjectTracker()
        self.cnn = CNNFeatureExtractor(backbone=cnn_backbone, device=device)
        self.lstm = LSTMIntentClassifier(device=device)

        self._initialized = False

    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return

        self.detector.initialize()
        self.cnn.initialize()
        self.lstm.initialize()
        self._initialized = True

    def process_video(
        self,
        video_path: str,
        frame_step: int = 1,
        progress_callback: Optional[callable] = None,
        store_frame_analyses: bool = False
    ) -> DeepPipelineResult:
        """
        Process a complete video through the deep pipeline.

        Args:
            video_path: Path to video file
            frame_step: Process every Nth frame
            progress_callback: Optional callback(progress, message)
            store_frame_analyses: Store detailed per-frame analysis

        Returns:
            DeepPipelineResult with all analysis outputs
        """
        import cv2

        self.initialize()
        self.tracker.reset()
        self.detector.reset_tracking()

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
        processed_count = 0

        if progress_callback:
            progress_callback(0.0, "Starting deep analysis...")

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # Skip frames based on step
            if frame_count % frame_step != 0:
                continue

            processed_count += 1

            # Step 1: YOLO detection with tracking
            detections = self.detector.detect_with_tracking(
                frame, frame_count, timestamp
            )

            # Step 2: Update tracker
            active_tracks = self.tracker.update(detections, timestamp)

            # Step 3: Detect interactions
            persons = [d for d in detections if d.class_name == "person"]
            products = [d for d in detections if d.class_name != "person"]
            interactions = self.detector.detect_interactions(
                persons, products, frame_count, timestamp
            )

            # Record interactions
            if interactions:
                self.tracker.update(detections, timestamp, interactions)

            # Step 4: Extract CNN features
            frame_features = self.cnn.extract_features(
                frame, frame_count, timestamp
            )
            all_features.append(frame_features.features)

            # Store frame analysis
            if store_frame_analyses:
                analysis = FrameAnalysis(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    detections=detections,
                    tracked_objects=active_tracks,
                    interactions=interactions,
                    features=frame_features.features
                )
                frame_analyses.append(analysis)

            # Progress update
            if progress_callback and processed_count % 50 == 0:
                progress = frame_count / total_frames
                progress_callback(
                    progress * 0.8,  # Reserve 20% for LSTM
                    f"Processing frame {frame_count}/{total_frames}..."
                )

        cap.release()

        if progress_callback:
            progress_callback(0.8, "Running temporal classification...")

        # Step 5: LSTM classification on feature sequences
        intent_predictions = self._classify_sequences(all_features, fps)

        if progress_callback:
            progress_callback(0.95, "Generating summary...")

        # Generate results
        tracking_summary = self.tracker.get_tracking_summary()

        # Determine overall intent
        overall_intent, overall_confidence = self._aggregate_predictions(
            intent_predictions
        )

        # Find suspicious segments
        suspicious_segments = self._find_suspicious_segments(
            intent_predictions,
            self.tracker.interaction_history
        )

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return DeepPipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_count,
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
            frame_analyses=frame_analyses
        )

    def _classify_sequences(
        self,
        features: List[np.ndarray],
        fps: float
    ) -> List[IntentPrediction]:
        """
        Classify feature sequences using LSTM with sliding window approach.

        The sliding window creates overlapping sequences to capture behavior
        transitions that might span window boundaries.

        Example with 100 frames, seq_length=30, stride=15:
        - Window 1: frames 0-29
        - Window 2: frames 15-44  (50% overlap)
        - Window 3: frames 30-59
        - ... continues until end of video
        """
        if not features:
            return []

        predictions = []
        feature_array = np.array(features)  # Shape: (num_frames, feature_dim)
        num_frames = len(features)

        # Create sliding windows with overlap for temporal continuity
        # This ensures behaviors spanning window edges are captured
        for start in range(0, num_frames - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            sequence = feature_array[start:end]  # Shape: (seq_length, feature_dim)

            # Convert frame indices to timestamps for timeline correlation
            start_time = start / fps
            end_time = end / fps

            # LSTM predicts intent class for this temporal window
            prediction = self.lstm.predict_sequence(
                sequence,
                sequence_id=f"seq_{start}_{end}",
                start_time=start_time,
                end_time=end_time
            )
            predictions.append(prediction)

        return predictions

    def _aggregate_predictions(
        self,
        predictions: List[IntentPrediction]
    ) -> Tuple[str, float]:
        """
        Aggregate sequence predictions to determine overall video intent.

        Strategy:
        ---------
        1. Count occurrences of each intent class across all windows
        2. Apply 2x weight to suspicious classes (concealment, bypass, shoplifting)
           to ensure even brief suspicious activity is not drowned out by normal behavior
        3. If suspicious activity exceeds 30% weighted presence, report the
           most confident suspicious prediction
        4. Otherwise, report the most common class with average confidence

        This weighted approach addresses the "needle in haystack" problem where
        a few seconds of shoplifting would be outvoted by minutes of normal behavior.
        """
        if not predictions:
            return "normal", 0.0

        # Count occurrences of each intent class
        class_counts = {}
        class_confidences = {}

        for pred in predictions:
            cls = pred.intent_class
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in class_confidences:
                class_confidences[cls] = []
            class_confidences[cls].append(pred.confidence)

        # Apply 2x weight to suspicious classes to prevent normal behavior
        # from drowning out brief but critical suspicious activity
        suspicious_weight = 2.0
        weighted_counts = {}

        for cls, count in class_counts.items():
            if cls in ["concealment", "bypass", "shoplifting"]:
                weighted_counts[cls] = count * suspicious_weight
            else:
                weighted_counts[cls] = count

        # Calculate normalized scores for each class
        total = sum(weighted_counts.values())
        if total == 0:
            return "normal", 0.0

        class_scores = {
            cls: weighted_counts.get(cls, 0) / total
            for cls in class_counts
        }

        # Determine overall intent with suspicious activity priority
        # 30% threshold: if suspicious activity makes up significant portion
        suspicious_classes = ["concealment", "bypass", "shoplifting"]
        suspicious_score = sum(
            class_scores.get(cls, 0) for cls in suspicious_classes
        )

        if suspicious_score > 0.3:
            # Report the highest confidence suspicious prediction
            # This ensures alerts are based on strongest evidence
            suspicious_preds = [
                p for p in predictions
                if p.intent_class in suspicious_classes
            ]
            if suspicious_preds:
                most_confident = max(suspicious_preds, key=lambda p: p.confidence)
                return most_confident.intent_class, most_confident.confidence

        # No significant suspicious activity - report dominant class
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

        # From predictions
        for pred in predictions:
            if pred.intent_class in ["concealment", "bypass", "shoplifting"]:
                if pred.confidence > 0.5:
                    suspicious.append({
                        "type": "behavior",
                        "class": pred.intent_class,
                        "confidence": pred.confidence,
                        "start_time": pred.start_time,
                        "end_time": pred.end_time,
                        "source": "lstm_prediction"
                    })

        # From interactions
        suspicious_interactions = ["conceal", "pickup"]
        for interaction in interactions:
            if interaction.interaction_type in suspicious_interactions:
                if interaction.confidence > 0.6:
                    suspicious.append({
                        "type": "interaction",
                        "class": interaction.interaction_type,
                        "confidence": interaction.confidence,
                        "start_time": interaction.timestamp,
                        "end_time": interaction.timestamp + 1.0,  # Approximate
                        "person_id": interaction.person_id,
                        "product": interaction.product_class,
                        "source": "interaction_detection"
                    })

        # Sort by time
        suspicious.sort(key=lambda x: x["start_time"])

        return suspicious

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> FrameAnalysis:
        """
        Process a single frame (for streaming/real-time use).

        Args:
            frame: BGR image
            frame_number: Frame number
            timestamp: Timestamp

        Returns:
            FrameAnalysis for this frame
        """
        self.initialize()

        # Detection
        detections = self.detector.detect_with_tracking(
            frame, frame_number, timestamp
        )

        # Tracking
        active_tracks = self.tracker.update(detections, timestamp)

        # Interactions
        persons = [d for d in detections if d.class_name == "person"]
        products = [d for d in detections if d.class_name != "person"]
        interactions = self.detector.detect_interactions(
            persons, products, frame_number, timestamp
        )

        # Features
        frame_features = self.cnn.extract_features(frame, frame_number, timestamp)

        return FrameAnalysis(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            tracked_objects=active_tracks,
            interactions=interactions,
            features=frame_features.features
        )

    def reset(self):
        """Reset pipeline state for new video."""
        self.tracker.reset()
        self.detector.reset_tracking()

    def close(self):
        """Release all resources."""
        self.detector.close()
        self.cnn.close()
        self.lstm.close()
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
