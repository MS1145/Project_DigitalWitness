"""
Quality analysis module for Digital Witness.

Analyzes video and pose estimation quality to determine if a video
is suitable for reliable behavior classification. Detects:
- Low pose detection rates
- Occlusion (partial body visibility)
- Low confidence pose estimates
- Overall video quality issues

This enables the system to flag unreliable analyses and avoid
false positives/negatives from poor quality input.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from ..pose.estimator import PoseResult
from ..config import (
    QUALITY_MIN_POSE_CONFIDENCE,
    QUALITY_MIN_DETECTION_RATE,
    QUALITY_OCCLUSION_THRESHOLD,
    QUALITY_MIN_VISIBLE_LANDMARKS
)


@dataclass
class FrameQualityMetrics:
    """Quality metrics for a single frame."""
    frame_number: int
    timestamp: float
    pose_detected: bool
    pose_confidence: float          # Average landmark visibility
    visible_landmarks: int          # Number of visible landmarks
    total_landmarks: int            # Total expected landmarks
    occlusion_score: float          # 0.0 = fully visible, 1.0 = fully occluded
    overall_quality: float          # Combined quality score [0, 1]
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class VideoQualityReport:
    """Aggregated quality report for an entire video analysis."""
    total_frames: int
    frames_with_pose: int
    pose_detection_rate: float              # Ratio of frames with detected pose
    average_pose_confidence: float          # Mean confidence across all frames
    average_occlusion_score: float          # Mean occlusion across all frames
    occlusion_segments: List[Tuple[float, float]]   # Time ranges with occlusion
    low_quality_segments: List[Tuple[float, float]] # Time ranges with low quality
    low_confidence_segments: List[Tuple[float, float]]  # Time ranges with low confidence
    quality_flags: List[str]                # Summary warnings
    overall_quality_score: float            # Combined video quality [0, 1]
    usable_for_analysis: bool               # Whether video meets minimum quality
    frame_metrics: List[FrameQualityMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to dictionary for serialization."""
        return {
            "total_frames": self.total_frames,
            "frames_with_pose": self.frames_with_pose,
            "pose_detection_rate": self.pose_detection_rate,
            "average_pose_confidence": self.average_pose_confidence,
            "average_occlusion_score": self.average_occlusion_score,
            "occlusion_segments": self.occlusion_segments,
            "low_quality_segments": self.low_quality_segments,
            "low_confidence_segments": self.low_confidence_segments,
            "quality_flags": self.quality_flags,
            "overall_quality_score": self.overall_quality_score,
            "usable_for_analysis": self.usable_for_analysis
        }


class QualityAnalyzer:
    """
    Analyzes pose estimation quality to assess reliability of behavior classification.

    Quality analysis is performed after pose estimation and before feature extraction
    to identify problematic video segments that may lead to unreliable predictions.
    """

    def __init__(
        self,
        min_pose_confidence: float = QUALITY_MIN_POSE_CONFIDENCE,
        min_detection_rate: float = QUALITY_MIN_DETECTION_RATE,
        occlusion_threshold: float = QUALITY_OCCLUSION_THRESHOLD,
        min_visible_landmarks: int = QUALITY_MIN_VISIBLE_LANDMARKS
    ):
        """
        Initialize quality analyzer with configurable thresholds.

        Args:
            min_pose_confidence: Minimum average landmark visibility to consider reliable
            min_detection_rate: Minimum ratio of frames with detected poses
            occlusion_threshold: Occlusion score above which frame is considered occluded
            min_visible_landmarks: Minimum visible landmarks for valid pose
        """
        self.min_pose_confidence = min_pose_confidence
        self.min_detection_rate = min_detection_rate
        self.occlusion_threshold = occlusion_threshold
        self.min_visible_landmarks = min_visible_landmarks

    def analyze_frame(self, pose_result: PoseResult) -> FrameQualityMetrics:
        """
        Analyze quality of a single frame's pose estimation.

        Args:
            pose_result: PoseResult from pose estimator

        Returns:
            FrameQualityMetrics with detailed quality breakdown
        """
        quality_issues = []

        if pose_result.landmarks is None:
            return FrameQualityMetrics(
                frame_number=pose_result.frame_number,
                timestamp=pose_result.timestamp,
                pose_detected=False,
                pose_confidence=0.0,
                visible_landmarks=0,
                total_landmarks=0,
                occlusion_score=1.0,
                overall_quality=0.0,
                quality_issues=["No pose detected"]
            )

        # Calculate visibility metrics
        landmarks = pose_result.landmarks
        total_landmarks = len(landmarks)
        visibilities = [lm.visibility for lm in landmarks.values()]

        # Count visible landmarks (visibility > 0.5)
        visible_count = sum(1 for v in visibilities if v > 0.5)
        avg_confidence = np.mean(visibilities)

        # Calculate occlusion score (inverse of visibility ratio)
        visibility_ratio = visible_count / total_landmarks if total_landmarks > 0 else 0
        occlusion_score = 1.0 - visibility_ratio

        # Identify quality issues
        if avg_confidence < self.min_pose_confidence:
            quality_issues.append(f"Low confidence: {avg_confidence:.2f}")

        if visible_count < self.min_visible_landmarks:
            quality_issues.append(f"Low landmark visibility: {visible_count}/{total_landmarks}")

        if occlusion_score > self.occlusion_threshold:
            quality_issues.append(f"Significant occlusion: {occlusion_score:.2f}")

        # Check for specific body part occlusions important for behavior detection
        critical_parts = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']
        critical_visible = sum(
            1 for part in critical_parts
            if part in landmarks and landmarks[part].visibility > 0.5
        )
        if critical_visible < len(critical_parts):
            quality_issues.append(f"Critical landmarks occluded: {len(critical_parts) - critical_visible}")

        # Calculate overall quality score
        confidence_factor = min(1.0, avg_confidence / self.min_pose_confidence)
        visibility_factor = min(1.0, visible_count / self.min_visible_landmarks)
        occlusion_factor = max(0.0, 1.0 - occlusion_score / self.occlusion_threshold)

        overall_quality = (confidence_factor * 0.4 +
                         visibility_factor * 0.3 +
                         occlusion_factor * 0.3)

        return FrameQualityMetrics(
            frame_number=pose_result.frame_number,
            timestamp=pose_result.timestamp,
            pose_detected=True,
            pose_confidence=avg_confidence,
            visible_landmarks=visible_count,
            total_landmarks=total_landmarks,
            occlusion_score=occlusion_score,
            overall_quality=overall_quality,
            quality_issues=quality_issues
        )

    def analyze_sequence(
        self,
        pose_results: List[PoseResult],
        fps: float = 30.0
    ) -> VideoQualityReport:
        """
        Analyze quality of an entire pose sequence.

        Args:
            pose_results: List of PoseResult objects from pose estimation
            fps: Video frames per second for timestamp calculation

        Returns:
            VideoQualityReport with overall quality assessment
        """
        if not pose_results:
            return VideoQualityReport(
                total_frames=0,
                frames_with_pose=0,
                pose_detection_rate=0.0,
                average_pose_confidence=0.0,
                average_occlusion_score=1.0,
                occlusion_segments=[],
                low_quality_segments=[],
                low_confidence_segments=[],
                quality_flags=["No frames to analyze"],
                overall_quality_score=0.0,
                usable_for_analysis=False,
                frame_metrics=[]
            )

        # Analyze each frame
        frame_metrics = [self.analyze_frame(pr) for pr in pose_results]

        # Calculate aggregate statistics
        total_frames = len(frame_metrics)
        frames_with_pose = sum(1 for fm in frame_metrics if fm.pose_detected)
        detection_rate = frames_with_pose / total_frames

        # Calculate averages only for frames with detected poses
        detected_frames = [fm for fm in frame_metrics if fm.pose_detected]

        if detected_frames:
            avg_confidence = np.mean([fm.pose_confidence for fm in detected_frames])
            avg_occlusion = np.mean([fm.occlusion_score for fm in detected_frames])
        else:
            avg_confidence = 0.0
            avg_occlusion = 1.0

        # Find problematic segments
        occlusion_segments = self._find_segments(
            frame_metrics,
            lambda fm: fm.occlusion_score > self.occlusion_threshold
        )

        low_quality_segments = self._find_segments(
            frame_metrics,
            lambda fm: fm.overall_quality < 0.5
        )

        low_confidence_segments = self._find_segments(
            frame_metrics,
            lambda fm: fm.pose_confidence < self.min_pose_confidence and fm.pose_detected
        )

        # Generate quality flags
        quality_flags = self._generate_quality_flags(
            detection_rate,
            avg_confidence,
            avg_occlusion,
            occlusion_segments,
            low_quality_segments
        )

        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(
            detection_rate,
            avg_confidence,
            avg_occlusion,
            len(low_quality_segments),
            total_frames
        )

        # Determine if video is usable
        usable = (
            detection_rate >= self.min_detection_rate and
            avg_confidence >= self.min_pose_confidence * 0.8 and
            overall_quality >= 0.4
        )

        return VideoQualityReport(
            total_frames=total_frames,
            frames_with_pose=frames_with_pose,
            pose_detection_rate=detection_rate,
            average_pose_confidence=avg_confidence,
            average_occlusion_score=avg_occlusion,
            occlusion_segments=occlusion_segments,
            low_quality_segments=low_quality_segments,
            low_confidence_segments=low_confidence_segments,
            quality_flags=quality_flags,
            overall_quality_score=overall_quality,
            usable_for_analysis=usable,
            frame_metrics=frame_metrics
        )

    def _find_segments(
        self,
        frame_metrics: List[FrameQualityMetrics],
        condition: callable
    ) -> List[Tuple[float, float]]:
        """
        Find contiguous time segments where condition is true.

        Args:
            frame_metrics: List of frame quality metrics
            condition: Function that returns True for problematic frames

        Returns:
            List of (start_time, end_time) tuples
        """
        segments = []
        in_segment = False
        segment_start = 0.0

        for fm in frame_metrics:
            if condition(fm):
                if not in_segment:
                    in_segment = True
                    segment_start = fm.timestamp
            else:
                if in_segment:
                    in_segment = False
                    segments.append((segment_start, fm.timestamp))

        # Close final segment if still open
        if in_segment and frame_metrics:
            segments.append((segment_start, frame_metrics[-1].timestamp))

        return segments

    def _generate_quality_flags(
        self,
        detection_rate: float,
        avg_confidence: float,
        avg_occlusion: float,
        occlusion_segments: List[Tuple[float, float]],
        low_quality_segments: List[Tuple[float, float]]
    ) -> List[str]:
        """Generate human-readable quality warning flags."""
        flags = []

        if detection_rate < self.min_detection_rate:
            flags.append(f"LOW_DETECTION_RATE: Only {detection_rate:.1%} of frames have detected poses")

        if avg_confidence < self.min_pose_confidence:
            flags.append(f"LOW_CONFIDENCE: Average pose confidence {avg_confidence:.2f} below threshold {self.min_pose_confidence}")

        if avg_occlusion > self.occlusion_threshold:
            flags.append(f"HIGH_OCCLUSION: Average occlusion score {avg_occlusion:.2f} indicates significant body obstruction")

        if len(occlusion_segments) > 3:
            flags.append(f"FREQUENT_OCCLUSION: {len(occlusion_segments)} separate occlusion events detected")

        if len(low_quality_segments) > 0:
            total_low_quality_time = sum(end - start for start, end in low_quality_segments)
            flags.append(f"LOW_QUALITY_SEGMENTS: {total_low_quality_time:.1f}s of low quality footage")

        if not flags:
            flags.append("QUALITY_OK: Video meets minimum quality requirements")

        return flags

    def _calculate_overall_quality(
        self,
        detection_rate: float,
        avg_confidence: float,
        avg_occlusion: float,
        low_quality_segment_count: int,
        total_frames: int
    ) -> float:
        """Calculate weighted overall quality score."""
        # Normalize factors to [0, 1] range
        detection_factor = min(1.0, detection_rate / self.min_detection_rate)
        confidence_factor = min(1.0, avg_confidence / self.min_pose_confidence) if avg_confidence > 0 else 0
        occlusion_factor = max(0.0, 1.0 - avg_occlusion)

        # Penalize for many low quality segments
        segment_penalty = min(0.3, low_quality_segment_count * 0.05)

        # Weighted combination
        quality_score = (
            detection_factor * 0.35 +
            confidence_factor * 0.35 +
            occlusion_factor * 0.30 -
            segment_penalty
        )

        return max(0.0, min(1.0, quality_score))

    def get_reliable_segments(
        self,
        quality_report: VideoQualityReport,
        min_quality: float = 0.6
    ) -> List[Tuple[float, float]]:
        """
        Get time segments with acceptable quality for analysis.

        Args:
            quality_report: VideoQualityReport from analyze_sequence
            min_quality: Minimum frame quality score to include

        Returns:
            List of (start_time, end_time) tuples for reliable segments
        """
        return self._find_segments(
            quality_report.frame_metrics,
            lambda fm: fm.overall_quality >= min_quality
        )

    def get_quality_summary(self, quality_report: VideoQualityReport) -> str:
        """Generate human-readable quality summary."""
        lines = [
            "VIDEO QUALITY ANALYSIS",
            "=" * 40,
            "",
            f"Frames analyzed: {quality_report.total_frames}",
            f"Frames with pose: {quality_report.frames_with_pose} ({quality_report.pose_detection_rate:.1%})",
            f"Average confidence: {quality_report.average_pose_confidence:.2f}",
            f"Average occlusion: {quality_report.average_occlusion_score:.2f}",
            f"Overall quality: {quality_report.overall_quality_score:.2f}",
            "",
            "Quality Flags:",
        ]

        for flag in quality_report.quality_flags:
            lines.append(f"  - {flag}")

        lines.append("")

        if quality_report.usable_for_analysis:
            lines.append("STATUS: Video is USABLE for behavior analysis")
        else:
            lines.append("STATUS: Video quality is TOO LOW for reliable analysis")
            lines.append("        Results should be treated with low confidence")

        return "\n".join(lines)
