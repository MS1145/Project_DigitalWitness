"""
Edge case handler for Digital Witness.

Manages scenarios where standard analysis may be unreliable:
- Low pose confidence
- Significant occlusion
- Insufficient data
- Ambiguous classifications

The handler applies conservative adjustments to prevent false positives
from edge cases while clearly documenting confidence limitations.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from .quality_analyzer import VideoQualityReport, FrameQualityMetrics
from ..pose.behavior_classifier import BehaviorEvent
from ..config import (
    EDGE_CASE_MIN_CONFIDENCE,
    EDGE_CASE_MIN_QUALITY,
    EDGE_CASE_AMBIGUITY_THRESHOLD
)


class EdgeCaseType(Enum):
    """Types of edge cases that can affect analysis reliability."""
    LOW_POSE_CONFIDENCE = "low_pose_confidence"
    HIGH_OCCLUSION = "high_occlusion"
    INSUFFICIENT_DATA = "insufficient_data"
    AMBIGUOUS_CLASSIFICATION = "ambiguous_classification"
    POOR_VIDEO_QUALITY = "poor_video_quality"
    INTERMITTENT_DETECTION = "intermittent_detection"
    CRITICAL_LANDMARK_MISSING = "critical_landmark_missing"


@dataclass
class EdgeCaseFlag:
    """A detected edge case with its impact assessment."""
    case_type: EdgeCaseType
    severity: str              # "low", "medium", "high"
    affected_time_range: Optional[tuple]  # (start_time, end_time) or None for global
    affected_events: List[int]  # Indices of affected behavior events
    description: str
    recommended_action: str
    confidence_adjustment: float  # Factor to multiply confidence (0-1)


@dataclass
class EdgeCaseReport:
    """Complete edge case analysis for a video."""
    total_flags: int
    flags_by_severity: Dict[str, int]
    edge_cases: List[EdgeCaseFlag]
    overall_reliability: float     # 0-1, overall analysis reliability
    requires_manual_review: bool
    reliability_explanation: str
    adjusted_events: List[BehaviorEvent]  # Events with adjusted confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_flags": self.total_flags,
            "flags_by_severity": self.flags_by_severity,
            "edge_cases": [
                {
                    "type": ec.case_type.value,
                    "severity": ec.severity,
                    "affected_time_range": ec.affected_time_range,
                    "affected_events": ec.affected_events,
                    "description": ec.description,
                    "recommended_action": ec.recommended_action,
                    "confidence_adjustment": ec.confidence_adjustment
                }
                for ec in self.edge_cases
            ],
            "overall_reliability": self.overall_reliability,
            "requires_manual_review": self.requires_manual_review,
            "reliability_explanation": self.reliability_explanation
        }


class EdgeCaseHandler:
    """
    Detects and handles edge cases that may affect analysis reliability.

    Applied after quality analysis and behavior classification to:
    1. Flag unreliable segments
    2. Adjust confidence scores conservatively
    3. Document limitations for human reviewers
    """

    def __init__(
        self,
        min_confidence: float = EDGE_CASE_MIN_CONFIDENCE,
        min_quality: float = EDGE_CASE_MIN_QUALITY,
        ambiguity_threshold: float = EDGE_CASE_AMBIGUITY_THRESHOLD
    ):
        """
        Initialize edge case handler.

        Args:
            min_confidence: Minimum confidence to avoid low confidence flag
            min_quality: Minimum quality score to avoid quality flag
            ambiguity_threshold: Max probability gap to consider ambiguous
        """
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.ambiguity_threshold = ambiguity_threshold

    def analyze(
        self,
        behavior_events: List[BehaviorEvent],
        quality_report: VideoQualityReport
    ) -> EdgeCaseReport:
        """
        Analyze for edge cases and generate report.

        Args:
            behavior_events: Classified behavior events
            quality_report: Video quality analysis report

        Returns:
            EdgeCaseReport with flags and adjusted events
        """
        edge_cases = []

        # Check video-level quality issues
        edge_cases.extend(self._check_video_quality(quality_report))

        # Check behavior-level issues
        edge_cases.extend(self._check_behavior_reliability(behavior_events, quality_report))

        # Check for ambiguous classifications
        edge_cases.extend(self._check_ambiguous_classifications(behavior_events))

        # Check for intermittent detection
        edge_cases.extend(self._check_intermittent_detection(quality_report))

        # Calculate severity distribution
        flags_by_severity = {
            "low": sum(1 for ec in edge_cases if ec.severity == "low"),
            "medium": sum(1 for ec in edge_cases if ec.severity == "medium"),
            "high": sum(1 for ec in edge_cases if ec.severity == "high")
        }

        # Calculate overall reliability
        overall_reliability = self._calculate_reliability(edge_cases, quality_report)

        # Determine if manual review is required
        requires_manual_review = (
            flags_by_severity["high"] > 0 or
            flags_by_severity["medium"] > 2 or
            overall_reliability < 0.6
        )

        # Generate explanation
        reliability_explanation = self._generate_explanation(
            edge_cases,
            overall_reliability,
            requires_manual_review
        )

        # Apply confidence adjustments to events
        adjusted_events = self._apply_adjustments(behavior_events, edge_cases, quality_report)

        return EdgeCaseReport(
            total_flags=len(edge_cases),
            flags_by_severity=flags_by_severity,
            edge_cases=edge_cases,
            overall_reliability=overall_reliability,
            requires_manual_review=requires_manual_review,
            reliability_explanation=reliability_explanation,
            adjusted_events=adjusted_events
        )

    def _check_video_quality(self, quality_report: VideoQualityReport) -> List[EdgeCaseFlag]:
        """Check for video-level quality issues."""
        flags = []

        # Low detection rate
        if quality_report.pose_detection_rate < 0.7:
            severity = "high" if quality_report.pose_detection_rate < 0.5 else "medium"
            flags.append(EdgeCaseFlag(
                case_type=EdgeCaseType.INSUFFICIENT_DATA,
                severity=severity,
                affected_time_range=None,
                affected_events=[],
                description=f"Only {quality_report.pose_detection_rate:.1%} of frames have detected poses",
                recommended_action="Review video quality; consider re-recording or manual analysis",
                confidence_adjustment=0.7 if severity == "medium" else 0.5
            ))

        # High overall occlusion
        if quality_report.average_occlusion_score > 0.4:
            severity = "high" if quality_report.average_occlusion_score > 0.6 else "medium"
            flags.append(EdgeCaseFlag(
                case_type=EdgeCaseType.HIGH_OCCLUSION,
                severity=severity,
                affected_time_range=None,
                affected_events=[],
                description=f"High average occlusion ({quality_report.average_occlusion_score:.2f}) reduces reliability",
                recommended_action="Check for obstructions in camera view",
                confidence_adjustment=0.6 if severity == "high" else 0.8
            ))

        # Low average confidence
        if quality_report.average_pose_confidence < 0.6:
            severity = "high" if quality_report.average_pose_confidence < 0.4 else "medium"
            flags.append(EdgeCaseFlag(
                case_type=EdgeCaseType.LOW_POSE_CONFIDENCE,
                severity=severity,
                affected_time_range=None,
                affected_events=[],
                description=f"Low average pose confidence ({quality_report.average_pose_confidence:.2f})",
                recommended_action="Verify lighting and camera angle",
                confidence_adjustment=0.6 if severity == "high" else 0.75
            ))

        # Low overall quality
        if quality_report.overall_quality_score < self.min_quality:
            flags.append(EdgeCaseFlag(
                case_type=EdgeCaseType.POOR_VIDEO_QUALITY,
                severity="high",
                affected_time_range=None,
                affected_events=[],
                description=f"Overall video quality ({quality_report.overall_quality_score:.2f}) below minimum",
                recommended_action="Results should not be trusted; manual review required",
                confidence_adjustment=0.4
            ))

        # Specific occlusion segments
        for start, end in quality_report.occlusion_segments:
            if end - start > 2.0:  # Only flag segments > 2 seconds
                flags.append(EdgeCaseFlag(
                    case_type=EdgeCaseType.HIGH_OCCLUSION,
                    severity="medium",
                    affected_time_range=(start, end),
                    affected_events=[],
                    description=f"Occlusion detected from {start:.1f}s to {end:.1f}s",
                    recommended_action="Events in this time range may be unreliable",
                    confidence_adjustment=0.7
                ))

        return flags

    def _check_behavior_reliability(
        self,
        behavior_events: List[BehaviorEvent],
        quality_report: VideoQualityReport
    ) -> List[EdgeCaseFlag]:
        """Check reliability of behavior classifications based on quality."""
        flags = []

        # Map frame quality to time ranges
        quality_by_time = {}
        for fm in quality_report.frame_metrics:
            quality_by_time[fm.timestamp] = fm

        for i, event in enumerate(behavior_events):
            # Check for critical landmark missing during suspicious events
            if event.behavior_type in ["pickup", "concealment", "bypass"]:
                # Find frames in this event's time range
                event_frames = [
                    fm for fm in quality_report.frame_metrics
                    if event.start_time <= fm.timestamp <= event.end_time
                ]

                if event_frames:
                    avg_event_quality = sum(fm.overall_quality for fm in event_frames) / len(event_frames)

                    if avg_event_quality < 0.5:
                        flags.append(EdgeCaseFlag(
                            case_type=EdgeCaseType.POOR_VIDEO_QUALITY,
                            severity="medium",
                            affected_time_range=(event.start_time, event.end_time),
                            affected_events=[i],
                            description=f"Suspicious event '{event.behavior_type}' at {event.start_time:.1f}s has low quality ({avg_event_quality:.2f})",
                            recommended_action="Verify this event manually before acting on it",
                            confidence_adjustment=0.6
                        ))

                    # Check for hand visibility (critical for behavior detection)
                    hand_visible_frames = sum(
                        1 for fm in event_frames
                        if 'Critical landmarks occluded' not in ' '.join(fm.quality_issues)
                    )
                    if hand_visible_frames < len(event_frames) * 0.5:
                        flags.append(EdgeCaseFlag(
                            case_type=EdgeCaseType.CRITICAL_LANDMARK_MISSING,
                            severity="high",
                            affected_time_range=(event.start_time, event.end_time),
                            affected_events=[i],
                            description=f"Hands not visible during '{event.behavior_type}' event",
                            recommended_action="Cannot reliably detect hand actions; manual review required",
                            confidence_adjustment=0.5
                        ))

        return flags

    def _check_ambiguous_classifications(
        self,
        behavior_events: List[BehaviorEvent]
    ) -> List[EdgeCaseFlag]:
        """Check for ambiguous behavior classifications."""
        flags = []

        for i, event in enumerate(behavior_events):
            # Check probability distribution
            probs = list(event.probabilities.values())
            sorted_probs = sorted(probs, reverse=True)

            if len(sorted_probs) >= 2:
                gap = sorted_probs[0] - sorted_probs[1]

                # If top two classes are close, classification is ambiguous
                if gap < self.ambiguity_threshold:
                    # Get the two top classes
                    sorted_classes = sorted(
                        event.probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    top_class = sorted_classes[0][0]
                    second_class = sorted_classes[1][0]

                    # More severe if ambiguity is between normal and suspicious
                    suspicious_types = {"pickup", "concealment", "bypass"}
                    is_critical_ambiguity = (
                        (top_class in suspicious_types and second_class not in suspicious_types) or
                        (top_class not in suspicious_types and second_class in suspicious_types)
                    )

                    severity = "high" if is_critical_ambiguity else "low"

                    flags.append(EdgeCaseFlag(
                        case_type=EdgeCaseType.AMBIGUOUS_CLASSIFICATION,
                        severity=severity,
                        affected_time_range=(event.start_time, event.end_time),
                        affected_events=[i],
                        description=f"Ambiguous between '{top_class}' ({sorted_probs[0]:.2f}) and '{second_class}' ({sorted_probs[1]:.2f})",
                        recommended_action="Manual verification recommended for this segment",
                        confidence_adjustment=0.7 if is_critical_ambiguity else 0.9
                    ))

        return flags

    def _check_intermittent_detection(
        self,
        quality_report: VideoQualityReport
    ) -> List[EdgeCaseFlag]:
        """Check for intermittent pose detection (flickering)."""
        flags = []

        # Count transitions between detected and not detected
        transitions = 0
        last_detected = None

        for fm in quality_report.frame_metrics:
            if last_detected is not None and fm.pose_detected != last_detected:
                transitions += 1
            last_detected = fm.pose_detected

        # High transition rate indicates tracking issues
        if quality_report.total_frames > 10:
            transition_rate = transitions / quality_report.total_frames

            if transition_rate > 0.1:  # More than 10% frames have detection changes
                severity = "high" if transition_rate > 0.2 else "medium"
                flags.append(EdgeCaseFlag(
                    case_type=EdgeCaseType.INTERMITTENT_DETECTION,
                    severity=severity,
                    affected_time_range=None,
                    affected_events=[],
                    description=f"Pose detection is intermittent ({transition_rate:.1%} frame transitions)",
                    recommended_action="Check for fast movement, lighting changes, or subject leaving frame",
                    confidence_adjustment=0.6 if severity == "high" else 0.75
                ))

        return flags

    def _calculate_reliability(
        self,
        edge_cases: List[EdgeCaseFlag],
        quality_report: VideoQualityReport
    ) -> float:
        """Calculate overall analysis reliability score."""
        # Start with quality score
        reliability = quality_report.overall_quality_score

        # Apply penalties for edge cases
        for ec in edge_cases:
            penalty = 1.0 - ec.confidence_adjustment
            if ec.severity == "high":
                reliability -= penalty * 0.15
            elif ec.severity == "medium":
                reliability -= penalty * 0.08
            else:
                reliability -= penalty * 0.03

        return max(0.0, min(1.0, reliability))

    def _generate_explanation(
        self,
        edge_cases: List[EdgeCaseFlag],
        overall_reliability: float,
        requires_manual_review: bool
    ) -> str:
        """Generate human-readable reliability explanation."""
        lines = [
            "RELIABILITY ASSESSMENT",
            "=" * 40,
            "",
            f"Overall Reliability: {overall_reliability:.1%}",
            f"Edge Cases Detected: {len(edge_cases)}",
            ""
        ]

        if edge_cases:
            lines.append("Issues Found:")
            for ec in edge_cases:
                severity_marker = {"high": "!!!", "medium": "!!", "low": "!"}[ec.severity]
                lines.append(f"  {severity_marker} [{ec.case_type.value}] {ec.description}")

        lines.append("")

        if requires_manual_review:
            lines.extend([
                "*** MANUAL REVIEW REQUIRED ***",
                "The analysis has significant reliability concerns.",
                "A human operator should verify results before any action.",
            ])
        elif overall_reliability < 0.8:
            lines.extend([
                "CAUTION: Moderate reliability concerns.",
                "Results should be interpreted with care.",
            ])
        else:
            lines.append("Reliability is acceptable for automated processing.")

        return "\n".join(lines)

    def _apply_adjustments(
        self,
        behavior_events: List[BehaviorEvent],
        edge_cases: List[EdgeCaseFlag],
        quality_report: VideoQualityReport
    ) -> List[BehaviorEvent]:
        """Apply confidence adjustments to behavior events based on edge cases."""
        adjusted_events = []

        # Build adjustment map for each event
        event_adjustments = {i: [] for i in range(len(behavior_events))}

        for ec in edge_cases:
            if ec.affected_events:
                # Specific events affected
                for idx in ec.affected_events:
                    if idx < len(behavior_events):
                        event_adjustments[idx].append(ec.confidence_adjustment)
            elif ec.affected_time_range:
                # Time range affected - find overlapping events
                start, end = ec.affected_time_range
                for i, event in enumerate(behavior_events):
                    if event.start_time < end and event.end_time > start:
                        event_adjustments[i].append(ec.confidence_adjustment)
            else:
                # Global adjustment
                for i in range(len(behavior_events)):
                    event_adjustments[i].append(ec.confidence_adjustment)

        # Apply adjustments
        for i, event in enumerate(behavior_events):
            adjustments = event_adjustments[i]

            if adjustments:
                # Use minimum adjustment (most conservative)
                adjustment_factor = min(adjustments)
                new_confidence = event.confidence * adjustment_factor

                # Create adjusted event
                adjusted_event = BehaviorEvent(
                    behavior_type=event.behavior_type,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    start_frame=event.start_frame,
                    end_frame=event.end_frame,
                    confidence=new_confidence,
                    probabilities={
                        k: v * adjustment_factor
                        for k, v in event.probabilities.items()
                    }
                )
                adjusted_events.append(adjusted_event)
            else:
                adjusted_events.append(event)

        return adjusted_events

    def filter_low_confidence_events(
        self,
        events: List[BehaviorEvent],
        min_confidence: float = None
    ) -> List[BehaviorEvent]:
        """
        Filter out events below confidence threshold.

        Args:
            events: List of behavior events
            min_confidence: Minimum confidence (uses default if None)

        Returns:
            Filtered list of events
        """
        threshold = min_confidence or self.min_confidence
        return [e for e in events if e.confidence >= threshold]
