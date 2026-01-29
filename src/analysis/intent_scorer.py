"""
Intent scoring module for Digital Witness.

Computes a weighted risk score (0.0-1.0) from multiple evidence sources:
POS discrepancies, concealment behaviors, checkout bypass, and duration.

The score is NOT a guilt determination - it's a prioritization metric
for human review. All weights are configurable in config.py.
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .cross_checker import DiscrepancyReport
from ..pose.behavior_classifier import BehaviorEvent
from ..config import (
    WEIGHT_DISCREPANCY,
    WEIGHT_CONCEALMENT,
    WEIGHT_BYPASS,
    WEIGHT_DURATION,
    SEVERITY_LEVELS,
    INTENT_THRESHOLD_LOW,
    INTENT_THRESHOLD_MEDIUM,
    INTENT_THRESHOLD_HIGH
)


class Severity(Enum):
    """Severity tiers for alert prioritization."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class IntentScore:
    """
    Composite risk assessment result.

    The components dict provides transparency into how the score
    was computed, enabling operators to understand the reasoning.
    """
    score: float           # Weighted sum, range [0.0, 1.0]
    severity: Severity     # Derived from score thresholds
    components: dict       # Per-factor breakdown for explainability
    explanation: str       # Human-readable summary


class IntentScorer:
    """
    Multi-factor risk scoring engine.

    Combines evidence from video analysis (behaviors) and POS data
    (discrepancies) into a single normalized score.
    """

    def __init__(
        self,
        weight_discrepancy: float = WEIGHT_DISCREPANCY,
        weight_concealment: float = WEIGHT_CONCEALMENT,
        weight_bypass: float = WEIGHT_BYPASS,
        weight_duration: float = WEIGHT_DURATION
    ):
        """
        Initialize intent scorer with configurable weights.

        Args:
            weight_discrepancy: Weight for POS discrepancy component
            weight_concealment: Weight for concealment behavior component
            weight_bypass: Weight for checkout bypass component
            weight_duration: Weight for duration of suspicious behavior
        """
        self.weight_discrepancy = weight_discrepancy
        self.weight_concealment = weight_concealment
        self.weight_bypass = weight_bypass
        self.weight_duration = weight_duration

        # Normalize weights
        total = (weight_discrepancy + weight_concealment +
                 weight_bypass + weight_duration)
        self.weight_discrepancy /= total
        self.weight_concealment /= total
        self.weight_bypass /= total
        self.weight_duration /= total

    def calculate_score(
        self,
        discrepancy_report: DiscrepancyReport,
        behavior_events: List[BehaviorEvent],
        video_duration: float
    ) -> IntentScore:
        """
        Calculate intent score based on all available evidence.

        Args:
            discrepancy_report: Results from cross-checking
            behavior_events: Classified behavior events
            video_duration: Total video duration in seconds

        Returns:
            IntentScore with score, severity, and explanation
        """
        # Calculate component scores
        discrepancy_score = self._score_discrepancy(discrepancy_report)
        concealment_score = self._score_concealment(behavior_events)
        bypass_score = self._score_bypass(behavior_events)
        duration_score = self._score_duration(behavior_events, video_duration)

        # Calculate weighted total
        total_score = (
            self.weight_discrepancy * discrepancy_score +
            self.weight_concealment * concealment_score +
            self.weight_bypass * bypass_score +
            self.weight_duration * duration_score
        )

        # Clamp to [0, 1]
        total_score = max(0.0, min(1.0, total_score))

        # Determine severity
        severity = self._determine_severity(total_score)

        # Build components dict
        components = {
            "discrepancy": {
                "score": discrepancy_score,
                "weight": self.weight_discrepancy,
                "contribution": discrepancy_score * self.weight_discrepancy
            },
            "concealment": {
                "score": concealment_score,
                "weight": self.weight_concealment,
                "contribution": concealment_score * self.weight_concealment
            },
            "bypass": {
                "score": bypass_score,
                "weight": self.weight_bypass,
                "contribution": bypass_score * self.weight_bypass
            },
            "duration": {
                "score": duration_score,
                "weight": self.weight_duration,
                "contribution": duration_score * self.weight_duration
            }
        }

        # Generate explanation
        explanation = self._generate_explanation(
            total_score, severity, components, discrepancy_report, behavior_events
        )

        return IntentScore(
            score=total_score,
            severity=severity,
            components=components,
            explanation=explanation
        )

    def _score_discrepancy(self, report: DiscrepancyReport) -> float:
        """
        Score from POS mismatches (strongest signal).

        Items picked up but not billed are the primary indicator.
        Scale factor of 1.5 means 67% unbilled items = max score.
        """
        if report.total_detected == 0:
            return 0.0
        missing_ratio = report.discrepancy_count / report.total_detected
        return min(1.0, missing_ratio * 1.5)

    def _score_concealment(self, events: List[BehaviorEvent]) -> float:
        """
        Score from concealment behaviors (hiding items on person).

        Factors in both count and confidence. Maxes out at 3 events
        to avoid over-weighting repeated low-confidence detections.
        """
        concealment_events = [e for e in events if e.behavior_type == "concealment"]

        if not concealment_events:
            return 0.0

        avg_confidence = sum(e.confidence for e in concealment_events) / len(concealment_events)
        count_factor = min(1.0, len(concealment_events) / 3)

        return avg_confidence * count_factor

    def _score_bypass(self, events: List[BehaviorEvent]) -> float:
        """
        Score from checkout bypass detection.

        Uses max confidence rather than average since a single high-confidence
        bypass is more significant than multiple uncertain ones.
        """
        bypass_events = [e for e in events if e.behavior_type == "bypass"]

        if not bypass_events:
            return 0.0

        return max(e.confidence for e in bypass_events)

    def _score_duration(
        self,
        events: List[BehaviorEvent],
        video_duration: float
    ) -> float:
        """
        Score from time spent in suspicious states.

        Longer durations of concealment/bypass increase risk score.
        Scale factor of 3 means ~33% suspicious time = max score.
        """
        suspicious_types = {"concealment", "bypass"}
        suspicious_events = [e for e in events if e.behavior_type in suspicious_types]

        if not suspicious_events or video_duration <= 0:
            return 0.0

        suspicious_duration = sum(e.end_time - e.start_time for e in suspicious_events)
        ratio = suspicious_duration / video_duration
        return min(1.0, ratio * 3)

    def _determine_severity(self, score: float) -> Severity:
        """Determine severity level from score."""
        if score < INTENT_THRESHOLD_LOW:
            return Severity.NONE
        elif score < INTENT_THRESHOLD_MEDIUM:
            return Severity.LOW
        elif score < INTENT_THRESHOLD_HIGH:
            return Severity.MEDIUM
        elif score < 0.85:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def _generate_explanation(
        self,
        score: float,
        severity: Severity,
        components: dict,
        discrepancy_report: DiscrepancyReport,
        events: List[BehaviorEvent]
    ) -> str:
        """Generate human-readable explanation of the score."""
        lines = []

        # Summary
        lines.append(f"Intent Score: {score:.2f} ({severity.value})")
        lines.append("")

        # Discrepancy explanation
        if discrepancy_report.discrepancy_count > 0:
            lines.append(f"- {discrepancy_report.discrepancy_count} item(s) detected but not billed")
            for sku in discrepancy_report.missing_from_billing:
                lines.append(f"  * {sku}")

        # Concealment explanation
        concealment_count = sum(
            1 for e in events if e.behavior_type == "concealment"
        )
        if concealment_count > 0:
            lines.append(f"- {concealment_count} concealment behavior(s) detected")

        # Bypass explanation
        bypass_count = sum(
            1 for e in events if e.behavior_type == "bypass"
        )
        if bypass_count > 0:
            lines.append(f"- {bypass_count} checkout bypass behavior(s) detected")

        # Conclusion
        lines.append("")
        if severity == Severity.NONE:
            lines.append("No suspicious activity detected.")
        elif severity == Severity.LOW:
            lines.append("Minor anomalies detected. Review recommended.")
        elif severity == Severity.MEDIUM:
            lines.append("Moderate concern. Human review required.")
        elif severity == Severity.HIGH:
            lines.append("High concern. Immediate human review required.")
        else:
            lines.append("Critical concern. Immediate intervention may be required.")

        lines.append("")
        lines.append("NOTE: Final decision must be made by human operator.")

        return "\n".join(lines)
