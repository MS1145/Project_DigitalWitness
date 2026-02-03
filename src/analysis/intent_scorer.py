"""
Intent scoring module for Digital Witness.

Computes a weighted risk score (0.0-1.0) from multiple evidence sources:
POS discrepancies, concealment behaviors, checkout bypass, and duration.

The score is NOT a guilt determination - it's a prioritization metric
for human review. All weights are configurable in config.py.

Scoring Formula:
----------------
Intent Score = (w1 * discrepancy) + (w2 * concealment) + (w3 * bypass) + (w4 * duration)

Where:
- discrepancy (40%): Items detected but not billed (strongest signal)
- concealment (30%): Hiding behavior detected by LSTM
- bypass (20%): Checkout avoidance behavior detected
- duration (10%): Time spent in suspicious state

Why These Weights?
------------------
POS discrepancy is weighted highest because it's the most concrete evidence:
an item was physically handled but not paid for. Behavioral signals are
weighted lower because they have higher false positive rates (e.g., someone
might appear to conceal an item while actually adjusting their jacket).

Severity Thresholds:
--------------------
- NONE: score < 0.3 (normal shopping behavior)
- LOW: 0.3 <= score < 0.5 (minor anomalies)
- MEDIUM: 0.5 <= score < 0.7 (review recommended)
- HIGH: 0.7 <= score < 0.85 (immediate review)
- CRITICAL: score >= 0.85 (potential intervention)
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .cross_checker import DiscrepancyReport
from ..models.behavior_event import BehaviorEvent
from ..config import (
    WEIGHT_DISCREPANCY,       # 0.4 - POS mismatch weight
    WEIGHT_CONCEALMENT,       # 0.3 - hiding behavior weight
    WEIGHT_BYPASS,            # 0.2 - checkout avoidance weight
    WEIGHT_DURATION,          # 0.1 - time in suspicious state weight
    SEVERITY_LEVELS,
    INTENT_THRESHOLD_LOW,     # 0.3
    INTENT_THRESHOLD_MEDIUM,  # 0.5
    INTENT_THRESHOLD_HIGH     # 0.7
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
        Scale factor of 1.5 means 67% unbilled items = max score (1.0).

        Examples:
        - 0 items unbilled → score = 0.0
        - 1 of 3 items unbilled (33%) → score = 0.5
        - 2 of 3 items unbilled (67%) → score = 1.0 (capped)
        - 3 of 3 items unbilled (100%) → score = 1.0 (capped)
        """
        if report.total_detected == 0:
            return 0.0

        # Calculate ratio of missing (unbilled) items
        missing_ratio = report.discrepancy_count / report.total_detected
        # Scale up and cap at 1.0 (67% unbilled = max score)
        return min(1.0, missing_ratio * 1.5)

    def _score_concealment(self, events: List[BehaviorEvent]) -> float:
        """
        Score from concealment behaviors (hiding items on person).

        Factors in both count and confidence. Maxes out at 3 events
        to avoid over-weighting repeated low-confidence detections.

        Formula: avg_confidence * min(count/3, 1.0)

        This means:
        - 1 high-confidence concealment (0.9) → 0.9 * 0.33 = 0.30
        - 3 high-confidence concealments (0.9) → 0.9 * 1.0 = 0.90
        - 5 low-confidence concealments (0.5) → 0.5 * 1.0 = 0.50 (capped at 3)
        """
        concealment_events = [e for e in events if e.behavior_type == "concealment"]

        if not concealment_events:
            return 0.0

        # Average confidence across all concealment events
        avg_confidence = sum(e.confidence for e in concealment_events) / len(concealment_events)
        # Count factor: saturates at 3 events to prevent gaming via many low-conf detections
        count_factor = min(1.0, len(concealment_events) / 3)

        return avg_confidence * count_factor

    def _score_bypass(self, events: List[BehaviorEvent]) -> float:
        """
        Score from checkout bypass detection.

        Uses max confidence rather than average since a single high-confidence
        bypass is more significant than multiple uncertain ones.

        Rationale: You only need to bypass checkout once to steal items.
        Multiple bypass detections often indicate the same exit attempt
        captured across multiple time windows.
        """
        bypass_events = [e for e in events if e.behavior_type == "bypass"]

        if not bypass_events:
            return 0.0

        # Return highest confidence bypass detection
        return max(e.confidence for e in bypass_events)

    def _score_duration(
        self,
        events: List[BehaviorEvent],
        video_duration: float
    ) -> float:
        """
        Score from time spent in suspicious states.

        Longer durations of concealment/bypass increase risk score.
        Scale factor of 3 means ~33% suspicious time = max score (1.0).

        Why duration matters:
        - Brief concealment might be adjusting clothing
        - Extended concealment more likely intentional
        - Prolonged bypass (wandering near exit) suggests hesitation/intent

        Examples (in 60-second video):
        - 5 sec suspicious → 5/60 * 3 = 0.25
        - 10 sec suspicious → 10/60 * 3 = 0.50
        - 20 sec suspicious → 20/60 * 3 = 1.00 (capped)
        """
        suspicious_types = {"concealment", "bypass"}
        suspicious_events = [e for e in events if e.behavior_type in suspicious_types]

        if not suspicious_events or video_duration <= 0:
            return 0.0

        # Sum total time spent in suspicious states
        suspicious_duration = sum(e.end_time - e.start_time for e in suspicious_events)
        ratio = suspicious_duration / video_duration
        # Scale up and cap (33% suspicious time = max score)
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
