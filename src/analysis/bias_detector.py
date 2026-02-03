"""
Bias detection module for Digital Witness.

Detects potential sources of bias in the analysis pipeline:
- Confidence variance across different segments
- Detection rate stability
- Temporal patterns that might indicate systematic bias
- Model calibration issues

This module provides transparency about analysis reliability
and flags cases where bias might affect results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np

from ..models.behavior_event import BehaviorEvent
from .quality_analyzer import VideoQualityReport
from ..config import (
    BIAS_SENSITIVITY,
    BIAS_DETECTION_RATE_MIN,
    BIAS_CONFIDENCE_VARIANCE_MAX
)


class BiasRiskLevel(Enum):
    """Risk levels for bias assessment."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class BiasMetrics:
    """Quantitative metrics for bias assessment."""
    # Confidence distribution metrics
    confidence_mean: float              # Mean prediction confidence
    confidence_std: float               # Standard deviation of confidence
    confidence_variance: float          # Variance in predictions
    confidence_skew: float              # Skewness (asymmetry in distribution)

    # Detection consistency metrics
    detection_rate_stability: float     # How stable is pose detection
    temporal_consistency: float         # Consistency across time segments

    # Class distribution metrics
    class_balance_ratio: float          # Ratio of minority to majority class
    suspicious_ratio: float             # Ratio of suspicious predictions

    # Calibration metrics
    calibration_score: float            # How well calibrated are probabilities

    # Overall assessment
    temporal_bias_score: float          # Overall temporal bias indicator
    overall_bias_risk: BiasRiskLevel    # Summary risk level


@dataclass
class BiasFlag:
    """A detected bias indicator."""
    bias_type: str                      # Type of bias detected
    severity: str                       # "low", "medium", "high"
    description: str                    # Human-readable description
    affected_segments: List[Tuple[float, float]]  # Time ranges affected
    recommendation: str                 # Suggested action
    confidence_impact: float            # How much confidence should be adjusted


@dataclass
class FairnessReport:
    """Complete fairness/bias assessment report."""
    bias_metrics: BiasMetrics
    bias_flags: List[BiasFlag]
    recommendations: List[str]
    flagged_issues: List[str]
    overall_fairness_score: float       # 0-1, higher is better
    analysis_reliable: bool             # Whether analysis can be trusted
    requires_manual_review: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bias_metrics": {
                "confidence_mean": self.bias_metrics.confidence_mean,
                "confidence_std": self.bias_metrics.confidence_std,
                "confidence_variance": self.bias_metrics.confidence_variance,
                "detection_rate_stability": self.bias_metrics.detection_rate_stability,
                "temporal_consistency": self.bias_metrics.temporal_consistency,
                "class_balance_ratio": self.bias_metrics.class_balance_ratio,
                "calibration_score": self.bias_metrics.calibration_score,
                "temporal_bias_score": self.bias_metrics.temporal_bias_score,
                "overall_bias_risk": self.bias_metrics.overall_bias_risk.value
            },
            "bias_flags": [
                {
                    "type": f.bias_type,
                    "severity": f.severity,
                    "description": f.description,
                    "affected_segments": f.affected_segments,
                    "recommendation": f.recommendation
                }
                for f in self.bias_flags
            ],
            "recommendations": self.recommendations,
            "flagged_issues": self.flagged_issues,
            "overall_fairness_score": self.overall_fairness_score,
            "analysis_reliable": self.analysis_reliable,
            "requires_manual_review": self.requires_manual_review
        }


class BiasDetector:
    """
    Detects potential biases in the analysis pipeline.

    Analyzes behavior predictions and quality metrics to identify
    systematic patterns that might indicate unreliable analysis.
    """

    def __init__(
        self,
        sensitivity: float = BIAS_SENSITIVITY,
        min_detection_rate: float = BIAS_DETECTION_RATE_MIN,
        max_confidence_variance: float = BIAS_CONFIDENCE_VARIANCE_MAX
    ):
        """
        Initialize bias detector.

        Args:
            sensitivity: How sensitive to potential bias (0-1)
            min_detection_rate: Minimum acceptable detection rate
            max_confidence_variance: Maximum acceptable confidence variance
        """
        self.sensitivity = sensitivity
        self.min_detection_rate = min_detection_rate
        self.max_confidence_variance = max_confidence_variance

    def analyze(
        self,
        behavior_events: List[BehaviorEvent],
        quality_report: Optional[VideoQualityReport] = None,
        video_duration: float = 0.0
    ) -> FairnessReport:
        """
        Analyze for potential bias in predictions.

        Args:
            behavior_events: Classified behavior events
            quality_report: Optional video quality report
            video_duration: Total video duration

        Returns:
            FairnessReport with bias assessment
        """
        # Calculate bias metrics
        metrics = self._calculate_metrics(behavior_events, quality_report)

        # Detect bias flags
        flags = self._detect_bias_flags(behavior_events, quality_report, metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(flags, metrics)

        # Summarize flagged issues
        flagged_issues = [f.description for f in flags if f.severity in ["medium", "high"]]

        # Calculate overall fairness score
        fairness_score = self._calculate_fairness_score(metrics, flags)

        # Determine if analysis is reliable
        analysis_reliable = (
            fairness_score >= 0.6 and
            metrics.overall_bias_risk != BiasRiskLevel.HIGH
        )

        # Determine if manual review required
        requires_manual_review = (
            not analysis_reliable or
            any(f.severity == "high" for f in flags) or
            fairness_score < 0.5
        )

        return FairnessReport(
            bias_metrics=metrics,
            bias_flags=flags,
            recommendations=recommendations,
            flagged_issues=flagged_issues,
            overall_fairness_score=fairness_score,
            analysis_reliable=analysis_reliable,
            requires_manual_review=requires_manual_review
        )

    def _calculate_metrics(
        self,
        behavior_events: List[BehaviorEvent],
        quality_report: Optional[VideoQualityReport]
    ) -> BiasMetrics:
        """Calculate quantitative bias metrics."""
        if not behavior_events:
            return BiasMetrics(
                confidence_mean=0.0,
                confidence_std=0.0,
                confidence_variance=0.0,
                confidence_skew=0.0,
                detection_rate_stability=0.0,
                temporal_consistency=0.0,
                class_balance_ratio=0.0,
                suspicious_ratio=0.0,
                calibration_score=0.0,
                temporal_bias_score=0.0,
                overall_bias_risk=BiasRiskLevel.HIGH
            )

        # Confidence metrics
        confidences = [e.confidence for e in behavior_events]
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        conf_variance = np.var(confidences)

        # Calculate skewness
        if conf_std > 0:
            conf_skew = np.mean(((np.array(confidences) - conf_mean) / conf_std) ** 3)
        else:
            conf_skew = 0.0

        # Detection rate stability from quality report
        detection_stability = 1.0
        if quality_report:
            detection_stability = quality_report.pose_detection_rate

        # Temporal consistency - check if predictions are stable over time
        temporal_consistency = self._calculate_temporal_consistency(behavior_events)

        # Class balance
        class_counts = {}
        for event in behavior_events:
            cls = event.behavior_type
            class_counts[cls] = class_counts.get(cls, 0) + 1

        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            class_balance = min_count / max_count if max_count > 0 else 0.0
        else:
            class_balance = 0.0

        # Suspicious ratio
        suspicious_types = {"pickup", "concealment", "bypass", "shoplifting"}
        suspicious_count = sum(
            1 for e in behavior_events if e.behavior_type in suspicious_types
        )
        suspicious_ratio = suspicious_count / len(behavior_events)

        # Calibration score (simplified - checks if confidence aligns with accuracy)
        calibration = self._estimate_calibration(behavior_events)

        # Temporal bias score
        temporal_bias = self._calculate_temporal_bias(behavior_events)

        # Overall bias risk
        bias_risk = self._determine_risk_level(
            conf_variance, detection_stability, temporal_consistency, calibration
        )

        return BiasMetrics(
            confidence_mean=conf_mean,
            confidence_std=conf_std,
            confidence_variance=conf_variance,
            confidence_skew=conf_skew,
            detection_rate_stability=detection_stability,
            temporal_consistency=temporal_consistency,
            class_balance_ratio=class_balance,
            suspicious_ratio=suspicious_ratio,
            calibration_score=calibration,
            temporal_bias_score=temporal_bias,
            overall_bias_risk=bias_risk
        )

    def _calculate_temporal_consistency(
        self,
        behavior_events: List[BehaviorEvent]
    ) -> float:
        """
        Calculate how consistent predictions are over time.

        Frequent rapid changes indicate potential instability.
        """
        if len(behavior_events) < 2:
            return 1.0

        # Count class transitions
        transitions = 0
        for i in range(1, len(behavior_events)):
            if behavior_events[i].behavior_type != behavior_events[i-1].behavior_type:
                transitions += 1

        # Calculate transition rate
        max_transitions = len(behavior_events) - 1
        transition_rate = transitions / max_transitions if max_transitions > 0 else 0

        # Lower transition rate = higher consistency
        # Expect some transitions (around 0.3), but not too many
        if transition_rate < 0.5:
            return 1.0 - (transition_rate * 0.5)
        else:
            return 0.5 - (transition_rate - 0.5)

    def _estimate_calibration(
        self,
        behavior_events: List[BehaviorEvent]
    ) -> float:
        """
        Estimate prediction calibration quality.

        Well-calibrated models have confidence that reflects actual accuracy.
        """
        if not behavior_events:
            return 0.0

        # Group by confidence ranges
        ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        calibration_errors = []

        for low, high in ranges:
            events_in_range = [
                e for e in behavior_events
                if low <= e.confidence < high
            ]

            if events_in_range:
                # Expected accuracy = mid-point of confidence range
                expected = (low + high) / 2

                # Check if confidence distribution is reasonable
                # (In real system, would compare to actual accuracy)
                avg_conf = np.mean([e.confidence for e in events_in_range])

                # Penalize if average is very different from expected
                error = abs(avg_conf - expected)
                calibration_errors.append(error)

        if calibration_errors:
            avg_error = np.mean(calibration_errors)
            return max(0.0, 1.0 - avg_error * 2)

        return 0.8  # Default reasonable calibration

    def _calculate_temporal_bias(
        self,
        behavior_events: List[BehaviorEvent]
    ) -> float:
        """
        Calculate temporal bias score.

        High score indicates bias toward certain time periods.
        """
        if len(behavior_events) < 4:
            return 0.0

        # Split into halves
        mid = len(behavior_events) // 2
        first_half = behavior_events[:mid]
        second_half = behavior_events[mid:]

        # Compare suspicious ratios
        suspicious_types = {"pickup", "concealment", "bypass"}

        first_suspicious = sum(
            1 for e in first_half if e.behavior_type in suspicious_types
        ) / len(first_half)

        second_suspicious = sum(
            1 for e in second_half if e.behavior_type in suspicious_types
        ) / len(second_half)

        # Large difference indicates temporal bias
        temporal_bias = abs(first_suspicious - second_suspicious)

        return temporal_bias

    def _determine_risk_level(
        self,
        conf_variance: float,
        detection_stability: float,
        temporal_consistency: float,
        calibration: float
    ) -> BiasRiskLevel:
        """Determine overall bias risk level."""
        risk_score = 0

        # High variance is concerning
        if conf_variance > self.max_confidence_variance:
            risk_score += 2

        # Low detection stability
        if detection_stability < self.min_detection_rate:
            risk_score += 2

        # Poor temporal consistency
        if temporal_consistency < 0.5:
            risk_score += 1

        # Poor calibration
        if calibration < 0.5:
            risk_score += 1

        if risk_score >= 4:
            return BiasRiskLevel.HIGH
        elif risk_score >= 2:
            return BiasRiskLevel.MEDIUM
        else:
            return BiasRiskLevel.LOW

    def _detect_bias_flags(
        self,
        behavior_events: List[BehaviorEvent],
        quality_report: Optional[VideoQualityReport],
        metrics: BiasMetrics
    ) -> List[BiasFlag]:
        """Detect specific bias indicators."""
        flags = []

        # High confidence variance
        if metrics.confidence_variance > self.max_confidence_variance:
            flags.append(BiasFlag(
                bias_type="high_confidence_variance",
                severity="medium",
                description=f"Prediction confidence varies significantly ({metrics.confidence_variance:.3f})",
                affected_segments=[],
                recommendation="Results may be inconsistent; manual verification recommended",
                confidence_impact=0.8
            ))

        # Low detection stability
        if metrics.detection_rate_stability < self.min_detection_rate:
            flags.append(BiasFlag(
                bias_type="unstable_detection",
                severity="high",
                description=f"Pose detection rate is low ({metrics.detection_rate_stability:.1%})",
                affected_segments=[],
                recommendation="Video quality may be causing unreliable analysis",
                confidence_impact=0.6
            ))

        # Temporal bias
        if metrics.temporal_bias_score > 0.3:
            flags.append(BiasFlag(
                bias_type="temporal_bias",
                severity="medium",
                description=f"Suspicious predictions concentrated in certain time periods",
                affected_segments=[],
                recommendation="Review if this pattern reflects actual behavior or bias",
                confidence_impact=0.85
            ))

        # Class imbalance flag
        if metrics.class_balance_ratio < 0.1:
            flags.append(BiasFlag(
                bias_type="class_imbalance",
                severity="low",
                description="One behavior class dominates predictions",
                affected_segments=[],
                recommendation="Consider if this reflects actual behavior distribution",
                confidence_impact=0.95
            ))

        # High suspicious ratio
        if metrics.suspicious_ratio > 0.5:
            flags.append(BiasFlag(
                bias_type="high_suspicious_rate",
                severity="medium",
                description=f"Unusually high rate of suspicious predictions ({metrics.suspicious_ratio:.1%})",
                affected_segments=[],
                recommendation="Verify that model is not biased toward false positives",
                confidence_impact=0.75
            ))

        # Poor calibration
        if metrics.calibration_score < 0.5:
            flags.append(BiasFlag(
                bias_type="poor_calibration",
                severity="medium",
                description="Model confidence may not reflect actual accuracy",
                affected_segments=[],
                recommendation="Treat confidence scores with skepticism",
                confidence_impact=0.7
            ))

        # Confidence skew
        if abs(metrics.confidence_skew) > 1.0:
            direction = "low" if metrics.confidence_skew < 0 else "high"
            flags.append(BiasFlag(
                bias_type="confidence_skew",
                severity="low",
                description=f"Confidence scores skewed toward {direction} values",
                affected_segments=[],
                recommendation="Model may be overconfident or underconfident",
                confidence_impact=0.9
            ))

        return flags

    def _generate_recommendations(
        self,
        flags: List[BiasFlag],
        metrics: BiasMetrics
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Always include baseline recommendation
        recommendations.append(
            "All alerts require human validation before any action is taken"
        )

        if metrics.overall_bias_risk == BiasRiskLevel.HIGH:
            recommendations.append(
                "HIGH BIAS RISK: Analysis results should be treated as preliminary only"
            )

        if any(f.bias_type == "unstable_detection" for f in flags):
            recommendations.append(
                "Consider re-analyzing with better quality video footage"
            )

        if any(f.bias_type == "high_suspicious_rate" for f in flags):
            recommendations.append(
                "Manually review suspicious segments to verify accuracy"
            )

        if any(f.bias_type == "temporal_bias" for f in flags):
            recommendations.append(
                "Review whether suspicious activity clustering is genuine or artifact"
            )

        if metrics.calibration_score < 0.6:
            recommendations.append(
                "Do not rely solely on confidence scores for decision-making"
            )

        if not flags:
            recommendations.append(
                "No significant bias indicators detected; standard review process applies"
            )

        return recommendations

    def _calculate_fairness_score(
        self,
        metrics: BiasMetrics,
        flags: List[BiasFlag]
    ) -> float:
        """Calculate overall fairness score (0-1)."""
        # Start with base score
        score = 1.0

        # Deduct for high variance
        if metrics.confidence_variance > self.max_confidence_variance:
            score -= 0.15

        # Deduct for low detection stability
        stability_penalty = max(0, (self.min_detection_rate - metrics.detection_rate_stability) * 0.3)
        score -= stability_penalty

        # Deduct for temporal bias
        score -= metrics.temporal_bias_score * 0.2

        # Deduct for poor calibration
        score -= (1 - metrics.calibration_score) * 0.15

        # Deduct for each flag
        for flag in flags:
            if flag.severity == "high":
                score -= 0.1
            elif flag.severity == "medium":
                score -= 0.05

        return max(0.0, min(1.0, score))

    def get_bias_summary(self, report: FairnessReport) -> str:
        """Generate human-readable bias summary."""
        lines = [
            "FAIRNESS ASSESSMENT",
            "=" * 40,
            "",
            f"Overall Fairness Score: {report.overall_fairness_score:.1%}",
            f"Bias Risk Level: {report.bias_metrics.overall_bias_risk.value}",
            f"Analysis Reliable: {'Yes' if report.analysis_reliable else 'No'}",
            ""
        ]

        if report.bias_flags:
            lines.append("Bias Indicators Detected:")
            for flag in report.bias_flags:
                severity_marker = {"high": "!!!", "medium": "!!", "low": "!"}[flag.severity]
                lines.append(f"  {severity_marker} {flag.description}")
            lines.append("")

        if report.recommendations:
            lines.append("Recommendations:")
            for rec in report.recommendations:
                lines.append(f"  - {rec}")

        if report.requires_manual_review:
            lines.extend([
                "",
                "*** MANUAL REVIEW REQUIRED ***",
                "Potential bias indicators warrant human verification"
            ])

        return "\n".join(lines)
