"""
Bias-aware intent scoring for Digital Witness.

Extends the base IntentScorer with fairness adjustments based
on detected bias indicators. Ensures that potential biases in
the analysis are reflected in the final risk assessment.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict

from .intent_scorer import IntentScorer, IntentScore, Severity
from .cross_checker import DiscrepancyReport
from .bias_detector import BiasDetector, FairnessReport, BiasRiskLevel
from .quality_analyzer import VideoQualityReport
from ..models.behavior_event import BehaviorEvent
from ..config import (
    WEIGHT_DISCREPANCY,
    WEIGHT_CONCEALMENT,
    WEIGHT_BYPASS,
    WEIGHT_DURATION
)


@dataclass
class BiasAwareIntentScore:
    """Intent score with bias adjustments and fairness information."""
    # Base score information
    raw_score: float                    # Original unadjusted score
    adjusted_score: float               # Score after bias adjustment
    severity: Severity                  # Severity based on adjusted score
    components: dict                    # Per-factor breakdown

    # Bias adjustment information
    bias_adjustment_factor: float       # Factor applied to raw score
    fairness_report: FairnessReport     # Full fairness assessment
    adjustment_explanation: str         # Why adjustment was applied

    # Final outputs
    final_score: float                  # The score to use for decisions
    explanation: str                    # Human-readable summary
    confidence_level: str               # "high", "medium", "low"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_score": self.raw_score,
            "adjusted_score": self.adjusted_score,
            "final_score": self.final_score,
            "severity": self.severity.value,
            "components": self.components,
            "bias_adjustment_factor": self.bias_adjustment_factor,
            "adjustment_explanation": self.adjustment_explanation,
            "confidence_level": self.confidence_level,
            "fairness_report": self.fairness_report.to_dict(),
            "explanation": self.explanation
        }


class BiasAwareScorer:
    """
    Intent scorer with integrated bias detection and adjustment.

    Extends the standard scoring process with:
    1. Bias detection on predictions
    2. Confidence adjustment based on bias indicators
    3. Fairness-aware severity classification
    4. Enhanced explainability
    """

    def __init__(
        self,
        weight_discrepancy: float = WEIGHT_DISCREPANCY,
        weight_concealment: float = WEIGHT_CONCEALMENT,
        weight_bypass: float = WEIGHT_BYPASS,
        weight_duration: float = WEIGHT_DURATION
    ):
        """
        Initialize bias-aware scorer.

        Args:
            weight_discrepancy: Weight for POS discrepancy component
            weight_concealment: Weight for concealment behavior
            weight_bypass: Weight for checkout bypass
            weight_duration: Weight for suspicious duration
        """
        # Initialize base scorer
        self.base_scorer = IntentScorer(
            weight_discrepancy=weight_discrepancy,
            weight_concealment=weight_concealment,
            weight_bypass=weight_bypass,
            weight_duration=weight_duration
        )

        # Initialize bias detector
        self.bias_detector = BiasDetector()

    def calculate_score(
        self,
        discrepancy_report: DiscrepancyReport,
        behavior_events: List[BehaviorEvent],
        video_duration: float,
        quality_report: Optional[VideoQualityReport] = None
    ) -> BiasAwareIntentScore:
        """
        Calculate bias-aware intent score.

        Args:
            discrepancy_report: Results from cross-checking
            behavior_events: Classified behavior events
            video_duration: Total video duration in seconds
            quality_report: Optional video quality report

        Returns:
            BiasAwareIntentScore with full bias assessment
        """
        # Step 1: Calculate base intent score
        base_score = self.base_scorer.calculate_score(
            discrepancy_report,
            behavior_events,
            video_duration
        )

        # Step 2: Run bias detection
        fairness_report = self.bias_detector.analyze(
            behavior_events,
            quality_report,
            video_duration
        )

        # Step 3: Calculate bias adjustment factor
        adjustment_factor, adjustment_reason = self._calculate_adjustment(
            fairness_report
        )

        # Step 4: Apply adjustment
        adjusted_score = self._apply_adjustment(
            base_score.score,
            adjustment_factor,
            fairness_report
        )

        # Step 5: Determine final severity
        final_severity = self._determine_severity(
            adjusted_score,
            fairness_report
        )

        # Step 6: Determine confidence level
        confidence_level = self._determine_confidence_level(
            fairness_report,
            quality_report
        )

        # Step 7: Generate explanation
        explanation = self._generate_explanation(
            base_score,
            adjusted_score,
            adjustment_factor,
            fairness_report,
            confidence_level
        )

        return BiasAwareIntentScore(
            raw_score=base_score.score,
            adjusted_score=adjusted_score,
            severity=final_severity,
            components=base_score.components,
            bias_adjustment_factor=adjustment_factor,
            fairness_report=fairness_report,
            adjustment_explanation=adjustment_reason,
            final_score=adjusted_score,
            explanation=explanation,
            confidence_level=confidence_level
        )

    def _calculate_adjustment(
        self,
        fairness_report: FairnessReport
    ) -> tuple:
        """
        Calculate the bias adjustment factor.

        Returns (adjustment_factor, reason_string).
        Adjustment < 1.0 reduces the score (more conservative).
        """
        factors = []
        reasons = []

        # Adjust based on overall fairness score
        if fairness_report.overall_fairness_score < 0.5:
            factors.append(0.7)
            reasons.append("Low fairness score")
        elif fairness_report.overall_fairness_score < 0.7:
            factors.append(0.85)
            reasons.append("Moderate fairness concerns")

        # Adjust for each high-severity flag
        high_flags = [f for f in fairness_report.bias_flags if f.severity == "high"]
        for flag in high_flags:
            factors.append(flag.confidence_impact)
            reasons.append(flag.bias_type)

        # Adjust for overall bias risk
        if fairness_report.bias_metrics.overall_bias_risk == BiasRiskLevel.HIGH:
            factors.append(0.7)
            reasons.append("High bias risk")
        elif fairness_report.bias_metrics.overall_bias_risk == BiasRiskLevel.MEDIUM:
            factors.append(0.9)
            reasons.append("Medium bias risk")

        # Calculate combined adjustment
        if factors:
            combined_factor = min(factors)  # Most conservative adjustment
            reason = f"Adjusted for: {', '.join(reasons)}"
        else:
            combined_factor = 1.0
            reason = "No bias adjustment needed"

        return combined_factor, reason

    def _apply_adjustment(
        self,
        raw_score: float,
        adjustment_factor: float,
        fairness_report: FairnessReport
    ) -> float:
        """
        Apply bias adjustment to raw score.

        Uses a nuanced approach that considers score magnitude.
        """
        # For high raw scores with bias concerns, apply stronger adjustment
        if raw_score > 0.7 and adjustment_factor < 1.0:
            # More aggressive adjustment for high scores with bias
            adjusted = raw_score * adjustment_factor * 0.9
        else:
            adjusted = raw_score * adjustment_factor

        # Ensure score stays in valid range
        return max(0.0, min(1.0, adjusted))

    def _determine_severity(
        self,
        adjusted_score: float,
        fairness_report: FairnessReport
    ) -> Severity:
        """
        Determine severity level with bias awareness.

        More conservative classification when bias is detected.
        """
        from ..config import (
            INTENT_THRESHOLD_LOW,
            INTENT_THRESHOLD_MEDIUM,
            INTENT_THRESHOLD_HIGH
        )

        # If high bias risk, use more conservative thresholds
        if fairness_report.bias_metrics.overall_bias_risk == BiasRiskLevel.HIGH:
            # Raise thresholds for conservative classification
            threshold_low = INTENT_THRESHOLD_LOW + 0.1
            threshold_medium = INTENT_THRESHOLD_MEDIUM + 0.1
            threshold_high = INTENT_THRESHOLD_HIGH + 0.1
        elif fairness_report.bias_metrics.overall_bias_risk == BiasRiskLevel.MEDIUM:
            threshold_low = INTENT_THRESHOLD_LOW + 0.05
            threshold_medium = INTENT_THRESHOLD_MEDIUM + 0.05
            threshold_high = INTENT_THRESHOLD_HIGH + 0.05
        else:
            threshold_low = INTENT_THRESHOLD_LOW
            threshold_medium = INTENT_THRESHOLD_MEDIUM
            threshold_high = INTENT_THRESHOLD_HIGH

        # Classify
        if adjusted_score < threshold_low:
            return Severity.NONE
        elif adjusted_score < threshold_medium:
            return Severity.LOW
        elif adjusted_score < threshold_high:
            return Severity.MEDIUM
        elif adjusted_score < 0.85:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def _determine_confidence_level(
        self,
        fairness_report: FairnessReport,
        quality_report: Optional[VideoQualityReport]
    ) -> str:
        """Determine overall confidence in the analysis."""
        score = 0

        # Check fairness score
        if fairness_report.overall_fairness_score >= 0.8:
            score += 2
        elif fairness_report.overall_fairness_score >= 0.6:
            score += 1

        # Check analysis reliability
        if fairness_report.analysis_reliable:
            score += 2

        # Check quality report
        if quality_report:
            if quality_report.usable_for_analysis:
                score += 1
            if quality_report.overall_quality_score >= 0.7:
                score += 1

        # Check bias flags
        high_flags = sum(1 for f in fairness_report.bias_flags if f.severity == "high")
        medium_flags = sum(1 for f in fairness_report.bias_flags if f.severity == "medium")

        if high_flags == 0 and medium_flags == 0:
            score += 2
        elif high_flags == 0:
            score += 1

        # Determine level
        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _generate_explanation(
        self,
        base_score: IntentScore,
        adjusted_score: float,
        adjustment_factor: float,
        fairness_report: FairnessReport,
        confidence_level: str
    ) -> str:
        """Generate comprehensive explanation."""
        lines = [
            "=" * 50,
            "BIAS-AWARE INTENT ASSESSMENT",
            "=" * 50,
            "",
            f"Final Score: {adjusted_score:.2f} ({self._determine_severity(adjusted_score, fairness_report).value})",
            f"Analysis Confidence: {confidence_level.upper()}",
            "",
            "--- SCORING BREAKDOWN ---",
            "",
            f"Raw Intent Score: {base_score.score:.2f}",
        ]

        # Component breakdown
        for name, comp in base_score.components.items():
            lines.append(f"  - {name}: {comp['score']:.2f} (weight: {comp['weight']:.1%})")

        lines.append("")

        # Bias adjustment
        if adjustment_factor < 1.0:
            lines.extend([
                "--- BIAS ADJUSTMENT ---",
                "",
                f"Adjustment Factor: {adjustment_factor:.2f}",
                f"Reason: Potential bias indicators detected",
                f"Adjusted Score: {adjusted_score:.2f}",
                ""
            ])

        # Fairness assessment
        lines.extend([
            "--- FAIRNESS ASSESSMENT ---",
            "",
            f"Fairness Score: {fairness_report.overall_fairness_score:.1%}",
            f"Bias Risk Level: {fairness_report.bias_metrics.overall_bias_risk.value}",
        ])

        if fairness_report.bias_flags:
            lines.append("")
            lines.append("Bias Indicators:")
            for flag in fairness_report.bias_flags:
                lines.append(f"  [{flag.severity.upper()}] {flag.description}")

        # Recommendations
        if fairness_report.recommendations:
            lines.extend([
                "",
                "--- RECOMMENDATIONS ---",
                ""
            ])
            for rec in fairness_report.recommendations:
                lines.append(f"  - {rec}")

        # Final notes
        lines.extend([
            "",
            "--- IMPORTANT ---",
            "",
            "This is an advisory assessment with bias awareness.",
            "Final decisions MUST be made by human operators.",
        ])

        if fairness_report.requires_manual_review:
            lines.extend([
                "",
                "*** MANUAL REVIEW REQUIRED ***",
                "Bias indicators warrant additional scrutiny."
            ])

        lines.extend([
            "",
            "=" * 50
        ])

        return "\n".join(lines)

    def get_quick_summary(self, score: BiasAwareIntentScore) -> Dict:
        """Get a quick summary for display."""
        return {
            "final_score": score.final_score,
            "severity": score.severity.value,
            "confidence": score.confidence_level,
            "bias_risk": score.fairness_report.bias_metrics.overall_bias_risk.value,
            "fairness_score": score.fairness_report.overall_fairness_score,
            "requires_review": score.fairness_report.requires_manual_review,
            "num_bias_flags": len(score.fairness_report.bias_flags)
        }
