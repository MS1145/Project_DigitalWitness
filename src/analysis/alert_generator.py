"""
Alert generation module for Digital Witness.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .intent_scorer import IntentScore, Severity
from .cross_checker import DiscrepancyReport
from ..pose.behavior_classifier import BehaviorEvent
from ..video.clip_extractor import ExtractedClip
from ..config import ALERT_THRESHOLD


@dataclass
class Alert:
    """A generated alert for human review."""
    alert_id: str
    timestamp: datetime
    severity: Severity
    intent_score: float
    explanation: str
    evidence_clips: List[ExtractedClip]
    discrepancy_summary: str
    behavior_summary: str
    requires_human_review: bool = True

    def to_dict(self) -> dict:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "intent_score": self.intent_score,
            "explanation": self.explanation,
            "evidence_clips": [
                {
                    "path": str(clip.path),
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "event_type": clip.event_type
                }
                for clip in self.evidence_clips
            ],
            "discrepancy_summary": self.discrepancy_summary,
            "behavior_summary": self.behavior_summary,
            "requires_human_review": self.requires_human_review
        }


class AlertGenerator:
    """Generates alerts based on intent scores and evidence."""

    def __init__(self, threshold: float = ALERT_THRESHOLD):
        """
        Initialize alert generator.

        Args:
            threshold: Minimum intent score to generate alert
        """
        self.threshold = threshold
        self._alert_counter = 0

    def should_generate_alert(self, intent_score: IntentScore) -> bool:
        """
        Determine if an alert should be generated.

        Args:
            intent_score: Calculated intent score

        Returns:
            True if alert should be generated
        """
        return intent_score.score >= self.threshold

    def generate_alert(
        self,
        intent_score: IntentScore,
        discrepancy_report: DiscrepancyReport,
        behavior_events: List[BehaviorEvent],
        evidence_clips: List[ExtractedClip]
    ) -> Optional[Alert]:
        """
        Generate an alert if threshold is exceeded.

        Args:
            intent_score: Calculated intent score
            discrepancy_report: POS discrepancy report
            behavior_events: Detected behavior events
            evidence_clips: Extracted video clips

        Returns:
            Alert object or None if threshold not exceeded
        """
        if not self.should_generate_alert(intent_score):
            return None

        self._alert_counter += 1
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

        # Generate summaries
        discrepancy_summary = self._generate_discrepancy_summary(discrepancy_report)
        behavior_summary = self._generate_behavior_summary(behavior_events)

        # Build explanation
        explanation = self._build_explanation(
            intent_score,
            discrepancy_report,
            behavior_events
        )

        return Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=intent_score.severity,
            intent_score=intent_score.score,
            explanation=explanation,
            evidence_clips=evidence_clips,
            discrepancy_summary=discrepancy_summary,
            behavior_summary=behavior_summary,
            requires_human_review=True
        )

    def _generate_discrepancy_summary(
        self,
        report: DiscrepancyReport
    ) -> str:
        """Generate summary of POS discrepancies."""
        if report.discrepancy_count == 0:
            return "No billing discrepancies detected."

        lines = [
            f"{report.discrepancy_count} item(s) detected but not billed:",
        ]
        for sku in report.missing_from_billing:
            lines.append(f"  - {sku}")

        lines.append(f"\nMatch rate: {report.match_rate:.1%}")

        return "\n".join(lines)

    def _generate_behavior_summary(
        self,
        events: List[BehaviorEvent]
    ) -> str:
        """Generate summary of detected behaviors."""
        # Count by type
        counts = {}
        for event in events:
            behavior_type = event.behavior_type
            if behavior_type not in counts:
                counts[behavior_type] = 0
            counts[behavior_type] += 1

        if not counts:
            return "No behaviors detected."

        lines = ["Detected behaviors:"]
        for behavior_type, count in counts.items():
            lines.append(f"  - {behavior_type}: {count} occurrence(s)")

        # Highlight suspicious behaviors
        suspicious = ["concealment", "bypass"]
        suspicious_count = sum(counts.get(s, 0) for s in suspicious)
        if suspicious_count > 0:
            lines.append(f"\nSuspicious behaviors: {suspicious_count}")

        return "\n".join(lines)

    def _build_explanation(
        self,
        intent_score: IntentScore,
        discrepancy_report: DiscrepancyReport,
        behavior_events: List[BehaviorEvent]
    ) -> str:
        """Build comprehensive explanation for the alert."""
        lines = [
            "=" * 50,
            "DIGITAL WITNESS ALERT",
            "=" * 50,
            "",
            f"Severity: {intent_score.severity.value}",
            f"Intent Score: {intent_score.score:.2f}",
            "",
            "--- WHY THIS ALERT WAS GENERATED ---",
            ""
        ]

        # Reasons
        reasons = []

        if discrepancy_report.discrepancy_count > 0:
            reasons.append(
                f"1. {discrepancy_report.discrepancy_count} product(s) were detected "
                f"being picked up but were not found in the billing records."
            )

        concealment_events = [
            e for e in behavior_events
            if e.behavior_type == "concealment"
        ]
        if concealment_events:
            avg_conf = sum(e.confidence for e in concealment_events) / len(concealment_events)
            reasons.append(
                f"2. {len(concealment_events)} concealment behavior(s) detected "
                f"(avg confidence: {avg_conf:.1%})."
            )

        bypass_events = [
            e for e in behavior_events
            if e.behavior_type == "bypass"
        ]
        if bypass_events:
            reasons.append(
                f"3. {len(bypass_events)} checkout bypass behavior(s) detected."
            )

        if reasons:
            lines.extend(reasons)
        else:
            lines.append("Alert generated based on cumulative risk factors.")

        lines.extend([
            "",
            "--- EVIDENCE ---",
            "",
            f"Products not billed: {', '.join(discrepancy_report.missing_from_billing) or 'None'}",
            f"Total detected: {discrepancy_report.total_detected}",
            f"Total billed: {discrepancy_report.total_billed}",
            "",
            "--- REQUIRED ACTION ---",
            "",
            "This alert requires HUMAN REVIEW before any action is taken.",
            "The system provides advisory information only.",
            "Final decisions must be made by authorized personnel.",
            "",
            "=" * 50
        ])

        return "\n".join(lines)

    def format_for_display(self, alert: Alert) -> str:
        """Format alert for console display."""
        lines = [
            "",
            "=" * 60,
            f"  ALERT: {alert.alert_id}",
            "=" * 60,
            f"  Severity: {alert.severity.value}",
            f"  Score: {alert.intent_score:.2f}",
            f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 60,
            "",
            alert.explanation,
            "",
            "Evidence clips:",
        ]

        for clip in alert.evidence_clips:
            lines.append(f"  - {clip.path.name} ({clip.event_type})")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
