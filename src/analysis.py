"""
Analysis module for Digital Witness.
Handles cross-checking, intent scoring, and alert generation.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Set, Optional
from enum import Enum

from .config import (
    WEIGHT_DISCREPANCY, WEIGHT_CONCEALMENT, WEIGHT_BYPASS, WEIGHT_DURATION,
    INTENT_THRESHOLD_LOW, INTENT_THRESHOLD_MEDIUM, INTENT_THRESHOLD_HIGH,
    ALERT_THRESHOLD
)


class Severity(Enum):
    """Severity tiers for alert prioritization."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ProductInteraction:
    """A product pickup detected from video analysis."""
    sku: str
    timestamp: float
    interaction_type: str
    confidence: float


@dataclass
class DiscrepancyReport:
    """Reconciliation result between video and POS data."""
    missing_from_billing: List[str]
    extra_in_billing: List[str]
    matched_items: List[str]
    total_detected: int
    total_billed: int
    discrepancy_count: int
    match_rate: float


@dataclass
class IntentScore:
    """Composite risk assessment result."""
    score: float
    severity: Severity
    components: dict
    explanation: str


@dataclass
class Alert:
    """Packaged alert for human operator review."""
    alert_id: str
    timestamp: datetime
    severity: Severity
    intent_score: float
    explanation: str
    evidence_clips: List  # List of ExtractedClip
    discrepancy_summary: str
    behavior_summary: str
    requires_human_review: bool = True

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "intent_score": self.intent_score,
            "explanation": self.explanation,
            "evidence_clips": [
                {"path": str(clip.path), "start_time": clip.start_time,
                 "end_time": clip.end_time, "event_type": clip.event_type}
                for clip in self.evidence_clips
            ],
            "discrepancy_summary": self.discrepancy_summary,
            "behavior_summary": self.behavior_summary,
            "requires_human_review": self.requires_human_review
        }


class CrossChecker:
    """Reconciles video-detected product interactions with POS billing."""

    def check_discrepancies(self, detected_interactions: List[ProductInteraction],
                            transactions: List) -> DiscrepancyReport:
        detected_skus = {i.sku for i in detected_interactions if i.interaction_type == "pickup"}
        billed_skus = set()
        for txn in transactions:
            for item in txn.items:
                billed_skus.add(item.sku)

        missing_from_billing = list(detected_skus - billed_skus)
        extra_in_billing = list(billed_skus - detected_skus)
        matched_items = list(detected_skus & billed_skus)
        total_detected = len(detected_skus)
        total_billed = len(billed_skus)
        match_rate = len(matched_items) / total_detected if total_detected > 0 else 1.0

        return DiscrepancyReport(
            missing_from_billing=missing_from_billing,
            extra_in_billing=extra_in_billing,
            matched_items=matched_items,
            total_detected=total_detected,
            total_billed=total_billed,
            discrepancy_count=len(missing_from_billing),
            match_rate=match_rate
        )


class IntentScorer:
    """Multi-factor risk scoring engine."""

    def __init__(self, weight_discrepancy: float = WEIGHT_DISCREPANCY,
                 weight_concealment: float = WEIGHT_CONCEALMENT,
                 weight_bypass: float = WEIGHT_BYPASS,
                 weight_duration: float = WEIGHT_DURATION):
        total = weight_discrepancy + weight_concealment + weight_bypass + weight_duration
        self.weight_discrepancy = weight_discrepancy / total
        self.weight_concealment = weight_concealment / total
        self.weight_bypass = weight_bypass / total
        self.weight_duration = weight_duration / total

    def calculate_score(self, discrepancy_report: DiscrepancyReport,
                        behavior_events: List, video_duration: float) -> IntentScore:
        discrepancy_score = self._score_discrepancy(discrepancy_report)
        concealment_score = self._score_concealment(behavior_events)
        bypass_score = self._score_bypass(behavior_events)
        duration_score = self._score_duration(behavior_events, video_duration)

        total_score = (
            self.weight_discrepancy * discrepancy_score +
            self.weight_concealment * concealment_score +
            self.weight_bypass * bypass_score +
            self.weight_duration * duration_score
        )
        total_score = max(0.0, min(1.0, total_score))
        severity = self._determine_severity(total_score)

        components = {
            "discrepancy": {"score": discrepancy_score, "weight": self.weight_discrepancy},
            "concealment": {"score": concealment_score, "weight": self.weight_concealment},
            "bypass": {"score": bypass_score, "weight": self.weight_bypass},
            "duration": {"score": duration_score, "weight": self.weight_duration}
        }

        explanation = self._generate_explanation(total_score, severity, discrepancy_report, behavior_events)
        return IntentScore(score=total_score, severity=severity, components=components, explanation=explanation)

    def _score_discrepancy(self, report: DiscrepancyReport) -> float:
        if report.total_detected == 0:
            return 0.0
        return min(1.0, (report.discrepancy_count / report.total_detected) * 1.5)

    def _score_concealment(self, events: List) -> float:
        concealment_events = [e for e in events if e.behavior_type == "concealment"]
        if not concealment_events:
            return 0.0
        avg_confidence = sum(e.confidence for e in concealment_events) / len(concealment_events)
        return avg_confidence * min(1.0, len(concealment_events) / 3)

    def _score_bypass(self, events: List) -> float:
        bypass_events = [e for e in events if e.behavior_type == "bypass"]
        return max((e.confidence for e in bypass_events), default=0.0)

    def _score_duration(self, events: List, video_duration: float) -> float:
        suspicious = [e for e in events if e.behavior_type in {"concealment", "bypass"}]
        if not suspicious or video_duration <= 0:
            return 0.0
        duration = sum(e.end_time - e.start_time for e in suspicious)
        return min(1.0, (duration / video_duration) * 3)

    def _determine_severity(self, score: float) -> Severity:
        if score < INTENT_THRESHOLD_LOW:
            return Severity.NONE
        elif score < INTENT_THRESHOLD_MEDIUM:
            return Severity.LOW
        elif score < INTENT_THRESHOLD_HIGH:
            return Severity.MEDIUM
        elif score < 0.85:
            return Severity.HIGH
        return Severity.CRITICAL

    def _generate_explanation(self, score: float, severity: Severity,
                               report: DiscrepancyReport, events: List) -> str:
        lines = [f"Intent Score: {score:.2f} ({severity.value})", ""]
        if report.discrepancy_count > 0:
            lines.append(f"- {report.discrepancy_count} item(s) detected but not billed")
        concealment_count = sum(1 for e in events if e.behavior_type == "concealment")
        if concealment_count > 0:
            lines.append(f"- {concealment_count} concealment behavior(s) detected")
        bypass_count = sum(1 for e in events if e.behavior_type == "bypass")
        if bypass_count > 0:
            lines.append(f"- {bypass_count} checkout bypass behavior(s) detected")
        lines.append("")
        lines.append("NOTE: Final decision must be made by human operator.")
        return "\n".join(lines)


class AlertGenerator:
    """Generates alerts based on intent scores and evidence."""

    def __init__(self, threshold: float = ALERT_THRESHOLD):
        self.threshold = threshold
        self._alert_counter = 0

    def should_generate_alert(self, intent_score: IntentScore) -> bool:
        return intent_score.score >= self.threshold

    def generate_alert(self, intent_score: IntentScore, discrepancy_report: DiscrepancyReport,
                       behavior_events: List, evidence_clips: List) -> Optional[Alert]:
        if not self.should_generate_alert(intent_score):
            return None

        self._alert_counter += 1
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

        discrepancy_summary = self._generate_discrepancy_summary(discrepancy_report)
        behavior_summary = self._generate_behavior_summary(behavior_events)
        explanation = self._build_explanation(intent_score, discrepancy_report, behavior_events)

        return Alert(
            alert_id=alert_id, timestamp=datetime.now(), severity=intent_score.severity,
            intent_score=intent_score.score, explanation=explanation,
            evidence_clips=evidence_clips, discrepancy_summary=discrepancy_summary,
            behavior_summary=behavior_summary, requires_human_review=True
        )

    def _generate_discrepancy_summary(self, report: DiscrepancyReport) -> str:
        if report.discrepancy_count == 0:
            return "No billing discrepancies detected."
        lines = [f"{report.discrepancy_count} item(s) detected but not billed:"]
        for sku in report.missing_from_billing:
            lines.append(f"  - {sku}")
        lines.append(f"\nMatch rate: {report.match_rate:.1%}")
        return "\n".join(lines)

    def _generate_behavior_summary(self, events: List) -> str:
        counts = {}
        for event in events:
            counts[event.behavior_type] = counts.get(event.behavior_type, 0) + 1
        if not counts:
            return "No behaviors detected."
        lines = ["Detected behaviors:"]
        for behavior_type, count in counts.items():
            lines.append(f"  - {behavior_type}: {count} occurrence(s)")
        return "\n".join(lines)

    def _build_explanation(self, intent_score: IntentScore, discrepancy_report: DiscrepancyReport,
                           behavior_events: List) -> str:
        lines = [
            "=" * 50, "DIGITAL WITNESS ALERT", "=" * 50, "",
            f"Severity: {intent_score.severity.value}",
            f"Intent Score: {intent_score.score:.2f}", "",
            "--- WHY THIS ALERT WAS GENERATED ---", ""
        ]
        if discrepancy_report.discrepancy_count > 0:
            lines.append(f"1. {discrepancy_report.discrepancy_count} product(s) detected but not billed.")
        concealment = [e for e in behavior_events if e.behavior_type == "concealment"]
        if concealment:
            avg_conf = sum(e.confidence for e in concealment) / len(concealment)
            lines.append(f"2. {len(concealment)} concealment behavior(s) detected (avg: {avg_conf:.1%}).")
        bypass = [e for e in behavior_events if e.behavior_type == "bypass"]
        if bypass:
            lines.append(f"3. {len(bypass)} checkout bypass behavior(s) detected.")
        lines.extend([
            "", "--- REQUIRED ACTION ---", "",
            "This alert requires HUMAN REVIEW before any action is taken.",
            "=" * 50
        ])
        return "\n".join(lines)
