"""
alert_generator.py - Rule-based alert generation.

Single responsibility: decide whether an adjusted risk score warrants an
alert and, if so, construct the AlertRecord.
"""
from __future__ import annotations
from datetime import datetime

from core.result_types import AlertRecord


class AlertGenerator:
    """
    Emits an AlertRecord when the adjusted risk score meets the threshold.
    All fields are deterministic given the same inputs.
    """

    THRESHOLD: float = 0.50

    def generate(self,
                 adjusted_score:     float,
                 severity:           str,
                 n_suspicious_events: int) -> AlertRecord | None:
        """
        Args:
            adjusted_score: bias-adjusted intent score (0–1)
            severity: severity label from IntentScorer
            n_suspicious_events: total count of shoplifting + look-around windows

        Returns:
            AlertRecord if score ≥ threshold, else None
        """
        if adjusted_score < self.THRESHOLD:
            return None

        return AlertRecord(
            alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-0001",
            level    = severity,
            message  = (
                f"Adjusted risk score: {adjusted_score:.2f} ({severity}). "
                "Human review required before any action."
            ),
        )
