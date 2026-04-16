"""
alert_generator.py - Rule-based alert generation.

Single responsibility: decide whether an adjusted risk score warrants an
alert and, if so, construct the AlertRecord.

The alert threshold (0.50) means only scores in the MEDIUM band and above
trigger an alert. LOW and NONE never generate an alert even if the YOLO
verdict was shoplifting - this avoids flooding staff with low-confidence
notifications that are likely false positives.

Python standard library datetime is used to timestamp alert IDs:
    Python Software Foundation. datetime module.
    https://docs.python.org/3/library/datetime.html
"""
from __future__ import annotations
from datetime import datetime

from core.result_types import AlertRecord


class AlertGenerator:
    """
    Emits an AlertRecord when the adjusted risk score meets the threshold.
    All fields are deterministic given the same inputs.
    """

    # alerts only fire when score is above 50% - anything below that is
    # probably not reliable enough to show to store staff
    THRESHOLD: float = 0.50

    def generate(self,
                 adjusted_score:      float,
                 severity:            str,
                 n_suspicious_events: int) -> AlertRecord | None:
        """
        Args:
            adjusted_score: bias-adjusted intent score (0 to 1)
            severity: severity label from IntentScorer
            n_suspicious_events: total count of shoplifting + look-around windows

        Returns:
            AlertRecord if score >= threshold, else None
        """
        if adjusted_score < self.THRESHOLD:
            return None

        # alert ID format: ALERT-YYYYMMDDHHMMSS-0001
        # the -0001 suffix is a placeholder - in a real system this would be
        # a database sequence number so you can track multiple alerts per session
        return AlertRecord(
            alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-0001",
            level    = severity,
            message  = (
                f"Adjusted risk score: {adjusted_score:.2f} ({severity}). "
                "Human review required before any action."
            ),
        )
