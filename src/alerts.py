"""
Alert management and notification module for Digital Witness.
Handles alert persistence, status tracking, and email notifications.
"""
import json
import smtplib
import ssl
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Callable, Any

from .analysis import Alert, Severity
from .video import ExtractedClip
from .config import (
    ALERTS_STORAGE_DIR, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT,
    EMAIL_USE_TLS, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS
)


class AlertStatus(Enum):
    """Alert lifecycle status."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


@dataclass
class AlertNote:
    """A note attached to an alert."""
    timestamp: datetime
    user: str
    note: str

    def to_dict(self) -> Dict:
        return {"timestamp": self.timestamp.isoformat(), "user": self.user, "note": self.note}

    @classmethod
    def from_dict(cls, data: Dict) -> "AlertNote":
        return cls(timestamp=datetime.fromisoformat(data["timestamp"]),
                   user=data["user"], note=data["note"])


@dataclass
class ManagedAlert:
    """Alert with management metadata."""
    alert: Alert
    status: AlertStatus = AlertStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    dismissed_at: Optional[datetime] = None
    dismissed_by: Optional[str] = None
    dismiss_reason: Optional[str] = None
    escalated_at: Optional[datetime] = None
    escalated_by: Optional[str] = None
    escalate_reason: Optional[str] = None
    notes: List[AlertNote] = field(default_factory=list)
    notification_sent: bool = False
    notification_sent_at: Optional[datetime] = None

    @property
    def alert_id(self) -> str:
        return self.alert.alert_id

    @property
    def severity(self) -> Severity:
        return self.alert.severity

    @property
    def intent_score(self) -> float:
        return self.alert.intent_score

    def acknowledge(self, user: str) -> None:
        if self.status == AlertStatus.PENDING:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledged_at = datetime.now()
            self.acknowledged_by = user
            self.add_note(user, "Alert acknowledged")

    def dismiss(self, user: str, reason: str) -> None:
        self.status = AlertStatus.DISMISSED
        self.dismissed_at = datetime.now()
        self.dismissed_by = user
        self.dismiss_reason = reason
        self.add_note(user, f"Alert dismissed: {reason}")

    def escalate(self, user: str, reason: str) -> None:
        self.status = AlertStatus.ESCALATED
        self.escalated_at = datetime.now()
        self.escalated_by = user
        self.escalate_reason = reason
        self.add_note(user, f"Alert escalated: {reason}")

    def add_note(self, user: str, note: str) -> None:
        self.notes.append(AlertNote(timestamp=datetime.now(), user=user, note=note))

    def mark_notification_sent(self) -> None:
        self.notification_sent = True
        self.notification_sent_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "alert": self.alert.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "dismissed_at": self.dismissed_at.isoformat() if self.dismissed_at else None,
            "dismissed_by": self.dismissed_by,
            "dismiss_reason": self.dismiss_reason,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "escalated_by": self.escalated_by,
            "escalate_reason": self.escalate_reason,
            "notes": [note.to_dict() for note in self.notes],
            "notification_sent": self.notification_sent,
            "notification_sent_at": self.notification_sent_at.isoformat() if self.notification_sent_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ManagedAlert":
        alert_data = data["alert"]
        evidence_clips = []
        for clip_data in alert_data.get("evidence_clips", []):
            evidence_clips.append(ExtractedClip(
                path=Path(clip_data["path"]),
                start_time=clip_data["start_time"],
                end_time=clip_data["end_time"],
                event_time=clip_data.get("event_time", clip_data["start_time"]),
                event_type=clip_data["event_type"]
            ))
        alert = Alert(
            alert_id=alert_data["alert_id"],
            timestamp=datetime.fromisoformat(alert_data["timestamp"]),
            severity=Severity(alert_data["severity"]),
            intent_score=alert_data["intent_score"],
            explanation=alert_data["explanation"],
            evidence_clips=evidence_clips,
            discrepancy_summary=alert_data["discrepancy_summary"],
            behavior_summary=alert_data["behavior_summary"],
            requires_human_review=alert_data.get("requires_human_review", True)
        )
        return cls(
            alert=alert,
            status=AlertStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            acknowledged_by=data.get("acknowledged_by"),
            dismissed_at=datetime.fromisoformat(data["dismissed_at"]) if data.get("dismissed_at") else None,
            dismissed_by=data.get("dismissed_by"),
            dismiss_reason=data.get("dismiss_reason"),
            escalated_at=datetime.fromisoformat(data["escalated_at"]) if data.get("escalated_at") else None,
            escalated_by=data.get("escalated_by"),
            escalate_reason=data.get("escalate_reason"),
            notes=[AlertNote.from_dict(n) for n in data.get("notes", [])],
            notification_sent=data.get("notification_sent", False),
            notification_sent_at=datetime.fromisoformat(data["notification_sent_at"]) if data.get("notification_sent_at") else None
        )


class AlertManager:
    """Manages alert lifecycle and persistence."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or ALERTS_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._alerts: Dict[str, ManagedAlert] = {}
        self._lock = threading.Lock()
        self._on_new_alert_callbacks: List[Callable[[ManagedAlert], None]] = []
        self._load_alerts()

    def add_alert(self, alert: Alert) -> ManagedAlert:
        with self._lock:
            managed_alert = ManagedAlert(alert=alert)
            self._alerts[alert.alert_id] = managed_alert
            self._save_alert(managed_alert)
        for callback in self._on_new_alert_callbacks:
            callback(managed_alert)
        return managed_alert

    def get_alert(self, alert_id: str) -> Optional[ManagedAlert]:
        with self._lock:
            return self._alerts.get(alert_id)

    def get_all_alerts(self) -> List[ManagedAlert]:
        with self._lock:
            return list(self._alerts.values())

    def get_pending_alerts(self) -> List[ManagedAlert]:
        with self._lock:
            return [a for a in self._alerts.values() if a.status == AlertStatus.PENDING]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False
            alert.acknowledge(user)
            self._save_alert(alert)
        return True

    def dismiss_alert(self, alert_id: str, user: str, reason: str) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False
            alert.dismiss(user, reason)
            self._save_alert(alert)
        return True

    def escalate_alert(self, alert_id: str, user: str, reason: str) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False
            alert.escalate(user, reason)
            self._save_alert(alert)
        return True

    def add_note(self, alert_id: str, user: str, note: str) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False
            alert.add_note(user, note)
            self._save_alert(alert)
        return True

    def mark_notification_sent(self, alert_id: str) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return False
            alert.mark_notification_sent()
            self._save_alert(alert)
        return True

    def on_new_alert(self, callback: Callable[[ManagedAlert], None]) -> None:
        self._on_new_alert_callbacks.append(callback)

    def _get_alert_path(self, alert_id: str) -> Path:
        return self.storage_dir / f"{alert_id}.json"

    def _save_alert(self, alert: ManagedAlert) -> None:
        path = self._get_alert_path(alert.alert_id)
        with open(path, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)

    def _load_alerts(self) -> None:
        for path in self.storage_dir.glob("ALERT-*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                alert = ManagedAlert.from_dict(data)
                self._alerts[alert.alert_id] = alert
            except Exception as e:
                print(f"[AlertManager] Failed to load {path}: {e}")

    def clear_old_alerts(self, days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        with self._lock:
            to_remove = [aid for aid, a in self._alerts.items() if a.created_at < cutoff]
            for alert_id in to_remove:
                del self._alerts[alert_id]
                path = self._get_alert_path(alert_id)
                if path.exists():
                    path.unlink()
                removed += 1
        return removed

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            alerts = list(self._alerts.values())
            status_counts = {s.value: sum(1 for a in alerts if a.status == s) for s in AlertStatus}
            severity_counts = {s.value: sum(1 for a in alerts if a.severity == s) for s in Severity}
            return {
                "total_alerts": len(alerts),
                "by_status": status_counts,
                "by_severity": severity_counts,
                "pending_count": status_counts.get("pending", 0),
                "notifications_sent": sum(1 for a in alerts if a.notification_sent)
            }


@dataclass
class EmailConfig:
    smtp_server: str = EMAIL_SMTP_SERVER
    smtp_port: int = EMAIL_SMTP_PORT
    sender_email: str = EMAIL_SENDER
    sender_password: str = EMAIL_PASSWORD
    recipient_emails: List[str] = None
    use_tls: bool = EMAIL_USE_TLS

    def __post_init__(self):
        if self.recipient_emails is None:
            self.recipient_emails = EMAIL_RECIPIENTS if EMAIL_RECIPIENTS else []

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_server and self.sender_email and
                    self.sender_password and self.recipient_emails)


class NotificationService:
    """Email notification service for alerts."""

    def __init__(self, email_config: Optional[EmailConfig] = None):
        self.email_config = email_config or EmailConfig()

    def send_email_alert(self, alert: ManagedAlert) -> bool:
        if not self.email_config.is_configured:
            print("[NotificationService] Email not configured")
            return False
        try:
            message = self._create_email_message(alert)
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls(context=context)
                server.login(self.email_config.sender_email, self.email_config.sender_password)
                server.sendmail(self.email_config.sender_email,
                                self.email_config.recipient_emails, message.as_string())
            print(f"[NotificationService] Email sent to {len(self.email_config.recipient_emails)} recipient(s)")
            return True
        except Exception as e:
            print(f"[NotificationService] Error: {e}")
            return False

    def _create_email_message(self, alert: ManagedAlert) -> MIMEMultipart:
        message = MIMEMultipart("alternative")
        severity_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️", "NONE": "✓"}
        emoji = severity_emoji.get(alert.severity.value, "📢")
        message["Subject"] = f"{emoji} Digital Witness Alert: {alert.severity.value} Risk - {alert.alert_id}"
        message["From"] = self.email_config.sender_email
        message["To"] = ", ".join(self.email_config.recipient_emails)

        text_body = f"""
DIGITAL WITNESS SECURITY ALERT
==============================
Alert ID: {alert.alert_id}
Severity: {alert.severity.value}
Risk Score: {alert.intent_score:.2f}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

{alert.alert.discrepancy_summary}
{alert.alert.behavior_summary}

ACTION REQUIRED: Please log into the Security Manager to review.
---
This is an automated message from Digital Witness.
"""
        message.attach(MIMEText(text_body, "plain"))
        return message

    def test_connection(self) -> bool:
        if not self.email_config.is_configured:
            return False
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls(context=context)
                server.login(self.email_config.sender_email, self.email_config.sender_password)
            return True
        except Exception:
            return False

    @property
    def is_email_configured(self) -> bool:
        return self.email_config.is_configured
