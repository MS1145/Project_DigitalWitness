"""
Case builder module for Digital Witness.

Compiles all analysis artifacts into a JSON case file that serves
as a complete audit trail. Each case file is self-contained and
includes all evidence needed to review and validate the analysis.
"""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from ..video.loader import VideoMetadata
from ..pose.behavior_classifier import BehaviorEvent
from ..analysis.cross_checker import DiscrepancyReport
from ..analysis.intent_scorer import IntentScore
from ..analysis.alert_generator import Alert
from ..video.clip_extractor import ExtractedClip
from ..config import CASE_OUTPUT_DIR


@dataclass
class CaseFile:
    """
    Complete audit record for a single analysis run.

    Serializable to JSON for storage and later review.
    Includes full provenance: source video, all detections,
    scoring breakdown, and generated alerts.
    """
    case_id: str
    created_at: str
    video_metadata: Dict[str, Any]
    behavior_timeline: List[Dict[str, Any]]  # Chronological behavior events
    discrepancy_report: Dict[str, Any]
    intent_score: Dict[str, Any]
    alert: Optional[Dict[str, Any]]          # None if no alert triggered
    forensic_clips: List[Dict[str, Any]]
    summary: str                             # Human-readable summary
    notes: str = ""                          # Additional notes (e.g., deep learning info)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "case_id": self.case_id,
            "created_at": self.created_at,
            "video_metadata": self.video_metadata,
            "behavior_timeline": self.behavior_timeline,
            "discrepancy_report": self.discrepancy_report,
            "intent_score": self.intent_score,
            "alert": self.alert,
            "forensic_clips": self.forensic_clips,
            "summary": self.summary,
            "notes": self.notes
        }


class CaseBuilder:
    """Assembles and persists case files from analysis results."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize case builder.

        Args:
            output_dir: Directory to save case files
        """
        self.output_dir = output_dir or CASE_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._case_counter = 0

    def build_case(
        self,
        video_metadata: VideoMetadata,
        behavior_events: List[BehaviorEvent],
        discrepancy_report: DiscrepancyReport,
        intent_score: IntentScore,
        alert: Optional[Alert],
        forensic_clips: List[ExtractedClip]
    ) -> CaseFile:
        """
        Build a complete case file.

        Args:
            video_metadata: Source video metadata
            behavior_events: Detected behavior events
            discrepancy_report: POS cross-check results
            intent_score: Calculated intent score
            alert: Generated alert (if any)
            forensic_clips: Extracted video clips

        Returns:
            CaseFile object
        """
        self._case_counter += 1
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._case_counter:04d}"

        # Convert video metadata
        video_meta_dict = {
            "path": str(video_metadata.path),
            "fps": video_metadata.fps,
            "frame_count": video_metadata.frame_count,
            "duration": video_metadata.duration,
            "width": video_metadata.width,
            "height": video_metadata.height,
            "codec": video_metadata.codec
        }

        # Convert behavior timeline
        behavior_timeline = [
            {
                "behavior_type": e.behavior_type,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "confidence": e.confidence,
                "probabilities": e.probabilities
            }
            for e in behavior_events
        ]

        # Convert discrepancy report
        discrepancy_dict = {
            "missing_from_billing": discrepancy_report.missing_from_billing,
            "extra_in_billing": discrepancy_report.extra_in_billing,
            "matched_items": discrepancy_report.matched_items,
            "total_detected": discrepancy_report.total_detected,
            "total_billed": discrepancy_report.total_billed,
            "discrepancy_count": discrepancy_report.discrepancy_count,
            "match_rate": discrepancy_report.match_rate
        }

        # Convert intent score
        intent_dict = {
            "score": intent_score.score,
            "severity": intent_score.severity.value,
            "components": intent_score.components,
            "explanation": intent_score.explanation
        }

        # Convert alert
        alert_dict = alert.to_dict() if alert else None

        # Convert forensic clips
        clips_list = [
            {
                "path": str(clip.path),
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "event_time": clip.event_time,
                "event_type": clip.event_type
            }
            for clip in forensic_clips
        ]

        # Generate summary
        summary = self._generate_summary(
            video_metadata,
            behavior_events,
            discrepancy_report,
            intent_score,
            alert
        )

        return CaseFile(
            case_id=case_id,
            created_at=datetime.now().isoformat(),
            video_metadata=video_meta_dict,
            behavior_timeline=behavior_timeline,
            discrepancy_report=discrepancy_dict,
            intent_score=intent_dict,
            alert=alert_dict,
            forensic_clips=clips_list,
            summary=summary
        )

    def save_case(self, case: CaseFile) -> Path:
        """
        Save case file to disk.

        Args:
            case: CaseFile to save

        Returns:
            Path to saved file
        """
        filename = f"{case.case_id}.json"
        output_path = self.output_dir / filename

        # Convert to dict
        case_dict = case.to_dict()

        with open(output_path, 'w') as f:
            json.dump(case_dict, f, indent=2)

        return output_path

    def _generate_summary(
        self,
        video_metadata: VideoMetadata,
        behavior_events: List[BehaviorEvent],
        discrepancy_report: DiscrepancyReport,
        intent_score: IntentScore,
        alert: Optional[Alert]
    ) -> str:
        """Generate human-readable case summary."""
        lines = [
            "CASE SUMMARY",
            "=" * 40,
            "",
            f"Video: {video_metadata.path.name}",
            f"Duration: {video_metadata.duration:.1f} seconds",
            "",
            "--- ANALYSIS RESULTS ---",
            "",
            f"Total behaviors detected: {len(behavior_events)}",
        ]

        # Count behaviors by type
        behavior_counts = {}
        for event in behavior_events:
            t = event.behavior_type
            behavior_counts[t] = behavior_counts.get(t, 0) + 1

        for btype, count in behavior_counts.items():
            lines.append(f"  - {btype}: {count}")

        lines.extend([
            "",
            f"Items detected: {discrepancy_report.total_detected}",
            f"Items billed: {discrepancy_report.total_billed}",
            f"Discrepancies: {discrepancy_report.discrepancy_count}",
            f"Match rate: {discrepancy_report.match_rate:.1%}",
            "",
            f"Intent Score: {intent_score.score:.2f}",
            f"Severity: {intent_score.severity.value}",
            ""
        ])

        if alert:
            lines.extend([
                "--- ALERT GENERATED ---",
                f"Alert ID: {alert.alert_id}",
                f"Requires human review: Yes",
            ])
        else:
            lines.append("No alert generated.")

        lines.extend([
            "",
            "=" * 40,
            "END OF SUMMARY"
        ])

        return "\n".join(lines)

    def load_case(self, case_path: Path) -> Dict:
        """
        Load a case file from disk.

        Args:
            case_path: Path to case JSON file

        Returns:
            Case data as dictionary
        """
        with open(case_path, 'r') as f:
            return json.load(f)

    def list_cases(self) -> List[Path]:
        """List all case files in output directory."""
        return sorted(self.output_dir.glob("CASE-*.json"))
