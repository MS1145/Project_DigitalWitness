"""
Evidence compiler for Digital Witness.

Assembles all forensic evidence into a complete package:
- PDF report
- Annotated video clips
- Evidence screenshots
- Raw data files
- Timeline visualizations

This package provides everything needed for case review.
"""
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .report_generator import ReportGenerator, ForensicReport
from ..video.annotated_clip_generator import AnnotatedClipGenerator, AnnotatedClip
from ..config import FORENSIC_PACKAGES_DIR


@dataclass
class ForensicPackage:
    """Complete forensic evidence package."""
    case_id: str
    package_dir: Path
    created_at: str

    # Report
    report: ForensicReport
    report_path: Path

    # Video evidence
    annotated_clips: List[AnnotatedClip]
    clip_paths: List[Path]

    # Screenshots
    screenshots: List[Path]

    # Data files
    case_json_path: Path
    timeline_json_path: Optional[Path]

    # Summary
    total_files: int
    package_size_mb: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "package_dir": str(self.package_dir),
            "created_at": self.created_at,
            "report_path": str(self.report_path),
            "clip_count": len(self.clip_paths),
            "screenshot_count": len(self.screenshots),
            "total_files": self.total_files,
            "package_size_mb": self.package_size_mb
        }


class EvidenceCompiler:
    """
    Compiles all analysis outputs into a forensic evidence package.

    Creates a self-contained directory with all materials needed
    for case review, suitable for archiving or sharing.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize evidence compiler.

        Args:
            output_dir: Base directory for packages
        """
        self.output_dir = output_dir or FORENSIC_PACKAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_generator = ReportGenerator()
        self.clip_generator = AnnotatedClipGenerator()

    def compile_package(
        self,
        case_data: Dict[str, Any],
        video_path: Optional[Path] = None,
        pose_results: Optional[List] = None,
        behavior_events: Optional[List] = None,
        quality_report: Optional[Any] = None,
        fairness_report: Optional[Any] = None,
        generate_clips: bool = True,
        generate_screenshots: bool = True,
        max_clips: int = 5
    ) -> ForensicPackage:
        """
        Compile a complete forensic evidence package.

        Args:
            case_data: Case file data dictionary
            video_path: Path to source video
            pose_results: Pose estimation results
            behavior_events: Behavior classification events
            quality_report: Video quality report
            fairness_report: Fairness/bias report
            generate_clips: Generate annotated video clips
            generate_screenshots: Generate evidence screenshots
            max_clips: Maximum clips to generate

        Returns:
            ForensicPackage with all paths and metadata
        """
        case_id = case_data.get("case_id", f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}")

        # Create package directory
        package_dir = self.output_dir / case_id
        package_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        clips_dir = package_dir / "clips"
        screenshots_dir = package_dir / "screenshots"
        data_dir = package_dir / "data"

        clips_dir.mkdir(exist_ok=True)
        screenshots_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        # Add fairness report to case data if available
        if fairness_report:
            if hasattr(fairness_report, 'to_dict'):
                case_data["fairness_report"] = fairness_report.to_dict()
            else:
                case_data["fairness_report"] = fairness_report

        # Add quality report to case data if available
        if quality_report:
            if hasattr(quality_report, 'to_dict'):
                case_data["quality_report"] = quality_report.to_dict()
            else:
                case_data["quality_report"] = quality_report

        # Generate annotated clips
        annotated_clips = []
        clip_paths = []

        if generate_clips and video_path and behavior_events and pose_results:
            # Update clip generator output directory
            self.clip_generator.output_dir = clips_dir

            annotated_clips = self.clip_generator.generate_clips_for_events(
                video_path,
                behavior_events,
                pose_results,
                max_clips=max_clips
            )
            clip_paths = [clip.path for clip in annotated_clips]

        # Generate screenshots
        screenshots = []

        if generate_screenshots and video_path and behavior_events:
            screenshots = self._generate_screenshots(
                video_path,
                behavior_events,
                pose_results,
                screenshots_dir
            )

        # Generate PDF report
        report = self.report_generator.generate_report(
            case_data,
            screenshots=screenshots,
            include_timeline=True,
            include_fairness=fairness_report is not None
        )

        # Copy report to package
        report_dest = package_dir / report.pdf_path.name
        if report.pdf_path != report_dest:
            shutil.copy(report.pdf_path, report_dest)
        report_path = report_dest

        # Save case data JSON
        case_json_path = data_dir / "case_data.json"
        with open(case_json_path, 'w') as f:
            json.dump(case_data, f, indent=2, default=str)

        # Save timeline JSON
        timeline_json_path = None
        timeline = case_data.get("behavior_timeline", [])
        if timeline:
            timeline_json_path = data_dir / "timeline.json"
            with open(timeline_json_path, 'w') as f:
                json.dump(timeline, f, indent=2)

        # Generate timeline visualization
        self._generate_timeline_image(timeline, package_dir / "timeline.png")

        # Create package manifest
        manifest = self._create_manifest(
            case_id, package_dir, report_path, clip_paths, screenshots
        )
        manifest_path = package_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Calculate package size
        total_size = sum(
            f.stat().st_size for f in package_dir.rglob("*") if f.is_file()
        )
        package_size_mb = total_size / (1024 * 1024)

        # Count total files
        total_files = sum(1 for f in package_dir.rglob("*") if f.is_file())

        return ForensicPackage(
            case_id=case_id,
            package_dir=package_dir,
            created_at=datetime.now().isoformat(),
            report=report,
            report_path=report_path,
            annotated_clips=annotated_clips,
            clip_paths=clip_paths,
            screenshots=screenshots,
            case_json_path=case_json_path,
            timeline_json_path=timeline_json_path,
            total_files=total_files,
            package_size_mb=package_size_mb
        )

    def _generate_screenshots(
        self,
        video_path: Path,
        behavior_events: List,
        pose_results: Optional[List],
        output_dir: Path
    ) -> List[Path]:
        """Generate evidence screenshots for key moments."""
        import cv2

        screenshots = []
        suspicious_types = {"pickup", "concealment", "bypass", "shoplifting"}

        # Get suspicious events
        suspicious_events = [
            e for e in behavior_events
            if e.behavior_type in suspicious_types
        ]

        # Sort by confidence and take top events
        suspicious_events = sorted(
            suspicious_events,
            key=lambda e: e.confidence,
            reverse=True
        )[:10]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return screenshots

        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, event in enumerate(suspicious_events):
            # Get middle of event
            mid_time = (event.start_time + event.end_time) / 2
            frame_num = int(mid_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                output_path = output_dir / f"evidence_{i+1:02d}_{event.behavior_type}.jpg"
                cv2.imwrite(str(output_path), frame)
                screenshots.append(output_path)

        cap.release()
        return screenshots

    def _generate_timeline_image(
        self,
        timeline: List[Dict],
        output_path: Path
    ):
        """Generate a timeline visualization image."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return

        if not timeline:
            return

        # Color mapping
        colors = {
            "normal": "#00c853",
            "pickup": "#ffc107",
            "concealment": "#ff9800",
            "bypass": "#f44336",
            "shoplifting": "#d32f2f"
        }

        fig, ax = plt.subplots(figsize=(12, 3))

        # Draw timeline bars
        for event in timeline:
            start = event.get("start_time", 0)
            end = event.get("end_time", start + 1)
            behavior = event.get("behavior_type", "normal")
            color = colors.get(behavior, "#808080")

            ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.8)

        # Legend
        patches = [
            mpatches.Patch(color=color, label=behavior)
            for behavior, color in colors.items()
        ]
        ax.legend(handles=patches, loc='upper right', ncol=len(colors))

        # Labels
        ax.set_xlabel("Time (seconds)")
        ax.set_yticks([])
        ax.set_title("Behavior Timeline")

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_manifest(
        self,
        case_id: str,
        package_dir: Path,
        report_path: Path,
        clip_paths: List[Path],
        screenshots: List[Path]
    ) -> Dict:
        """Create package manifest."""
        return {
            "package_version": "1.0",
            "case_id": case_id,
            "created_at": datetime.now().isoformat(),
            "created_by": "Digital Witness Forensic System",
            "contents": {
                "report": str(report_path.name),
                "clips": [str(p.name) for p in clip_paths],
                "screenshots": [str(p.name) for p in screenshots],
                "data_files": ["case_data.json", "timeline.json"]
            },
            "legal_notice": (
                "This forensic package is for advisory purposes only. "
                "All findings require human verification. "
                "This system does NOT determine guilt."
            )
        }

    def list_packages(self) -> List[Path]:
        """List all existing forensic packages."""
        return sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )

    def get_package_info(self, case_id: str) -> Optional[Dict]:
        """Get information about a specific package."""
        package_dir = self.output_dir / case_id
        manifest_path = package_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)

        return None
