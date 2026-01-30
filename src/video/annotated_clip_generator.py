"""
Annotated clip generation for Digital Witness.

Generates video clips with visual overlays including:
- Pose skeleton visualization
- Behavior labels
- Confidence indicators
- Timestamp overlays
- Event highlighting

These annotated clips provide clear forensic evidence for review.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from ..pose.estimator import PoseResult, LANDMARK_NAMES
from ..pose.behavior_classifier import BehaviorEvent
from ..config import (
    CLIP_OUTPUT_DIR,
    ANNOTATION_COLORS,
    CLIP_BUFFER_BEFORE,
    CLIP_BUFFER_AFTER
)


@dataclass
class AnnotationConfig:
    """Configuration for clip annotations."""
    show_pose_skeleton: bool = True
    show_behavior_label: bool = True
    show_confidence_bar: bool = True
    show_timestamp: bool = True
    show_frame_number: bool = True
    highlight_suspicious: bool = True
    skeleton_thickness: int = 2
    label_font_scale: float = 0.7
    label_padding: int = 10


@dataclass
class AnnotatedClip:
    """Generated annotated video clip."""
    path: Path
    start_time: float
    end_time: float
    event_time: float
    event_type: str
    duration: float
    frame_count: int
    annotations_applied: List[str]


class AnnotatedClipGenerator:
    """
    Generates video clips with forensic annotations.

    Creates clips around suspicious events with visual overlays
    that help reviewers understand what the system detected.
    """

    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ]

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[AnnotationConfig] = None
    ):
        """
        Initialize annotated clip generator.

        Args:
            output_dir: Directory to save clips
            config: Annotation configuration
        """
        self.output_dir = output_dir or CLIP_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AnnotationConfig()

        # Color scheme
        self.colors = ANNOTATION_COLORS

    def generate_clip(
        self,
        video_path: Path,
        event: BehaviorEvent,
        pose_results: List[PoseResult],
        clip_id: str,
        buffer_before: float = CLIP_BUFFER_BEFORE,
        buffer_after: float = CLIP_BUFFER_AFTER
    ) -> Optional[AnnotatedClip]:
        """
        Generate an annotated clip for a behavior event.

        Args:
            video_path: Source video path
            event: Behavior event to clip
            pose_results: Pose estimation results
            clip_id: Unique identifier for clip
            buffer_before: Seconds of context before event
            buffer_after: Seconds of context after event

        Returns:
            AnnotatedClip object or None if generation failed
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate clip boundaries
        start_time = max(0, event.start_time - buffer_before)
        end_time = min(total_frames / fps, event.end_time + buffer_after)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Build pose lookup by frame number
        pose_by_frame = {pr.frame_number: pr for pr in pose_results}

        # Setup output
        output_path = self.output_dir / f"annotated_{clip_id}_{event.behavior_type}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        annotations_applied = []

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            pose = pose_by_frame.get(frame_num)

            # Apply annotations
            annotated_frame = self._annotate_frame(
                frame,
                pose,
                event,
                timestamp,
                frame_num,
                width,
                height
            )

            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()

        # Track which annotations were applied
        if self.config.show_pose_skeleton:
            annotations_applied.append("pose_skeleton")
        if self.config.show_behavior_label:
            annotations_applied.append("behavior_label")
        if self.config.show_confidence_bar:
            annotations_applied.append("confidence_bar")
        if self.config.show_timestamp:
            annotations_applied.append("timestamp")

        return AnnotatedClip(
            path=output_path,
            start_time=start_time,
            end_time=end_time,
            event_time=event.start_time,
            event_type=event.behavior_type,
            duration=end_time - start_time,
            frame_count=frame_count,
            annotations_applied=annotations_applied
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        pose: Optional[PoseResult],
        event: BehaviorEvent,
        timestamp: float,
        frame_num: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """Apply all annotations to a frame."""
        annotated = frame.copy()

        # Check if we're in the event time range
        in_event = event.start_time <= timestamp <= event.end_time

        # Highlight suspicious frames
        if self.config.highlight_suspicious and in_event:
            if event.behavior_type in ["concealment", "bypass", "shoplifting"]:
                # Add red border
                cv2.rectangle(annotated, (0, 0), (width-1, height-1),
                            (0, 0, 255), 5)

        # Draw pose skeleton
        if self.config.show_pose_skeleton and pose and pose.landmarks:
            self._draw_skeleton(annotated, pose, width, height)

        # Draw behavior label
        if self.config.show_behavior_label:
            self._draw_behavior_label(annotated, event, in_event, width)

        # Draw confidence bar
        if self.config.show_confidence_bar:
            self._draw_confidence_bar(annotated, event.confidence, width, height)

        # Draw timestamp
        if self.config.show_timestamp:
            self._draw_timestamp(annotated, timestamp, height)

        # Draw frame number
        if self.config.show_frame_number:
            self._draw_frame_number(annotated, frame_num, height)

        return annotated

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        width: int,
        height: int
    ):
        """Draw pose skeleton on frame."""
        landmarks = pose.landmarks
        if not landmarks:
            return

        # Get color based on visibility
        def get_point_color(landmark):
            if landmark.visibility > 0.7:
                return (0, 255, 0)  # Green - high confidence
            elif landmark.visibility > 0.5:
                return (0, 255, 255)  # Yellow - medium confidence
            else:
                return (0, 0, 255)  # Red - low confidence

        # Draw points
        for name, lm in landmarks.items():
            x = int(lm.x * width)
            y = int(lm.y * height)
            color = get_point_color(lm)
            cv2.circle(frame, (x, y), 5, color, -1)

        # Draw connections
        for start_name, end_name in self.SKELETON_CONNECTIONS:
            if start_name in landmarks and end_name in landmarks:
                start = landmarks[start_name]
                end = landmarks[end_name]

                # Only draw if both points visible
                if start.visibility > 0.3 and end.visibility > 0.3:
                    pt1 = (int(start.x * width), int(start.y * height))
                    pt2 = (int(end.x * width), int(end.y * height))
                    cv2.line(frame, pt1, pt2, (255, 255, 255),
                            self.config.skeleton_thickness)

    def _draw_behavior_label(
        self,
        frame: np.ndarray,
        event: BehaviorEvent,
        in_event: bool,
        width: int
    ):
        """Draw behavior label at top of frame."""
        behavior = event.behavior_type.upper()
        confidence = event.confidence

        # Get color for behavior
        color = self.colors.get(event.behavior_type, (255, 255, 255))

        # Background
        label_text = f"{behavior} ({confidence:.0%})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX,
            self.config.label_font_scale, 2
        )

        padding = self.config.label_padding
        bg_width = text_width + padding * 2
        bg_height = text_height + padding * 2

        # Position at top center
        x = (width - bg_width) // 2
        y = padding

        # Draw background
        if in_event:
            cv2.rectangle(frame, (x, y), (x + bg_width, y + bg_height),
                         color, -1)
            text_color = (0, 0, 0)  # Black text on colored background
        else:
            cv2.rectangle(frame, (x, y), (x + bg_width, y + bg_height),
                         (50, 50, 50), -1)
            text_color = (200, 200, 200)

        # Draw text
        cv2.putText(frame, label_text,
                   (x + padding, y + text_height + padding // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.label_font_scale,
                   text_color, 2)

    def _draw_confidence_bar(
        self,
        frame: np.ndarray,
        confidence: float,
        width: int,
        height: int
    ):
        """Draw confidence bar at bottom of frame."""
        bar_height = 20
        bar_width = width - 40
        bar_x = 20
        bar_y = height - bar_height - 20

        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        # Filled portion
        filled_width = int(bar_width * confidence)
        if confidence >= 0.7:
            fill_color = (0, 0, 255)  # Red - high risk
        elif confidence >= 0.5:
            fill_color = (0, 165, 255)  # Orange - medium
        else:
            fill_color = (0, 255, 0)  # Green - low

        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + filled_width, bar_y + bar_height),
                     fill_color, -1)

        # Label
        cv2.putText(frame, f"Confidence: {confidence:.0%}",
                   (bar_x + 5, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_timestamp(self, frame: np.ndarray, timestamp: float, height: int):
        """Draw timestamp at bottom left."""
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"

        cv2.putText(frame, time_str, (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_frame_number(self, frame: np.ndarray, frame_num: int, height: int):
        """Draw frame number at bottom left."""
        cv2.putText(frame, f"Frame: {frame_num}", (10, height - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def generate_clips_for_events(
        self,
        video_path: Path,
        events: List[BehaviorEvent],
        pose_results: List[PoseResult],
        max_clips: int = 10,
        filter_suspicious: bool = True
    ) -> List[AnnotatedClip]:
        """
        Generate annotated clips for multiple events.

        Args:
            video_path: Source video path
            events: List of behavior events
            pose_results: Pose estimation results
            max_clips: Maximum clips to generate
            filter_suspicious: Only generate for suspicious events

        Returns:
            List of AnnotatedClip objects
        """
        # Filter events
        if filter_suspicious:
            suspicious_types = {"pickup", "concealment", "bypass", "shoplifting"}
            events = [e for e in events if e.behavior_type in suspicious_types]

        # Sort by confidence and take top N
        events = sorted(events, key=lambda e: e.confidence, reverse=True)[:max_clips]

        clips = []
        for i, event in enumerate(events):
            clip = self.generate_clip(
                video_path,
                event,
                pose_results,
                clip_id=f"{i:03d}"
            )
            if clip:
                clips.append(clip)

        return clips

    def generate_screenshot(
        self,
        video_path: Path,
        timestamp: float,
        pose: Optional[PoseResult],
        event: Optional[BehaviorEvent],
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Generate a single annotated screenshot."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Annotate
        if event:
            in_event = event.start_time <= timestamp <= event.end_time
            annotated = self._annotate_frame(
                frame, pose, event, timestamp, frame_num, width, height
            )
        else:
            annotated = frame

        # Save
        if output_path is None:
            output_path = self.output_dir / f"screenshot_{timestamp:.2f}.jpg"

        cv2.imwrite(str(output_path), annotated)
        return output_path
