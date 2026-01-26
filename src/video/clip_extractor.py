"""
Forensic clip extraction for Digital Witness.
"""
import cv2
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .loader import VideoLoader
from ..config import CLIP_BUFFER_BEFORE, CLIP_BUFFER_AFTER, CLIP_OUTPUT_DIR


@dataclass
class ExtractedClip:
    """Information about an extracted video clip."""
    path: Path
    start_time: float
    end_time: float
    event_time: float
    event_type: str


class ClipExtractor:
    """Extracts short video clips around detected events."""

    def __init__(
        self,
        video_loader: VideoLoader,
        output_dir: Optional[Path] = None,
        buffer_before: float = CLIP_BUFFER_BEFORE,
        buffer_after: float = CLIP_BUFFER_AFTER
    ):
        """
        Initialize clip extractor.

        Args:
            video_loader: VideoLoader instance for the source video
            output_dir: Directory to save extracted clips
            buffer_before: Seconds of video to include before event
            buffer_after: Seconds of video to include after event
        """
        self.video_loader = video_loader
        self.output_dir = output_dir or CLIP_OUTPUT_DIR
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_clip(
        self,
        event_time: float,
        event_type: str,
        clip_id: Optional[str] = None
    ) -> ExtractedClip:
        """
        Extract a clip around an event.

        Args:
            event_time: Time of the event in seconds
            event_type: Type of event (for naming)
            clip_id: Optional unique identifier for the clip

        Returns:
            ExtractedClip with path and timing info
        """
        metadata = self.video_loader.metadata

        # Calculate start and end times
        start_time = max(0, event_time - self.buffer_before)
        end_time = min(metadata.duration, event_time + self.buffer_after)

        # Generate clip filename
        if clip_id is None:
            clip_id = f"{event_time:.2f}"
        filename = f"{event_type}_{clip_id}.mp4"
        output_path = self.output_dir / filename

        # Calculate frame numbers
        start_frame = int(start_time * metadata.fps)
        end_frame = int(end_time * metadata.fps)

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height)
        )

        try:
            # Extract frames
            for frame_num, frame in self.video_loader.frames(start_frame, end_frame):
                writer.write(frame)
        finally:
            writer.release()

        return ExtractedClip(
            path=output_path,
            start_time=start_time,
            end_time=end_time,
            event_time=event_time,
            event_type=event_type
        )

    def extract_clips_for_events(
        self,
        events: List[dict]
    ) -> List[ExtractedClip]:
        """
        Extract clips for multiple events.

        Args:
            events: List of event dicts with 'timestamp' and 'type' keys

        Returns:
            List of ExtractedClip objects
        """
        clips = []
        for i, event in enumerate(events):
            clip = self.extract_clip(
                event_time=event['timestamp'],
                event_type=event['type'],
                clip_id=str(i)
            )
            clips.append(clip)
        return clips
