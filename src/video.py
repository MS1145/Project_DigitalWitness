"""
Video processing module for Digital Witness.
Handles video loading, metadata extraction, and forensic clip extraction.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List

from .config import CLIP_BUFFER_BEFORE, CLIP_BUFFER_AFTER, CLIP_OUTPUT_DIR


@dataclass
class VideoMetadata:
    """Video file properties extracted via OpenCV."""
    path: Path
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    codec: str


@dataclass
class ExtractedClip:
    """Metadata for an extracted forensic video clip."""
    path: Path
    start_time: float
    end_time: float
    event_time: float
    event_type: str


class VideoLoader:
    """OpenCV-based video reader with context manager support."""

    def __init__(self, video_path: str | Path):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def _load_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            duration = frame_count / fps if fps > 0 else 0.0
            return VideoMetadata(
                path=self.path, fps=fps, frame_count=frame_count,
                duration=duration, width=width, height=height, codec=codec
            )
        finally:
            cap.release()

    def open(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoLoader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Video not opened. Use context manager.")
        ret, frame = self._cap.read()
        return frame if ret else None

    def seek(self, frame_number: int) -> None:
        if self._cap is None:
            raise RuntimeError("Video not opened. Use context manager.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def frames(self, start: int = 0, end: Optional[int] = None,
               step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("Video not opened. Use context manager.")
        if end is None:
            end = self.metadata.frame_count
        self.seek(start)
        frame_num = start
        while frame_num < end:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame_num, frame
            if step > 1:
                frame_num += step
                self.seek(frame_num)
            else:
                frame_num += 1


class ClipExtractor:
    """Extracts evidence clips centered on suspicious events."""

    def __init__(self, video_loader: VideoLoader, output_dir: Optional[Path] = None,
                 buffer_before: float = CLIP_BUFFER_BEFORE, buffer_after: float = CLIP_BUFFER_AFTER):
        self.video_loader = video_loader
        self.output_dir = output_dir or CLIP_OUTPUT_DIR
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_clip(self, event_time: float, event_type: str,
                     clip_id: Optional[str] = None) -> ExtractedClip:
        metadata = self.video_loader.metadata
        start_time = max(0, event_time - self.buffer_before)
        end_time = min(metadata.duration, event_time + self.buffer_after)
        if clip_id is None:
            clip_id = f"{event_time:.2f}"
        filename = f"{event_type}_{clip_id}.mp4"
        output_path = self.output_dir / filename
        start_frame = int(start_time * metadata.fps)
        end_frame = int(end_time * metadata.fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, metadata.fps,
                                  (metadata.width, metadata.height))
        try:
            for frame_num, frame in self.video_loader.frames(start_frame, end_frame):
                writer.write(frame)
        finally:
            writer.release()
        return ExtractedClip(path=output_path, start_time=start_time,
                             end_time=end_time, event_time=event_time, event_type=event_type)

    def extract_clips_for_events(self, events: List[dict]) -> List[ExtractedClip]:
        clips = []
        for i, event in enumerate(events):
            clip = self.extract_clip(event['timestamp'], event['type'], str(i))
            clips.append(clip)
        return clips
