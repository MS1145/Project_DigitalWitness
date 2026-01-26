"""
Video loading utilities for Digital Witness.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    path: Path
    fps: float
    frame_count: int
    duration: float  # seconds
    width: int
    height: int
    codec: str


class VideoLoader:
    """Loads and iterates through video frames."""

    def __init__(self, video_path: str | Path):
        """
        Initialize video loader.

        Args:
            video_path: Path to the video file
        """
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata, loading it if necessary."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def _load_metadata(self) -> VideoMetadata:
        """Load metadata from video file."""
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
                path=self.path,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                width=width,
                height=height,
                codec=codec
            )
        finally:
            cap.release()

    def open(self) -> None:
        """Open video capture."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

    def close(self) -> None:
        """Close video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoLoader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame.

        Returns:
            Frame as numpy array (BGR) or None if no more frames
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() or use context manager.")

        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def seek(self, frame_number: int) -> None:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to (0-indexed)
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() or use context manager.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def seek_time(self, seconds: float) -> None:
        """
        Seek to a specific time.

        Args:
            seconds: Time in seconds from start
        """
        frame_number = int(seconds * self.metadata.fps)
        self.seek(frame_number)

    def get_current_frame_number(self) -> int:
        """Get current frame position."""
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() or use context manager.")
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_current_time(self) -> float:
        """Get current time position in seconds."""
        return self.get_current_frame_number() / self.metadata.fps

    def frames(self, start: int = 0, end: Optional[int] = None,
               step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate through frames.

        Args:
            start: Starting frame number
            end: Ending frame number (exclusive), None for all frames
            step: Frame step (1 = every frame, 2 = every other frame, etc.)

        Yields:
            Tuple of (frame_number, frame_data)
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() or use context manager.")

        if end is None:
            end = self.metadata.frame_count

        self.seek(start)
        frame_num = start

        while frame_num < end:
            frame = self.read_frame()
            if frame is None:
                break

            yield frame_num, frame

            # Skip frames if step > 1
            if step > 1:
                frame_num += step
                self.seek(frame_num)
            else:
                frame_num += 1

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame number to retrieve

        Returns:
            Frame as numpy array or None if invalid
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() or use context manager.")

        self.seek(frame_number)
        return self.read_frame()

    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at a specific time.

        Args:
            seconds: Time in seconds

        Returns:
            Frame as numpy array or None if invalid
        """
        frame_number = int(seconds * self.metadata.fps)
        return self.get_frame_at(frame_number)
