"""
Multi-object tracking module for Digital Witness.

Tracks detected objects across frames to maintain identity and
build interaction histories for behavior analysis.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .yolo_detector import Detection, InteractionEvent


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    track_id: int
    class_name: str
    class_id: int
    first_seen: float           # Timestamp when first detected
    last_seen: float            # Timestamp of most recent detection
    first_frame: int
    last_frame: int
    detection_count: int        # Number of frames detected
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)  # Estimated velocity (px/frame)

    @property
    def duration(self) -> float:
        """Duration object has been tracked."""
        return self.last_seen - self.first_seen

    @property
    def average_confidence(self) -> float:
        """Average detection confidence."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Most recent bounding box."""
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def last_center(self) -> Optional[Tuple[float, float]]:
        """Center of most recent bounding box."""
        if not self.bbox_history:
            return None
        x1, y1, x2, y2 = self.bbox_history[-1]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "detection_count": self.detection_count,
            "duration": self.duration,
            "average_confidence": self.average_confidence,
            "velocity": self.velocity
        }


@dataclass
class TrackingState:
    """Current state of the tracking system."""
    active_tracks: Dict[int, TrackedObject]
    completed_tracks: Dict[int, TrackedObject]
    interaction_history: List[InteractionEvent]
    frame_count: int
    current_timestamp: float


class ObjectTracker:
    """
    Multi-object tracker for maintaining object identities across frames.

    Builds on YOLO's tracking to provide:
    - Object history and trajectories
    - Interaction tracking between persons and products
    - Velocity estimation
    - Track lifecycle management
    """

    def __init__(
        self,
        max_track_age: int = 30,
        min_track_length: int = 3,
        velocity_smoothing: float = 0.3
    ):
        """
        Initialize object tracker.

        Args:
            max_track_age: Frames before inactive track is completed
            min_track_length: Minimum frames for valid track
            velocity_smoothing: Exponential smoothing factor for velocity
        """
        self.max_track_age = max_track_age
        self.min_track_length = min_track_length
        self.velocity_smoothing = velocity_smoothing

        # Tracking state
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.completed_tracks: Dict[int, TrackedObject] = {}
        self.interaction_history: List[InteractionEvent] = []

        # Frame tracking
        self.frame_count = 0
        self.current_timestamp = 0.0
        self.last_frame_tracks: Dict[int, Detection] = {}

        # ID management (for detections without track_id)
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        interactions: Optional[List[InteractionEvent]] = None
    ) -> List[TrackedObject]:
        """
        Update tracker with new frame detections.

        Args:
            detections: Detections from current frame
            timestamp: Current timestamp
            interactions: Optional interaction events to record

        Returns:
            List of currently active tracked objects
        """
        self.frame_count += 1
        self.current_timestamp = timestamp

        current_tracks: Dict[int, Detection] = {}

        for det in detections:
            track_id = det.track_id

            # Assign ID if not provided
            if track_id is None:
                track_id = self._assign_track_id(det)

            current_tracks[track_id] = det

            if track_id in self.active_tracks:
                # Update existing track
                self._update_track(self.active_tracks[track_id], det, timestamp)
            else:
                # Create new track
                self.active_tracks[track_id] = self._create_track(det, timestamp)

        # Handle tracks not seen in this frame
        tracks_to_complete = []
        for track_id in self.active_tracks:
            if track_id not in current_tracks:
                track = self.active_tracks[track_id]
                frames_since_seen = self.frame_count - track.last_frame

                if frames_since_seen > self.max_track_age:
                    tracks_to_complete.append(track_id)

        # Move old tracks to completed
        for track_id in tracks_to_complete:
            track = self.active_tracks.pop(track_id)
            if track.detection_count >= self.min_track_length:
                self.completed_tracks[track_id] = track

        # Record interactions
        if interactions:
            self.interaction_history.extend(interactions)

        self.last_frame_tracks = current_tracks

        return list(self.active_tracks.values())

    def _assign_track_id(self, detection: Detection) -> int:
        """Assign a track ID to a detection without one."""
        # Simple assignment - in production, would use Hungarian algorithm
        assigned_id = self._next_id
        self._next_id += 1
        return assigned_id

    def _create_track(self, detection: Detection, timestamp: float) -> TrackedObject:
        """Create a new tracked object."""
        track = TrackedObject(
            track_id=detection.track_id or self._next_id - 1,
            class_name=detection.class_name,
            class_id=detection.class_id,
            first_seen=timestamp,
            last_seen=timestamp,
            first_frame=self.frame_count,
            last_frame=self.frame_count,
            detection_count=1,
            bbox_history=[detection.bbox],
            confidence_history=[detection.confidence],
            velocity=(0.0, 0.0)
        )
        return track

    def _update_track(
        self,
        track: TrackedObject,
        detection: Detection,
        timestamp: float
    ):
        """Update an existing track with new detection."""
        # Calculate velocity
        if track.last_center:
            curr_center = detection.center
            dt = max(1, self.frame_count - track.last_frame)
            dx = (curr_center[0] - track.last_center[0]) / dt
            dy = (curr_center[1] - track.last_center[1]) / dt

            # Exponential smoothing
            alpha = self.velocity_smoothing
            track.velocity = (
                alpha * dx + (1 - alpha) * track.velocity[0],
                alpha * dy + (1 - alpha) * track.velocity[1]
            )

        # Update track
        track.last_seen = timestamp
        track.last_frame = self.frame_count
        track.detection_count += 1
        track.bbox_history.append(detection.bbox)
        track.confidence_history.append(detection.confidence)

        # Limit history length
        max_history = 100
        if len(track.bbox_history) > max_history:
            track.bbox_history = track.bbox_history[-max_history:]
            track.confidence_history = track.confidence_history[-max_history:]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get a tracked object by ID."""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        return self.completed_tracks.get(track_id)

    def get_persons(self) -> List[TrackedObject]:
        """Get all active person tracks."""
        return [t for t in self.active_tracks.values() if t.class_name == "person"]

    def get_products(self) -> List[TrackedObject]:
        """Get all active product tracks."""
        product_classes = {
            "bottle", "cup", "bowl", "banana", "apple", "sandwich",
            "backpack", "handbag", "book", "cell phone"
        }
        return [t for t in self.active_tracks.values() if t.class_name in product_classes]

    def get_person_interactions(self, person_id: int) -> List[InteractionEvent]:
        """Get all interactions for a specific person."""
        return [i for i in self.interaction_history if i.person_id == person_id]

    def get_product_interactions(self, product_id: int) -> List[InteractionEvent]:
        """Get all interactions involving a specific product."""
        return [i for i in self.interaction_history if i.product_id == product_id]

    def get_interaction_summary(self) -> Dict[str, int]:
        """Get summary of interaction types."""
        summary = defaultdict(int)
        for interaction in self.interaction_history:
            summary[interaction.interaction_type] += 1
        return dict(summary)

    def get_suspicious_tracks(
        self,
        min_interactions: int = 2,
        suspicious_types: Optional[List[str]] = None
    ) -> List[TrackedObject]:
        """
        Get person tracks with suspicious interaction patterns.

        Args:
            min_interactions: Minimum suspicious interactions
            suspicious_types: Types to consider suspicious

        Returns:
            List of suspicious person tracks
        """
        if suspicious_types is None:
            suspicious_types = ["conceal", "pickup"]

        suspicious = []

        for person in self.get_persons():
            interactions = self.get_person_interactions(person.track_id)
            suspicious_count = sum(
                1 for i in interactions
                if i.interaction_type in suspicious_types
            )

            if suspicious_count >= min_interactions:
                suspicious.append(person)

        return suspicious

    def get_state(self) -> TrackingState:
        """Get current tracking state."""
        return TrackingState(
            active_tracks=self.active_tracks.copy(),
            completed_tracks=self.completed_tracks.copy(),
            interaction_history=self.interaction_history.copy(),
            frame_count=self.frame_count,
            current_timestamp=self.current_timestamp
        )

    def reset(self):
        """Reset tracker for new video."""
        self.active_tracks.clear()
        self.completed_tracks.clear()
        self.interaction_history.clear()
        self.frame_count = 0
        self.current_timestamp = 0.0
        self.last_frame_tracks.clear()
        self._next_id = 1

    def get_tracking_summary(self) -> Dict:
        """Get summary of tracking results."""
        all_tracks = {**self.active_tracks, **self.completed_tracks}

        persons = [t for t in all_tracks.values() if t.class_name == "person"]
        products = [t for t in all_tracks.values() if t.class_name != "person"]

        return {
            "total_tracks": len(all_tracks),
            "active_tracks": len(self.active_tracks),
            "completed_tracks": len(self.completed_tracks),
            "persons_tracked": len(persons),
            "products_tracked": len(products),
            "total_interactions": len(self.interaction_history),
            "interaction_summary": self.get_interaction_summary(),
            "frames_processed": self.frame_count,
            "duration": self.current_timestamp
        }
