"""
YOLO-based object detection for Digital Witness.

Uses YOLOv8 to detect persons, products, hands, and other retail-relevant
objects in video frames. Detection results feed into the CNN-LSTM pipeline
for temporal behavior analysis.

YOLOv8 Overview:
----------------
- "You Only Look Once" - single-pass detection (fast!)
- YOLOv8n = "nano" variant, optimized for speed over accuracy
- Pretrained on COCO dataset (80 object classes)
- Built-in tracking via ByteTrack algorithm

Detection Pipeline:
-------------------
Frame → YOLOv8 → Bounding Boxes + Class Labels + Confidence Scores
                          ↓
              Tracking (ByteTrack maintains IDs across frames)
                          ↓
              Interaction Detection (spatial proximity analysis)

Interaction Types:
------------------
- "approach": Person within 150px of product
- "pickup": Product bbox overlaps person bbox 30-80%
- "hold": Product bbox mostly inside person bbox (>80%)
- "conceal": Product in lower torso area of person bbox
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from ..config import (
    YOLO_MODEL_PATH,       # Path to YOLOv8 weights
    YOLO_CONF_THRESHOLD,   # 0.5 - minimum detection confidence
    YOLO_CLASSES,          # Retail-relevant class names
    YOLO_IOU_THRESHOLD     # 0.5 - for non-max suppression
)


@dataclass
class Detection:
    """A single object detection result."""
    class_name: str                         # "person", "bottle", "bag", etc.
    class_id: int                           # COCO class ID
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2) bounding box
    confidence: float                       # Detection confidence [0, 1]
    track_id: Optional[int] = None          # ID for tracking across frames
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp
        }


@dataclass
class InteractionEvent:
    """A detected person-product interaction."""
    timestamp: float
    frame_number: int
    person_id: int                          # Track ID of person
    product_id: Optional[int]               # Track ID of product (if tracked)
    product_class: str                      # Product class name
    interaction_type: str                   # "approach", "pickup", "hold", "putback", "conceal"
    confidence: float                       # Interaction confidence [0, 1]
    person_bbox: Tuple[int, int, int, int]
    product_bbox: Optional[Tuple[int, int, int, int]]
    distance: float                         # Distance between person and product
    overlap_ratio: float                    # How much bboxes overlap

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "person_id": self.person_id,
            "product_id": self.product_id,
            "product_class": self.product_class,
            "interaction_type": self.interaction_type,
            "confidence": self.confidence,
            "person_bbox": self.person_bbox,
            "product_bbox": self.product_bbox,
            "distance": self.distance,
            "overlap_ratio": self.overlap_ratio
        }


class YOLODetector:
    """
    YOLOv8-based object detector for retail surveillance.

    Detects persons, products, and other relevant objects to enable
    behavior analysis through object interactions.
    """

    # COCO class IDs for retail-relevant objects
    RETAIL_CLASSES = {
        0: "person",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        60: "dining table",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush"
    }

    # Product categories for interaction detection
    PRODUCT_CLASSES = {
        "bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange",
        "donut", "cake", "book", "cell phone", "backpack", "handbag"
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        device: str = "auto"
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights (uses pretrained if None)
            conf_threshold: Minimum detection confidence
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ("auto", "cpu", "cuda:0")
        """
        self.model_path = model_path or str(YOLO_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of YOLO model."""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO

            # Check if model file exists, otherwise download
            model_path = Path(self.model_path)
            if not model_path.exists():
                # Use pretrained model name directly (will auto-download)
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(str(model_path))

            self._initialized = True

        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Detect objects in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            frame_number: Frame number for tracking
            timestamp: Timestamp in seconds
            classes: Optional list of class IDs to detect (None = all)

        Returns:
            List of Detection objects
        """
        self.initialize()

        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].tolist()))

            # Get class name
            class_name = self.RETAIL_CLASSES.get(
                class_id,
                results.names.get(class_id, f"class_{class_id}")
            )

            detection = Detection(
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                confidence=confidence,
                track_id=None,
                frame_number=frame_number,
                timestamp=timestamp
            )
            detections.append(detection)

        return detections

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> List[Detection]:
        """
        Detect and track objects across frames.

        Uses YOLO's built-in tracking (ByteTrack) to maintain object IDs.

        Args:
            frame: BGR image as numpy array
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            List of Detection objects with track_id populated
        """
        self.initialize()

        # Run tracking
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].tolist()))

            # Get track ID if available
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            class_name = self.RETAIL_CLASSES.get(
                class_id,
                results.names.get(class_id, f"class_{class_id}")
            )

            detection = Detection(
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                confidence=confidence,
                track_id=track_id,
                frame_number=frame_number,
                timestamp=timestamp
            )
            detections.append(detection)

        return detections

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> List[Detection]:
        """Detect only persons in frame."""
        return self.detect(frame, frame_number, timestamp, classes=[0])

    def detect_products(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> List[Detection]:
        """Detect retail products in frame."""
        # Get class IDs for products
        product_ids = [
            cid for cid, name in self.RETAIL_CLASSES.items()
            if name in self.PRODUCT_CLASSES
        ]
        return self.detect(frame, frame_number, timestamp, classes=product_ids)

    def detect_interactions(
        self,
        person_detections: List[Detection],
        product_detections: List[Detection],
        frame_number: int = 0,
        timestamp: float = 0.0,
        proximity_threshold: float = 100.0
    ) -> List[InteractionEvent]:
        """
        Detect person-product interactions based on spatial proximity.

        Args:
            person_detections: List of person detections
            product_detections: List of product detections
            frame_number: Current frame number
            timestamp: Current timestamp
            proximity_threshold: Maximum pixel distance for interaction

        Returns:
            List of InteractionEvent objects
        """
        interactions = []

        for person in person_detections:
            if person.class_name != "person":
                continue

            for product in product_detections:
                if product.class_name == "person":
                    continue

                # Calculate distance between centers
                person_center = person.center
                product_center = product.center
                distance = np.sqrt(
                    (person_center[0] - product_center[0])**2 +
                    (person_center[1] - product_center[1])**2
                )

                # Calculate bounding box overlap
                overlap = self._calculate_overlap(person.bbox, product.bbox)

                # Determine interaction type based on spatial relationship
                interaction_type = self._classify_interaction(
                    person, product, distance, overlap
                )

                if interaction_type and (distance < proximity_threshold or overlap > 0):
                    # Calculate interaction confidence
                    conf = self._calculate_interaction_confidence(
                        person, product, distance, overlap
                    )

                    interaction = InteractionEvent(
                        timestamp=timestamp,
                        frame_number=frame_number,
                        person_id=person.track_id or -1,
                        product_id=product.track_id,
                        product_class=product.class_name,
                        interaction_type=interaction_type,
                        confidence=conf,
                        person_bbox=person.bbox,
                        product_bbox=product.bbox,
                        distance=distance,
                        overlap_ratio=overlap
                    )
                    interactions.append(interaction)

        return interactions

    def _calculate_overlap(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU-like overlap ratio between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        return intersection / area2 if area2 > 0 else 0.0

    def _classify_interaction(
        self,
        person: Detection,
        product: Detection,
        distance: float,
        overlap: float
    ) -> Optional[str]:
        """
        Classify the type of interaction based on spatial features.

        Classification Logic (in order of priority):
        --------------------------------------------
        1. overlap > 0.8: Product almost entirely within person bbox
           - If in upper half: "hold" (carrying openly)
           - If in lower half: "conceal" (hiding in clothing/bag)

        2. overlap > 0.3: Significant overlap = active interaction
           - Classified as "pickup" (reaching for or grabbing item)

        3. distance < 150px: Close but not touching
           - Classified as "approach" (moving toward product)

        4. None: No significant spatial relationship

        Note: These thresholds are heuristics. In production, they should
        be tuned based on camera angles, resolution, and store layout.
        """
        # Product is almost entirely inside person bbox
        if overlap > 0.8:
            # Determine if holding openly (upper body) or concealing (lower body)
            person_center_y = (person.bbox[1] + person.bbox[3]) / 2
            product_center_y = product.center[1]

            # Product in upper half of person = likely holding openly
            # Product in lower half = likely in pocket, bag, or waistband
            if product_center_y < person_center_y:
                return "hold"
            else:
                return "conceal"

        # Significant overlap indicates active manipulation
        elif overlap > 0.3:
            return "pickup"

        # Close proximity without contact = approaching
        elif distance < 150:
            return "approach"

        return None

    def _calculate_interaction_confidence(
        self,
        person: Detection,
        product: Detection,
        distance: float,
        overlap: float
    ) -> float:
        """Calculate confidence score for an interaction."""
        # Base confidence from detection confidences
        base_conf = (person.confidence + product.confidence) / 2

        # Adjust based on overlap (higher overlap = higher confidence)
        if overlap > 0.5:
            overlap_factor = 1.0
        elif overlap > 0.2:
            overlap_factor = 0.8
        else:
            overlap_factor = 0.6

        # Adjust based on distance (closer = higher confidence)
        if distance < 50:
            distance_factor = 1.0
        elif distance < 100:
            distance_factor = 0.8
        else:
            distance_factor = 0.6

        return base_conf * overlap_factor * distance_factor

    def reset_tracking(self):
        """Reset tracking state for new video."""
        if self.model is not None:
            # Clear predictor to reset ByteTrack state
            self.model.predictor = None
            # Also reset tracker state if it exists
            if hasattr(self.model, 'trackers'):
                self.model.trackers = {}
            # Force re-initialization on next track call
            if hasattr(self.model, 'tracker'):
                self.model.tracker = None

    def close(self):
        """Release resources."""
        self.model = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
