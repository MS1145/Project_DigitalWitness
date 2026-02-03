"""
Detection module for Digital Witness.

Provides YOLO-based object detection and multi-object tracking
for detecting persons, products, and interactions in retail environments.
"""
from .yolo_detector import YOLODetector, Detection, InteractionEvent
from .tracker import ObjectTracker, TrackedObject

__all__ = [
    'YOLODetector',
    'Detection',
    'InteractionEvent',
    'ObjectTracker',
    'TrackedObject'
]
