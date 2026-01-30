"""
Pose estimation module for Digital Witness.

Used for quality analysis (pose detection rate, visibility metrics).
The main behavior classification is done by the LSTM in src/models/.
"""
from .estimator import PoseEstimator, PoseResult
from .behavior_classifier import BehaviorEvent

__all__ = [
    'PoseEstimator',
    'PoseResult',
    'BehaviorEvent'
]
