"""
Behavior event data structure for Digital Witness.

This module defines the BehaviorEvent dataclass used throughout the pipeline
to represent classified behavior segments with temporal bounds.

The BehaviorEvent is the output format from the LSTM classifier, storing
both the prediction and confidence information for each time window.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class BehaviorEvent:
    """
    A classified behavior segment with temporal bounds.

    Used to represent output from the LSTM classifier, storing both
    the prediction and confidence information for a time window.

    Attributes:
        behavior_type: Predicted class ("normal", "pickup", "concealment", "bypass")
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        start_frame: Starting frame number
        end_frame: Ending frame number
        confidence: Prediction confidence (max probability)
        probabilities: Per-class probability distribution for explainability

    Example:
        BehaviorEvent(
            behavior_type="concealment",
            start_time=27.0,
            end_time=30.0,
            start_frame=810,
            end_frame=900,
            confidence=0.85,
            probabilities={"normal": 0.10, "pickup": 0.03, "concealment": 0.85, "bypass": 0.02}
        )
    """
    behavior_type: str              # Predicted class label
    start_time: float               # Segment start (seconds)
    end_time: float                 # Segment end (seconds)
    start_frame: int                # Starting frame number
    end_frame: int                  # Ending frame number
    confidence: float               # Max probability (prediction confidence)
    probabilities: Dict[str, float] # Per-class probabilities for explainability
