"""
result_types.py - Typed value objects that cross the pipeline -> UI boundary.

All dataclasses are immutable (frozen=True) except PipelineResult and
BehaviourEvent which need to be mutable after construction.

Why dataclasses instead of dicts? Using typed objects means the IDE can
catch typos and missing fields at write-time rather than crashing at runtime.
It also makes the pipeline output self-documenting - you can read this file
and know exactly what the UI receives.

Python dataclasses were introduced in PEP 557:
    Van Rossum, G. et al. PEP 557 - Data Classes.
    https://peps.python.org/pep-0557/
    Python 3.7+, Python Software Foundation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    filename:    str
    duration:    float
    fps:         float
    width:       int
    height:      int
    frame_count: int


@dataclass(frozen=True)
class LstmVerdict:
    """Final classification decision produced by the YOLO peak-threshold rule.

    Despite the name 'lstm_verdict' this is actually determined by YOLO peak
    confidence now. The name is kept for backwards compatibility with the UI.
    """
    classification: str    # "normal" | "shoplifting"
    confidence:     float
    is_shoplifting: bool


@dataclass
class BehaviourEvent:
    """One BiLSTM sliding-window prediction.

    Not frozen because the clip extractor sometimes mutates events when
    building the fallback from YOLO segments.

    attention_weights: one scalar per input frame in this window (seq_len values).
    Higher weight = BiLSTM considered that frame more important. These are the
    raw softmax outputs of the TemporalAttention module and sum to 1.0 across
    the window. Primary XAI signal for temporal explainability.
    """
    behavior_type:    str
    start_time:       float
    end_time:         float
    confidence:       float
    probabilities:    dict[str, float]
    attention_weights: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class YoloSegment:
    """Dominant YOLO class for a 30-frame segment.

    max_shop_conf is tracked separately from confidence because confidence
    is the average across all frames in the segment, while max_shop_conf
    is the peak - useful for the clip extraction fallback threshold.
    """
    start_time:     float
    end_time:       float
    dominant_class: str
    confidence:     float   # average across the segment
    max_shop_conf:  float   # peak shoplifting confidence in this segment


@dataclass(frozen=True)
class YoloSignal:
    """Aggregated YOLO shoplifting signal across the full video."""
    frames_with_shoplifting: int
    peak_conf:               float
    mean_conf:               float
    total_segments:          int
    shop_segs_above_50pct:   int


@dataclass(frozen=True)
class DetectionSummary:
    persons_tracked:   int
    products_detected: int
    interactions:      int
    frames_processed:  int
    product_pickups:   dict[str, int]   # {class_name: n_persons_who_handled_it}


@dataclass(frozen=True)
class IntentScore:
    score:       float
    severity:    str                    # NONE | LOW | MEDIUM | HIGH | CRITICAL
    components:  dict[str, float]       # concealment, bypass, duration
    explanation: str


@dataclass(frozen=True)
class QualityAnalysis:
    reliability_score: float
    detection_rate:    float
    usable:            bool


@dataclass(frozen=True)
class BiasReport:
    overall_fairness_score: float
    analysis_reliable:      bool
    requires_review:        bool
    flags:                  tuple[str, ...]


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str
    level:    str
    message:  str


@dataclass
class PipelineResult:
    """
    Complete output of AnalysisPipeline.run().

    Not frozen because suspicious_frames is populated post-pipeline by
    ClipExtractor in app.py after the temp video file is still available.
    """
    success:              bool
    error:                str | None              = None
    early_exit:           bool                   = False
    early_exit_reason:    str | None              = None
    video_metadata:       VideoMetadata | None    = None
    lstm_verdict:         LstmVerdict | None      = None
    detection:            DetectionSummary | None = None
    behaviour_events:     list[BehaviourEvent]    = field(default_factory=list)
    yolo_segments:        list[YoloSegment]       = field(default_factory=list)
    yolo_shop_moments:    list[dict[str, Any]]    = field(default_factory=list)
    yolo_signal:          YoloSignal | None       = None
    intent:               IntentScore | None      = None
    quality:              QualityAnalysis | None  = None
    bias:                 BiasReport | None       = None
    alert:                AlertRecord | None      = None
    model_trained:        bool                   = False
    annotated_video_path: str | None              = None
    suspicious_frames:    list[dict[str, Any]]   = field(default_factory=list)
    # Aggregated attention timeline: list of {"time": float, "weight": float}
    # Built by averaging overlapping window weights onto a per-frame time axis.
    # Used by the XAI visualisation to show which moments the BiLSTM focused on.
    lstm_attention_timeline: list[dict[str, Any]] = field(default_factory=list)
