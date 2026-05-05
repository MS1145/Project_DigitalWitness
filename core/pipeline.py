"""
pipeline.py - AnalysisPipeline orchestrates the full inference sequence.

Sequence:
  1. Load YOLO via ModelRegistry
  2. Instantiate FeatureExtractor + BehaviourClassifier
  3. Iterate frames via VideoFrameIterator
     - Per frame: YOLO detect, MobileNet feature extract, rolling BiLSTM badge
  4. classify_sequence() - behaviour_events, lstm_verdict
  5. IntentScorer.score()
  6. BiasAssessor.assess() - bias_report, adjusted_score
  7. AlertGenerator.generate()
  8. Assemble PipelineResult

External libraries used:
  - ultralytics YOLO: Jocher et al. (2023) https://github.com/ultralytics/ultralytics
  - ByteTrack tracker: Zhang et al. (2022) https://arxiv.org/abs/2110.06864
  - PyTorch: Paszke et al. (2019) https://arxiv.org/abs/1912.01703
  - NumPy: Harris et al. (2020) https://doi.org/10.1038/s41586-020-2649-2
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Callable
from pathlib import Path

from core.config import PipelineParams, DEFAULT_PARAMS
from core.model_registry import ModelRegistry
from core.feature_extractor import FeatureExtractor
from core.behaviour_classifier import BehaviourClassifier
from core.video_processor import VideoFrameIterator
from core.intent_scorer import IntentScorer
from core.bias_assessor import BiasAssessor
from core.alert_generator import AlertGenerator
from core.result_types import (
    PipelineResult, VideoMetadata, LstmVerdict, DetectionSummary,
    YoloSignal, BehaviourEvent,
)

ProgressCallback  = Callable[[float, str], None]
LiveFrameCallback = Callable  # signature: (frame_rgb, frame_idx, total, stats) -> None


class YOLODetectorWrapper:
    """
    Thin wrapper around ultralytics.YOLO to give VideoFrameIterator a clean interface.

    YOLO source:
        Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO.
        GitHub. https://github.com/ultralytics/ultralytics
        License: AGPL-3.0

    Tracker:
        ByteTrack - Zhang, Y. et al. (2022).
        ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
        ECCV 2022. https://arxiv.org/abs/2110.06864

    We use ByteTrack over the default BoT-SORT because our clips are short
    and we don't need re-identification across long gaps. ByteTrack is faster
    and handles crowded retail scenes better for our use case.
    """

    BEHAVIOUR_4CLS = frozenset({"Looking around", "Picking-Holding", "normal", "shoplifting"})
    SUSPICIOUS_CLS = frozenset({"Looking around", "shoplifting"})
    PRODUCT_CLS    = frozenset({"Picking-Holding"})
    # higher number = more suspicious - used to pick the dominant label per frame
    PRIORITY = {"shoplifting": 3, "Looking around": 2, "Picking-Holding": 1, "normal": 0}

    def __init__(self, model_path: str, device: torch.device,
                 conf: float, iou: float) -> None:
        from ultralytics import YOLO
        self._model  = YOLO(model_path)
        self._device = 0 if str(device) == "cuda" else "cpu"
        self._half   = (str(device) == "cuda")  # FP16 only works on GPU
        self._conf   = conf
        self._iou    = iou
        _names_lower = {n.lower() for n in self._model.names.values()}
        # check what kind of model was loaded so we know how to parse detections
        self.is_4cls = ("Looking around" in self._model.names.values() or
                        "Picking-Holding" in self._model.names.values())
        self.is_coco = "person" in _names_lower
        self.names   = self._model.names
        self.mode    = ("4-class behaviour" if self.is_4cls else
                        ("COCO" if self.is_coco else "retail/unknown"))

    def track(self, frame, persist: bool = True):
        # persist=True tells ByteTrack to keep track IDs consistent across frames
        # imgsz=320 is a good balance between speed and accuracy for surveillance
        return self._model.track(
            frame, conf=self._conf, iou=self._iou,
            persist=persist, verbose=False,
            tracker="bytetrack.yaml",  # ByteTrack: Zhang et al. 2022
            imgsz=320, device=self._device, half=self._half,
        )[0]


class AnalysisPipeline:
    """
    End-to-end video analysis pipeline.
    Stateless - a new analysis is started fresh on each call to run().
    """

    def __init__(self, registry: ModelRegistry,
                 params: PipelineParams = DEFAULT_PARAMS) -> None:
        self._registry = registry
        self._params   = params
        self._scorer   = IntentScorer()
        self._assessor = BiasAssessor()
        self._alerter  = AlertGenerator()

    # public interface

    def run(self,
            video_path:          str,
            progress_callback:   ProgressCallback | None  = None,
            visual_out_path:     str | None               = None,
            live_frame_callback: LiveFrameCallback | None = None) -> PipelineResult:
        """
        Run the full pipeline on a video file.

        Returns a PipelineResult (suspicious_frames populated by the caller
        via ClipExtractor after this method returns, while temp file still exists).
        """
        def _cb(p: float, m: str) -> None:
            if progress_callback:
                progress_callback(p, m)

        _cb(0.0, "Initialising models...")
        device = self._device()
        if str(device) == "cuda":
            torch.backends.cudnn.benchmark = True  # speeds up fixed-size inference

        paths = self._registry.paths
        p     = self._params

        _cb(0.01, "Loading YOLO...")
        detector = YOLODetectorWrapper(
            self._registry.active_yolo_path(), device,
            conf=p.yolo_conf, iou=p.yolo_iou,
        )
        _cb(0.02, f"YOLO: {detector.mode} ({Path(self._registry.active_yolo_path()).name})")

        _cb(0.03, "Loading feature extractor...")
        extractor = FeatureExtractor(paths.mobilenet, device)

        _cb(0.04, "Loading behaviour classifier...")
        classifier = BehaviourClassifier(paths.bilstm, device, p)

        _cb(0.05, "Opening video...")
        try:
            iterator = VideoFrameIterator(video_path, detector, p, visual_out_path)
        except IOError as e:
            return PipelineResult(success=False, error=str(e))

        fps_val = iterator.fps
        total_f = iterator.total_frames

        # frame processing loop
        all_feats:             list[np.ndarray] = []
        persons_set:           set              = set()
        max_concurrent:        int              = 0
        product_pickup_events: set              = set()  # (track_id, class_name) pairs
        yolo_shop_confs:       list[float]      = []
        last_live_label:       str | None       = None
        last_live_conf:        float            = 0.0
        live_timeline:         list[str]        = []
        frame_num:             int              = 0
        _LIVE_STEP = 15  # send a frame to the UI every 15 frames (~2fps at 30fps source)

        for frame, detections, frame_num in iterator:
            if progress_callback and frame_num % 90 == 0:
                pct = min(0.05 + (frame_num / max(total_f, 1)) * 0.65, 0.70)
                _cb(pct, f"Processing frame {frame_num}/{total_f}...")

            persons_set.update(detections.person_boxes.keys())
            real_count = len(detections.person_boxes)
            if real_count > max_concurrent:
                max_concurrent = real_count

            # proximity-based product pickup detection
            # if a person's centre is within 1.2x their height of a product,
            # we count that as a pickup interaction
            for pid, pbox in detections.person_boxes.items():
                px1, py1, px2, py2 = pbox
                ph  = max(py2 - py1, 1)
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                for cls_name, prod_box in detections.product_boxes:
                    qx1, qy1, qx2, qy2 = prod_box
                    qcx = (qx1 + qx2) / 2
                    qcy = (qy1 + qy2) / 2
                    if ((pcx - qcx) ** 2 + (pcy - qcy) ** 2) ** 0.5 < ph * 1.2:
                        product_pickup_events.add((pid, cls_name))
            if detections.yolo_label == "shoplifting":
                yolo_shop_confs.append(detections.yolo_conf)

            # live badge: prefer a fresh high-confidence YOLO label, fall back to
            # the last rolling BiLSTM prediction when YOLO is not confident
            _SUSP = {"shoplifting", "Looking around"}
            if detections.yolo_label in _SUSP and detections.yolo_conf >= 0.50:
                badge_label = detections.yolo_label
                badge_conf  = detections.yolo_conf
            elif last_live_label:
                badge_label = last_live_label
                badge_conf  = last_live_conf
            else:
                badge_label = None
                badge_conf  = 0.0

            # frame annotation + live preview
            _need_live  = live_frame_callback and frame_num % _LIVE_STEP == 0
            _need_write = visual_out_path is not None

            if _need_write or _need_live:
                import cv2
                ann = iterator.annotate_and_write(
                    frame, detections, frame_num,
                    iterator._last_yolo_result,  # real YOLO result with actual boxes
                )
                if _need_live:
                    import cv2 as _cv2
                    _h, _w = ann.shape[:2]
                    if _w > 480:
                        # downscale before sending to browser - cuts websocket payload ~4x
                        _s  = 480 / _w
                        ann = _cv2.resize(ann, (480, int(_h * _s)),
                                          interpolation=_cv2.INTER_AREA)
                    ann_rgb  = _cv2.cvtColor(ann, _cv2.COLOR_BGR2RGB)
                    tl_color = ("red"   if badge_label in _SUSP else
                                "green" if badge_label == "normal" else "gray")
                    live_timeline.append(tl_color)
                    live_frame_callback(ann_rgb, frame_num, total_f, {
                        "persons_tracked"  : max_concurrent,
                        "products_detected": len({c for (_, c) in product_pickup_events}),
                        "label"            : badge_label or "Analyzing...",
                        "conf"             : badge_conf,
                        "timeline"         : list(live_timeline),
                    })

            # MobileNetV2 feature extraction (every feat_step frames for speed)
            if frame_num % p.feat_step == 0:
                feat = extractor.extract(frame)
                all_feats.append(feat)
                # rolling BiLSTM for live badge - run once every half-stride of features
                if (visual_out_path or live_frame_callback) and \
                        len(all_feats) >= p.lstm_seq_len and \
                        (len(all_feats) - p.lstm_seq_len) % max(p.lstm_stride // 2, 1) == 0:
                    last_live_label, last_live_conf = \
                        classifier.rolling_predict(all_feats)

        iterator.release()
        duration = frame_num / fps_val

        # early exit: video had no usable frames
        if not all_feats:
            _cb(1.0, "Video too short - returning normal.")
            return self._early_exit(
                f"Video too short ({frame_num} frame(s)) - insufficient data for "
                "temporal classification. Classified as normal.",
                frame_num, total_f, fps_val, iterator,
            )

        _cb(0.72, "Running LSTM temporal classification...")
        behaviour_events = classifier.classify_sequence(all_feats, duration, fps_val)

        _cb(0.85, "Classifying...")
        yolo_segments = iterator.build_yolo_segments(duration)
        yolo_peak     = max(yolo_shop_confs, default=0.0)
        yolo_mean     = (sum(yolo_shop_confs) / len(yolo_shop_confs)
                         if yolo_shop_confs else 0.0)

        # build a list of moments where YOLO detected shoplifting above review threshold
        yolo_shop_moments = sorted(
            [{"time": round(fn / fps_val, 2), "conf": round(conf, 3)}
             for fn, lbl, conf in iterator.frame_labels
             if lbl == "shoplifting" and conf >= p.review_threshold],
            key=lambda x: x["time"],
        )
        shop_segs = [s for s in yolo_segments if s.max_shop_conf >= p.review_threshold]

        # classification decision: peak YOLO confidence is the final arbiter
        # BiLSTM results are advisory only - see analysis_view.py for the rationale
        norm_events = [e for e in behaviour_events if e.behavior_type == "normal"]
        if yolo_peak >= p.shop_threshold:
            verdict = LstmVerdict(
                classification = "shoplifting",
                confidence     = min(yolo_peak, 0.99),
                is_shoplifting = True,
            )
        else:
            avg_norm = (float(np.mean([e.confidence for e in norm_events]))
                        if norm_events else 0.5)
            verdict  = LstmVerdict(
                classification = "normal",
                confidence     = min(avg_norm, 0.99),
                is_shoplifting = False,
            )

        _cb(0.90, "Scoring intent...")
        intent = self._scorer.score(
            verdict.classification, yolo_peak, len(shop_segs),
            behaviour_events, duration,
        )

        det_rate = min(1.0, len(all_feats) / max(1, total_f // p.feat_step))
        adj_score, bias, quality = self._assessor.assess(
            intent, self._registry.bilstm_exists(), det_rate
        )

        # rebuild intent with the bias-adjusted score
        from core.result_types import IntentScore
        adj_intent = IntentScore(
            score       = adj_score,
            severity    = intent.severity,
            components  = intent.components,
            explanation = intent.explanation,
        )

        direct_e = [e for e in behaviour_events if e.behavior_type == "shoplifting"]
        recon_e  = [e for e in behaviour_events if e.behavior_type == "Looking around"]
        alert    = self._alerter.generate(
            adj_score, intent.severity, len(direct_e) + len(recon_e)
        )

        # count how many unique (person, product_class) pairs we saw
        product_pickups: dict[str, int] = {}
        for _, cls_name in product_pickup_events:
            product_pickups[cls_name] = product_pickups.get(cls_name, 0) + 1

        # ── attention timeline ────────────────────────────────────────────────
        # Map each window's per-frame attention weights onto an absolute time axis
        # and average across overlapping windows.  This produces one attention
        # value per extracted feature frame, giving a continuous signal over
        # the full video that the UI can plot as the XAI explanation.
        _attn_buckets: dict[float, list[float]] = {}
        for evt in behaviour_events:
            if not evt.attention_weights:
                continue
            # feat index of the first frame in this window
            win_start_feat = round(evt.start_time * fps_val / p.feat_step)
            for i, w in enumerate(evt.attention_weights):
                t = round((win_start_feat + i) * p.feat_step / fps_val, 2)
                _attn_buckets.setdefault(t, []).append(float(w))
        lstm_attn_timeline = [
            {"time": t, "weight": float(np.mean(ws))}
            for t, ws in sorted(_attn_buckets.items())
        ]

        _cb(0.95, "Building case file...")
        detection = DetectionSummary(
            persons_tracked   = max_concurrent,
            products_detected = len({c for (_, c) in product_pickup_events}),
            interactions      = len(direct_e) + len(recon_e),
            frames_processed  = len(all_feats),
            product_pickups   = product_pickups,
        )
        yolo_sig = YoloSignal(
            frames_with_shoplifting = len(yolo_shop_confs),
            peak_conf               = round(yolo_peak, 4),
            mean_conf               = round(yolo_mean, 4),
            total_segments          = len(yolo_segments),
            shop_segs_above_50pct   = len(shop_segs),
        )
        vm = VideoMetadata(
            filename    = Path(video_path).name,
            duration    = duration,
            fps         = fps_val,
            width       = iterator._width,
            height      = iterator._height,
            frame_count = total_f,
        )
        _cb(1.0, "Complete!")
        return PipelineResult(
            success              = True,
            video_metadata       = vm,
            lstm_verdict         = verdict,
            detection            = detection,
            behaviour_events     = behaviour_events,
            yolo_segments        = yolo_segments,
            yolo_shop_moments    = yolo_shop_moments,
            yolo_signal          = yolo_sig,
            intent               = adj_intent,
            quality              = quality,
            bias                 = bias,
            alert                = alert,
            model_trained            = self._registry.bilstm_exists(),
            annotated_video_path     = visual_out_path,
            lstm_attention_timeline  = lstm_attn_timeline,
        )

    # private helpers

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _early_exit(self, reason: str, frame_num: int, total_f: int,
                    fps_val: float, iterator: VideoFrameIterator) -> PipelineResult:
        from core.result_types import IntentScore, QualityAnalysis, BiasReport
        return PipelineResult(
            success           = True,
            early_exit        = True,
            early_exit_reason = reason,
            video_metadata    = VideoMetadata(
                filename    = "",
                duration    = frame_num / max(fps_val, 1),
                fps         = fps_val,
                width       = iterator._width,
                height      = iterator._height,
                frame_count = total_f,
            ),
            lstm_verdict  = LstmVerdict("normal", 1.0, False),
            detection     = DetectionSummary(0, 0, 0, frame_num, {}),
            intent        = IntentScore(0.0, "NONE", {}, reason),
            quality       = QualityAnalysis(1.0, 1.0, True),
            bias          = BiasReport(1.0, True, False, ()),
            model_trained = self._registry.bilstm_exists(),
        )
