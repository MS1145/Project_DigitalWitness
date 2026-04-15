"""
video_processor.py - OpenCV video I/O, YOLO detection/tracking, and frame annotation.

VideoFrameIterator handles all frame-level YOLO operations so the pipeline
can focus on feature extraction and classification.
"""
from __future__ import annotations
import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from core.config import PipelineParams, DEFAULT_PARAMS
from core.result_types import VideoMetadata, YoloSegment


@dataclass
class FrameDetections:
    """YOLO detection state for a single frame."""
    person_boxes:  dict[int, np.ndarray] = field(default_factory=dict)  # {track_id: xyxy}
    product_boxes: list[tuple[str, np.ndarray]] = field(default_factory=list)
    yolo_label:    str | None = None
    yolo_conf:     float = 0.0


# Colour map for bounding-box annotation
_BOX_COLOURS = {
    "shoplifting"    : (0, 0, 220),
    "Looking around" : (0, 140, 255),
    "Picking-Holding": (0, 200, 255),
    "normal"         : (0, 200, 0),
}

# COCO product classes treated as "product interactions"
_COCO_PRODUCT_CLASSES = frozenset({
    "bottle", "cup", "bowl", "banana", "apple", "sandwich",
    "backpack", "handbag", "book", "cell phone", "orange",
    "backpack-or-handbag", "carrying-item",
    "empty-basket", "filled-basket",
    "empty-shopping-bags", "filled-shopping-bags",
    "empty-trolly", "filled-trolly",
})


class VideoFrameIterator:
    """
    Iterates a video file frame-by-frame, runs YOLO detection+tracking,
    handles bounding-box annotation, and optionally writes an annotated MP4.

    Usage:
        iterator = VideoFrameIterator(video_path, yolo_detector, params)
        for frame, detections, frame_num in iterator:
            ...
        iterator.release()
        meta     = iterator.metadata
        segments = iterator.build_yolo_segments(fps)
    """

    BEHAVIOUR_4CLS = frozenset({"Looking around", "Picking-Holding", "normal", "shoplifting"})
    SUSPICIOUS_CLS = frozenset({"Looking around", "shoplifting"})
    PRODUCT_CLS    = frozenset({"Picking-Holding"})
    PRIORITY       = {"shoplifting": 3, "Looking around": 2, "Picking-Holding": 1, "normal": 0}

    def __init__(self, video_path: str, yolo_detector,
                 params: PipelineParams = DEFAULT_PARAMS,
                 visual_out_path: str | None = None) -> None:
        self._video_path  = video_path
        self._detector    = yolo_detector
        self._params      = params
        self._cap         = cv2.VideoCapture(video_path)
        self._writer      = None
        self._frame_labels: list[tuple[int, str, float]] = []   # (frame_num, label, conf)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self._fps             = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total           = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width           = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height          = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_yolo_result = None   # updated each time YOLO runs; readable by pipeline

        if visual_out_path:
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                visual_out_path, fourcc, self._fps, (self._width, self._height)
            )

    #  Iterator protocol ─

    def __iter__(self):
        last_yolo_result = None
        frame_num        = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            frame_num += 1

            # Run YOLO every yolo_step frames; reuse otherwise
            if frame_num % self._params.yolo_step == 1 or last_yolo_result is None:
                last_yolo_result       = self._detector.track(frame)
                self._last_yolo_result = last_yolo_result   # expose to pipeline

            detections = self._parse_detections(last_yolo_result, frame_num)
            yield frame, detections, frame_num

    def annotate_and_write(self, frame: np.ndarray,
                           detections: FrameDetections,
                           frame_num: int,
                           yolo_result) -> np.ndarray:
        """
        Draw bounding boxes on a copy of frame and optionally write to MP4.
        Returns the annotated frame (RGB-ready for st.image).
        """
        ann = frame.copy()
        if yolo_result is not None and yolo_result.boxes is not None:
            for box_t, cls_id_v, conf_v in zip(
                    yolo_result.boxes.xyxy.cpu().numpy(),
                    yolo_result.boxes.cls.cpu().numpy(),
                    yolo_result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box_t)
                cn_v = self._detector.names[int(cls_id_v)]
                col_v, lbl_v = self._box_style(cn_v, conf_v)
                cv2.rectangle(ann, (x1, y1), (x2, y2), col_v, 2)
                (tw, th), _ = cv2.getTextSize(lbl_v, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lbl_y = max(y1 - 4, th + 6)
                cv2.rectangle(ann, (x1, lbl_y - th - 6),
                              (x1 + tw + 8, lbl_y + 2), col_v, -1)
                cv2.putText(ann, lbl_v, (x1 + 4, lbl_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(ann, f"{frame_num}/{self._total}",
                    (8, ann.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        if self._writer:
            self._writer.write(ann)
        return ann

    #  Segment timeline builder 

    def build_yolo_segments(self, duration: float) -> list[YoloSegment]:
        """Aggregate per-frame YOLO labels into 30-frame timeline segments."""
        if not self._frame_labels:
            return []
        SEG = 30
        seg_map: dict[int, list] = defaultdict(list)
        for fn, lbl, conf in self._frame_labels:
            seg_map[fn // SEG].append((lbl, conf))
        segments = []
        for seg_idx in sorted(seg_map.keys()):
            entries  = seg_map[seg_idx]
            dominant = max(entries, key=lambda x: (self.PRIORITY.get(x[0], 0), x[1]))
            avg_conf      = float(np.mean([c for _, c in entries]))
            shop_confs    = [c for lbl, c in entries if lbl == "shoplifting"]
            max_shop_conf = max(shop_confs) if shop_confs else 0.0
            segments.append(YoloSegment(
                start_time     = seg_idx * SEG / self._fps,
                end_time       = min((seg_idx + 1) * SEG / self._fps, duration),
                dominant_class = dominant[0],
                confidence     = avg_conf,
                max_shop_conf  = max_shop_conf,
            ))
        return segments

    #  Resource management ─

    def release(self) -> None:
        if self._writer:
            self._writer.release()
        self._cap.release()

    #  Properties ─

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            filename    = Path(self._video_path).name,
            duration    = 0.0,   # caller fills in final duration after frame count
            fps         = self._fps,
            width       = self._width,
            height      = self._height,
            frame_count = self._total,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def frame_labels(self) -> list[tuple[int, str, float]]:
        return self._frame_labels

    #  Private helpers ─

    def _parse_detections(self, yolo_result, frame_num: int) -> FrameDetections:
        """Parse a raw YOLO result into a FrameDetections object."""
        det = FrameDetections()
        if yolo_result.boxes is None:
            return det

        real_ids = (yolo_result.boxes.id.int().tolist()
                    if yolo_result.boxes.id is not None else None)

        for i, (box_t, cls_id) in enumerate(zip(
                yolo_result.boxes.xyxy.cpu().numpy(),
                yolo_result.boxes.cls)):
            cls_name = self._detector.names[int(cls_id)]

            if self._detector.is_4cls:
                if cls_name not in self.BEHAVIOUR_4CLS:
                    continue
                tid = (real_ids[i] if real_ids is not None else -(i + 1))
                det.person_boxes[tid] = box_t
                if cls_name in self.PRODUCT_CLS:
                    det.product_boxes.append((cls_name, box_t))
            else:
                cls_id_int = int(cls_id)
                is_person  = (
                    cls_id_int == 0
                    or cls_name.lower() == "person"
                    or cls_name.lower().startswith("person-")
                    or (not self._detector.is_coco)
                )
                if is_person:
                    tid = (real_ids[i] if real_ids is not None else i + 1)
                    det.person_boxes[tid] = box_t
                elif cls_name in _COCO_PRODUCT_CLASSES:
                    det.product_boxes.append((cls_name, box_t))

        # Dominant YOLO class for this frame (4-class model only)
        if self._detector.is_4cls and yolo_result.boxes is not None:
            for _cls_v, _cof_v in zip(yolo_result.boxes.cls.cpu().numpy(),
                                       yolo_result.boxes.conf.cpu().numpy()):
                _cn  = self._detector.names[int(_cls_v)]
                _pri = self.PRIORITY.get(_cn, -1)
                _cur = self.PRIORITY.get(det.yolo_label, -1)
                if _pri > _cur or (_pri == _cur and float(_cof_v) > det.yolo_conf):
                    det.yolo_label = _cn
                    det.yolo_conf  = float(_cof_v)

        if det.yolo_label:
            self._frame_labels.append((frame_num, det.yolo_label, det.yolo_conf))
        return det

    def _box_style(self, cls_name: str,
                   conf: float) -> tuple[tuple[int, int, int], str]:
        """Return (BGR colour, label string) for a detection box."""
        if self._detector.is_4cls:
            if cls_name == "shoplifting" and conf < 0.50:
                return (0, 200, 0), f"normal {conf:.0%}"
            return _BOX_COLOURS.get(cls_name, (0, 200, 0)), f"{cls_name} {conf:.0%}"
        is_p  = cls_name.lower() in ("person",) or cls_name.lower().startswith("person-")
        col   = (0, 200, 0) if is_p else (160, 160, 160)
        label = f"Person {conf:.0%}" if is_p else f"{cls_name} {conf:.0%}"
        return col, label
