"""
clip_extractor.py - Extracts animated GIF clips from shoplifting windows.
"""
from __future__ import annotations
import cv2
import io
from pathlib import Path
from PIL import Image

from core.result_types import BehaviourEvent


class ClipExtractor:
    """
    Extracts animated GIF clips from the highest-confidence shoplifting
    windows in a video.

    Each clip covers the detected window ± PAD_S seconds of context.
    Deduplication: windows whose midpoints are < MIN_GAP seconds apart are
    collapsed to the highest-confidence one, preventing near-duplicate clips.
    """

    MAX_W:   int   = 480 # max pixel width for GIF frames
    GIF_FPS: int   = 8   # output frame rate
    PAD_S:   float = 1.0 # seconds of padding around each window
    MIN_GAP: float = 2.0 # minimum seconds between clip midpoints

    _SUSPICIOUS = frozenset({"shoplifting", "Looking around", "concealment", "bypass"})

    def extract(self,
                video_path: str,
                events: list[BehaviourEvent | dict],
                max_clips: int = 4) -> list[dict]:
        """
        Args:
            video_path : path to the source video
            events     : BehaviourEvent objects or legacy dicts with start_time/end_time
            max_clips  : maximum number of clips to return

        Returns:
            List of dicts with keys:
                gif_bytes, thumbnail, timestamp, frame_start, frame_end,
                clip_start, clip_end, behavior, confidence
        """
        suspicious = [e for e in events if self._behavior(e) in self._SUSPICIOUS]
        if not suspicious:
            return []

        suspicious = sorted(suspicious, key=lambda e: self._conf(e), reverse=True)
        selected: list = []
        mids:     list[float] = []
        for ev in suspicious:
            mid = (self._start(ev) + self._end(ev)) / 2
            if all(abs(mid - t) >= self.MIN_GAP for t in mids):
                selected.append(ev)
                mids.append(mid)
            if len(selected) >= max_clips:
                break

        if not selected:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_s = total_f / fps_v
        step    = max(1, int(fps_v / self.GIF_FPS))
        clips   = []

        for ev in selected:
            ev_start   = self._start(ev)
            ev_end     = self._end(ev)
            clip_start = max(0.0,    ev_start - self.PAD_S)
            clip_end   = min(total_s, ev_end  + self.PAD_S)
            start_f    = int(clip_start * fps_v)
            end_f      = int(clip_end   * fps_v)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            pil_frames: list[Image.Image] = []
            thumbnail: np.ndarray | None  = None
            src_f = start_f

            while src_f <= end_f:
                ret, frame = cap.read()
                if not ret:
                    break
                if (src_f - start_f) % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    if w > self.MAX_W:
                        s   = self.MAX_W / w
                        rgb = cv2.resize(rgb, (self.MAX_W, int(h * s)),
                                         interpolation=cv2.INTER_AREA)
                    pil_frames.append(Image.fromarray(rgb))
                    if thumbnail is None:
                        thumbnail = rgb
                src_f += 1

            if not pil_frames:
                continue

            buf = io.BytesIO()
            pil_frames[0].save(
                buf, format="GIF", save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / self.GIF_FPS),
                loop=0, optimize=True,
            )
            clips.append({
                "gif_bytes"  : buf.getvalue(),
                "thumbnail"  : thumbnail,
                "timestamp"  : (ev_start + ev_end) / 2,
                "frame_start": int(ev_start * fps_v),
                "frame_end"  : int(ev_end   * fps_v),
                "clip_start" : clip_start,
                "clip_end"   : clip_end,
                "behavior"   : self._behavior(ev),
                "confidence" : self._conf(ev),
            })

        cap.release()
        return clips

    #  Accessors that work for both BehaviourEvent and legacy dict ─
    @staticmethod
    def _behavior(e) -> str:
        return e.behavior_type if isinstance(e, BehaviourEvent) else e.get("behavior_type", "")

    @staticmethod
    def _start(e) -> float:
        return e.start_time if isinstance(e, BehaviourEvent) else e.get("start_time", 0.0)

    @staticmethod
    def _end(e) -> float:
        return e.end_time if isinstance(e, BehaviourEvent) else e.get("end_time", 0.0)

    @staticmethod
    def _conf(e) -> float:
        return e.confidence if isinstance(e, BehaviourEvent) else e.get("confidence", 0.0)


# Avoid NameError for type hint in _extract
import numpy as np  # noqa: E402 (placed after class body intentionally)
