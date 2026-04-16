"""
clip_extractor.py - Extracts animated GIF clips from shoplifting windows.

OpenCV is used to read the video and decode frames:
    Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal.
    https://opencv.org

Pillow (PIL) is used to encode frames as animated GIFs:
    Clark, A. et al. (2015). Pillow (PIL Fork). GitHub.
    https://github.com/python-pillow/Pillow

Why GIFs and not MP4? Streamlit's st.image() can display animated GIFs
inline without needing a video player component. MP4 clips would require
st.video() which has a bigger footprint in the results page.
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

    Each clip covers the detected window plus/minus PAD_S seconds of context.
    Deduplication: windows whose midpoints are < MIN_GAP seconds apart are
    collapsed to the highest-confidence one, preventing near-duplicate clips.

    MAX_W limits the GIF width to keep file sizes reasonable. At 480px wide
    a 3-second clip at 8fps is roughly 1-2MB which is fine for the browser.
    """

    MAX_W:   int   = 480  # max pixel width for GIF frames
    GIF_FPS: int   = 8    # output frame rate - 8fps is smooth enough for review
    PAD_S:   float = 1.0  # seconds of padding around each detected window
    MIN_GAP: float = 2.0  # minimum seconds between clip midpoints (dedup)

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
        # filter to suspicious events only - ignore normal classifications
        suspicious = [e for e in events if self._behavior(e) in self._SUSPICIOUS]
        if not suspicious:
            return []

        # sort by confidence (highest first) then deduplicate by midpoint distance
        suspicious = sorted(suspicious, key=lambda e: self._conf(e), reverse=True)
        selected: list = []
        mids: list[float] = []
        for ev in suspicious:
            mid = (self._start(ev) + self._end(ev)) / 2
            # only keep this event if its midpoint is far enough from all already-selected ones
            if all(abs(mid - t) >= self.MIN_GAP for t in mids):
                selected.append(ev)
                mids.append(mid)
            if len(selected) >= max_clips:
                break

        if not selected:
            return []

        # open the video once and seek to each clip window
        # OpenCV - Bradski (2000)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_s = total_f / fps_v
        # step determines how many source frames we skip to hit the target GIF fps
        step  = max(1, int(fps_v / self.GIF_FPS))
        clips = []

        for ev in selected:
            ev_start   = self._start(ev)
            ev_end     = self._end(ev)
            clip_start = max(0.0,     ev_start - self.PAD_S)
            clip_end   = min(total_s, ev_end   + self.PAD_S)
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
                # only keep every Nth frame to match GIF_FPS
                if (src_f - start_f) % step == 0:
                    # OpenCV gives BGR, Pillow needs RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    if w > self.MAX_W:
                        s   = self.MAX_W / w
                        # INTER_AREA is best for downscaling - avoids aliasing
                        rgb = cv2.resize(rgb, (self.MAX_W, int(h * s)),
                                         interpolation=cv2.INTER_AREA)
                    pil_frames.append(Image.fromarray(rgb))
                    if thumbnail is None:
                        thumbnail = rgb
                src_f += 1

            if not pil_frames:
                continue

            # encode as animated GIF using Pillow
            # Pillow - Clark et al. (2015)
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

    # these accessors handle both BehaviourEvent dataclass and plain dict
    # (the dict format comes from the YOLO segment fallback path in app.py)
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


# numpy is used for the thumbnail type hint - imported here to avoid
# circular import issues (the class body uses np.ndarray in a type comment)
import numpy as np  # noqa: E402
