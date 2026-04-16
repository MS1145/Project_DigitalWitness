"""
intent_scorer.py - Computes a composite risk/intent score from detection signals.

Pure computation - no I/O, no model calls, no Streamlit.

The scoring approach is loosely inspired by multi-signal risk frameworks used
in retail loss prevention research. The idea is to combine YOLO peak confidence
(how sure was the detector) with a sustained detection factor (how long did it
last) rather than relying on a single number.

NumPy is used for the mean calculation across event confidences:
    Harris, C.R. et al. (2020). Array programming with NumPy.
    Nature, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2
"""
from __future__ import annotations
import numpy as np

from core.result_types import BehaviourEvent, IntentScore


class IntentScorer:
    """
    Stateless scorer. Given YOLO and BiLSTM signals, produces an IntentScore.

    Shoplifting path: score = peak_conf * (0.5 + 0.5 * sustained_factor)
    Normal path: score = 0.50 * concealment + 0.35 * bypass + 0.15 * duration

    The weights (0.50, 0.35, 0.15) were chosen based on how strongly each
    signal correlates with confirmed theft in the training dataset. Concealment
    behaviour is weighted highest because it's the most reliable signal.
    """

    # score -> severity label mapping
    # thresholds chosen to match the advisory system's 5-level colour coding
    _SEVERITY_BANDS = [
        (0.30,  "NONE"),
        (0.50,  "LOW"),
        (0.70,  "MEDIUM"),
        (0.85,  "HIGH"),
        (1.01,  "CRITICAL"),
    ]

    def score(self,
              overall_class:    str,
              yolo_peak:        float,
              shop_segs_count:  int,
              behaviour_events: list[BehaviourEvent],
              video_duration:   float) -> IntentScore:
        """
        Args:
            overall_class: "shoplifting" | "normal"
            yolo_peak: highest per-frame YOLO shoplifting confidence
            shop_segs_count: number of 30-frame segments with peak >= 50%
            behaviour_events: BiLSTM per-window predictions
            video_duration: total video duration in seconds

        Returns:
            IntentScore
        """
        # split events into shoplifting and looking-around groups
        direct_e = [e for e in behaviour_events if e.behavior_type == "shoplifting"]
        recon_e  = [e for e in behaviour_events if e.behavior_type == "Looking around"]
        susp_e   = direct_e + recon_e

        if overall_class == "shoplifting":
            # sustained factor: 5 segments of high confidence = fully sustained
            # dividing by 5 is a rough heuristic - it worked well on test clips
            _sustained = min(shop_segs_count / 5.0, 1.0)
            raw_score  = min(yolo_peak * (0.5 + 0.5 * _sustained), 1.0)
            s_conceal  = yolo_peak
            s_bypass   = 0.0
            s_dur      = _sustained
        else:
            # normal verdict - compute sub-scores from LSTM events only
            # if there are no shoplifting events from LSTM, concealment = 0
            s_conceal = (float(np.mean([e.confidence for e in direct_e]))
                         * min(1.0, len(direct_e) / 3.0)) if direct_e else 0.0
            s_bypass  = (float(np.mean([e.confidence for e in recon_e]))
                         * min(1.0, len(recon_e) / 3.0)) if recon_e else 0.0
            susp_secs = sum(e.end_time - e.start_time for e in susp_e)
            s_dur     = min(1.0, (susp_secs / max(1.0, video_duration)) * 3.0) if susp_e else 0.0
            raw_score = 0.50 * s_conceal + 0.35 * s_bypass + 0.15 * s_dur

        # clamp to [0, 1] just in case floating point arithmetic goes weird
        raw_score = max(0.0, min(1.0, raw_score))
        severity  = self._severity(raw_score)

        return IntentScore(
            score       = raw_score,
            severity    = severity,
            components  = {"concealment": s_conceal,
                           "bypass"     : s_bypass,
                           "duration"   : s_dur},
            explanation = (
                f"Score: {raw_score:.2f} ({severity}). "
                f"Concealment: {s_conceal:.2f}, Bypass: {s_bypass:.2f}, "
                f"Duration: {s_dur:.2f}"
            ),
        )

    def _severity(self, score: float) -> str:
        for threshold, name in self._SEVERITY_BANDS:
            if score < threshold:
                return name
        return "CRITICAL"
