"""
behaviour_classifier.py - Loads BiLSTMAttentionClassifier and runs sliding-window
                           inference over a sequence of CNN feature vectors.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

from core.model_definitions import BiLSTMAttentionClassifier
from core.config import PipelineParams, DEFAULT_PARAMS
from core.result_types import BehaviourEvent, LstmVerdict


class BehaviourClassifier:
    """
    Wraps BiLSTMAttentionClassifier with:
        checkpoint loading
        sliding-window inference over the full feature array
        short-video padding
        rolling single-window inference for the live badge
    """

    TEMPERATURE = 3.0   # flatten overconfident softmax distributions

    def __init__(self, checkpoint_path: Path,
                 device: torch.device,
                 params: PipelineParams = DEFAULT_PARAMS) -> None:
        self._device = device
        self._params = params
        self._model  = BiLSTMAttentionClassifier(
            num_classes=len(params.behavior_classes),
            hidden=256, layers=2, dropout=0.3,
        ).to(device)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            sd   = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(sd)
        self._model.eval()

    #  Full-video inference 
    def classify_sequence(self,
                          features: list[np.ndarray],
                          video_duration: float,
                          fps: float) -> list[BehaviourEvent]:
        """
        Sliding-window BiLSTM inference over all extracted features.

        Args:
            features       : list of (1280,) numpy arrays, one per feat_step frame
            video_duration : total video length in seconds
            fps            : source video fps

        Returns:
            list of BehaviourEvent (one per window)
        """
        p         = self._params
        feats_arr = np.array(features) if features else np.zeros((1, p.mobilenet_dim))
        events    = []

        if len(features) < p.lstm_seq_len:
            # Pad short video and run single inference
            pad    = np.zeros((p.lstm_seq_len - len(features), p.mobilenet_dim))
            seq    = np.vstack([feats_arr, pad])
            probs  = self._infer(seq)
            scale  = min(1.0, len(features) / p.lstm_seq_len)
            events.append(BehaviourEvent(
                behavior_type = p.behavior_classes[int(np.argmax(probs))],
                start_time    = 0.0,
                end_time      = video_duration,
                confidence    = min(float(np.max(probs)) * scale, 0.99),
                probabilities = {c: float(prob) for c, prob in
                                 zip(p.behavior_classes, probs)},
            ))
        else:
            for start in range(0,
                               len(features) - p.lstm_seq_len + 1,
                               p.lstm_stride):
                seq   = feats_arr[start:start + p.lstm_seq_len]
                probs = self._infer(seq)
                events.append(BehaviourEvent(
                    behavior_type = p.behavior_classes[int(np.argmax(probs))],
                    start_time    = start * p.feat_step / fps,
                    end_time      = (start + p.lstm_seq_len) * p.feat_step / fps,
                    confidence    = min(float(np.max(probs)), 0.99),
                    probabilities = {c: float(prob) for c, prob in
                                     zip(p.behavior_classes, probs)},
                ))
        return events

    #  Rolling live badge inference 
    def rolling_predict(self, recent_features: list[np.ndarray]) -> tuple[str, float]:
        """
        Single forward pass over the last lstm_seq_len features.
        Used to update the live classification badge during frame iteration.
        Returns (label, confidence).
        """
        seq   = np.array(recent_features[-self._params.lstm_seq_len:])
        probs = self._infer(seq)
        label = self._params.behavior_classes[int(np.argmax(probs))]
        conf  = min(float(np.max(probs)), 0.99)
        return label, conf

    #  Private helpers
    def _infer(self, seq: np.ndarray) -> np.ndarray:
        """Run one forward pass and return softmax probabilities."""
        x_t = torch.FloatTensor(seq[np.newaxis]).to(self._device)
        with torch.no_grad():
            logits, _ = self._model(x_t)
            probs     = torch.softmax(logits / self.TEMPERATURE,
                                      dim=1).cpu().numpy()[0]
        return probs
