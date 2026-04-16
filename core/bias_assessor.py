"""
bias_assessor.py - Applies fairness adjustments to the raw intent score.

Isolated fairness logic so it can be audited and updated independently
of the main scoring logic.

Why this exists: retail surveillance systems can produce biased results
for many reasons - bad lighting, camera angle, model training gaps, etc.
This class is the place to track those known issues and flag them. Right now
the only automated check is whether the LSTM was actually trained. More checks
can be added here without touching the rest of the pipeline.

The term "fairness" here is used loosely in the sense of result reliability,
not demographic fairness. See bias_assessor comments for more context.
"""
from __future__ import annotations
from core.result_types import IntentScore, BiasReport, QualityAnalysis


class BiasAssessor:
    """
    Produces a BiasReport and an adjusted risk score.

    Currently the only bias signal is whether the LSTM checkpoint was found.
    A missing checkpoint means predictions are random - the score is halved
    and the fairness score is reduced to flag unreliable results.

    In a production system you would also check:
    - camera resolution / lighting conditions
    - whether the demographic of the person matches training data coverage
    - time of day (night footage is much harder)
    For this FYP we keep it simple and just check the checkpoint.
    """

    _DEFAULT_FAIRNESS    = 0.85
    _UNTRAINED_FAIRNESS  = 0.50
    # if the LSTM is untrained, halve the score so it can't cross the alert threshold
    _UNTRAINED_SCALE     = 0.50

    def assess(self,
               intent: IntentScore,
               bilstm_trained: bool,
               detection_rate: float = 1.0) -> tuple[float, BiasReport, QualityAnalysis]:
        """
        Args:
            intent         : raw IntentScore from IntentScorer
            bilstm_trained : whether the LSTM checkpoint exists on disk
            detection_rate : fraction of frames where features were extracted

        Returns:
            (adjusted_score, BiasReport, QualityAnalysis)
        """
        flags: list[str] = []

        if not bilstm_trained:
            fairness  = self._UNTRAINED_FAIRNESS
            adj_score = intent.score * self._UNTRAINED_SCALE
            flags.append(
                "LSTM model not trained - results are random, do not act on them"
            )
        else:
            fairness  = self._DEFAULT_FAIRNESS
            adj_score = intent.score

        # make sure we didn't somehow go out of range
        adj_score = max(0.0, min(1.0, adj_score))

        bias = BiasReport(
            overall_fairness_score = fairness,
            analysis_reliable      = fairness >= 0.4,
            requires_review        = adj_score > 0.5 or bool(flags),
            flags                  = tuple(flags),
        )
        quality = QualityAnalysis(
            reliability_score = fairness,
            detection_rate    = detection_rate,
            usable            = fairness >= 0.4,
        )
        return adj_score, bias, quality
