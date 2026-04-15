"""
model_registry.py — Discovers and validates model weight files on disk.

Single responsibility: know which files exist and expose validated paths.
No model loading — that is delegated to FeatureExtractor and BehaviourClassifier.
"""
from __future__ import annotations
import json
from pathlib import Path

from core.config import ModelPaths, DEFAULT_PATHS


class ModelRegistry:
    """
    Discovers model weight files and exposes metadata.
    Constructed once per application run.
    """

    def __init__(self, paths: ModelPaths = DEFAULT_PATHS) -> None:
        self._paths = paths

    # ── Model path resolution ─────────────────────────────────────────────────

    def active_yolo_path(self) -> str:
        """
        Return the best available YOLO weight path.
        Priority: domain-adaptive fine-tune > COCO base > retail fallback.
        """
        for candidate in (self._paths.yolo_dw_v2,
                          self._paths.yolo_base,
                          self._paths.yolo_retail):
            if candidate.exists():
                return str(candidate)
        # Last resort: let ultralytics auto-download the base
        return str(self._paths.yolo_base)

    # ── Availability checks ───────────────────────────────────────────────────

    def bilstm_exists(self) -> bool:
        return self._paths.bilstm.exists()

    def mobilenet_exists(self) -> bool:
        return self._paths.mobilenet.exists()

    def yolo_exists(self) -> bool:
        return (self._paths.yolo_dw_v2.exists() or
                self._paths.yolo_base.exists() or
                self._paths.yolo_retail.exists())

    def system_status(self) -> dict[str, bool]:
        """Returns a status dict consumed by the sidebar renderer."""
        yolo_ready     = self.yolo_exists()
        mobilenet_ready = self.mobilenet_exists()
        bilstm_ready   = self.bilstm_exists()
        return {
            "yolo_ready"      : yolo_ready,
            "mobilenet_ready" : mobilenet_ready,
            "bilstm_ready"    : bilstm_ready,
            "all_ready"       : yolo_ready and mobilenet_ready and bilstm_ready,
        }

    # ── Metric file loading ───────────────────────────────────────────────────

    def load_bilstm_info(self) -> dict | None:
        """Return parsed bilstm_dw_info.json or None if not present."""
        return self._load_json(self._paths.bilstm_info)

    def load_mobilenet_info(self) -> dict | None:
        """Return parsed mobilenet_dw_info.json or None if not present."""
        return self._load_json(self._paths.mobilenet_info)

    @property
    def paths(self) -> ModelPaths:
        return self._paths

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None
