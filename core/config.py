"""
config.py - Application-wide constants and paths.

All values live in AppConfig. No logic, no I/O. Import this instead of
defining constants at module level in app.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class ModelPaths:
    """Filesystem paths for every model weight and info file."""
    yolo_dw_v2:     Path = field(default_factory=lambda: MODELS_DIR / "yolo26_dw_v2.pt")
    yolo_base:      Path = field(default_factory=lambda: MODELS_DIR / "yolo26n.pt")
    yolo_retail:    Path = field(default_factory=lambda: MODELS_DIR / "yolo26_retail.pt")
    mobilenet:      Path = field(default_factory=lambda: MODELS_DIR / "mobilenet_dw.pt")
    bilstm:         Path = field(default_factory=lambda: MODELS_DIR / "bilstm_dw.pt")
    bilstm_info:    Path = field(default_factory=lambda: MODELS_DIR / "bilstm_dw_info.json")
    mobilenet_info: Path = field(default_factory=lambda: MODELS_DIR / "mobilenet_dw_info.json")
    confusion_matrix: Path = field(default_factory=lambda: OUTPUTS_DIR / "confusion_matrix.png")
    learning_curve:   Path = field(default_factory=lambda: OUTPUTS_DIR / "learning_curve.png")


@dataclass(frozen=True)
class PipelineParams:
    """Hyper-parameters for the inference pipeline."""
    yolo_conf:        float = 0.20
    yolo_iou:         float = 0.45
    mobilenet_dim:    int   = 1280
    lstm_seq_len:     int   = 45
    lstm_stride:      int   = 15
    feat_step:        int   = 4
    yolo_step:        int   = 2
    shop_threshold:   float = 0.70
    review_threshold: float = 0.50
    behavior_classes: tuple = ("normal", "shoplifting")


# Singleton-style defaults - callers import these directly
DEFAULT_PATHS  = ModelPaths()
DEFAULT_PARAMS = PipelineParams()

PRODUCT_CATALOG: list[dict] = [
    {"sku": "ITEM001", "name": "Snack Bar",      "price": 2.99},
    {"sku": "ITEM002", "name": "Soda Bottle",    "price": 1.99},
    {"sku": "ITEM003", "name": "Chocolate Box",  "price": 5.99},
    {"sku": "ITEM004", "name": "Energy Drink",   "price": 3.49},
    {"sku": "ITEM005", "name": "Chips Bag",      "price": 2.49},
    {"sku": "ITEM006", "name": "Candy Pack",     "price": 1.49},
    {"sku": "ITEM007", "name": "Gum Pack",       "price": 0.99},
    {"sku": "ITEM008", "name": "Protein Bar",    "price": 3.99},
    {"sku": "ITEM009", "name": "Water Bottle",   "price": 1.29},
    {"sku": "ITEM010", "name": "Coffee Can",     "price": 2.79},
]
