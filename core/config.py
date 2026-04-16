"""
config.py - Application-wide constants and paths.

All values live here. No logic, no I/O. Import this instead of
defining constants scattered across multiple files.
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

    # yolo26_dw_v2.pt - our domain-adaptive fine-tune of YOLO26 on retail/surveillance frames
    # yolo26n.pt      - the base COCO-pretrained YOLO26 nano (used as fallback)
    # yolo26_retail.pt - older retail-specific fine-tune (lowest priority)
    #
    # YOLO model family:
    #   Jocher, G. et al. (2023). Ultralytics YOLO. GitHub.
    #   https://github.com/ultralytics/ultralytics
    #   Licensed under AGPL-3.0.
    yolo_dw_v2:  Path = field(default_factory=lambda: MODELS_DIR / "yolo26_dw_v2.pt")
    yolo_base:   Path = field(default_factory=lambda: MODELS_DIR / "yolo26n.pt")
    yolo_retail: Path = field(default_factory=lambda: MODELS_DIR / "yolo26_retail.pt")

    # mobilenet_dw.pt - MobileNetV2 fine-tuned on our frame dataset
    # Original MobileNetV2: Sandler et al. (2018) https://arxiv.org/abs/1801.04381
    # Pretrained weights: torchvision ImageNet-1K
    mobilenet: Path = field(default_factory=lambda: MODELS_DIR / "mobilenet_dw.pt")

    # bilstm_dw.pt - BiLSTM+Attention classifier trained on extracted features
    bilstm: Path = field(default_factory=lambda: MODELS_DIR / "bilstm_dw.pt")

    # JSON files written by the training notebook with accuracy / eval metrics
    bilstm_info:    Path = field(default_factory=lambda: MODELS_DIR / "bilstm_dw_info.json")
    mobilenet_info: Path = field(default_factory=lambda: MODELS_DIR / "mobilenet_dw_info.json")

    # plots saved during training - shown in the old model performance tab
    confusion_matrix: Path = field(default_factory=lambda: OUTPUTS_DIR / "confusion_matrix.png")
    learning_curve:   Path = field(default_factory=lambda: OUTPUTS_DIR / "learning_curve.png")


@dataclass(frozen=True)
class PipelineParams:
    """Hyper-parameters for the inference pipeline.

    These must match what was used in the training notebook exactly,
    otherwise the model will receive inputs it was never trained on.
    """
    yolo_conf:  float = 0.20   # minimum detection confidence to keep a box
    yolo_iou:   float = 0.45   # NMS IoU threshold

    mobilenet_dim: int = 1280  # MobileNetV2 feature dimension (no projection)

    # sliding window settings - seq_len=45 @ feat_step=4 covers roughly 6s of video
    lstm_seq_len: int = 45
    lstm_stride:  int = 15
    feat_step:    int = 4   # extract CNN features every 4th frame (speeds things up)
    yolo_step:    int = 2   # run YOLO every 2nd frame; reuse result otherwise

    # classification threshold - only YOLO peak confidence is used to decide
    shop_threshold:   float = 0.70   # >= this -> SHOPLIFTING
    review_threshold: float = 0.50   # used for segment counting and clip extraction

    behavior_classes: tuple = ("normal", "shoplifting")


# module-level singletons - most of the codebase just imports these directly
DEFAULT_PATHS  = ModelPaths()
DEFAULT_PARAMS = PipelineParams()

# mock product catalog used by the POS audit tab
# in a real deployment this would come from the store's inventory system
PRODUCT_CATALOG: list[dict] = [
    {"sku": "ITEM001", "name": "Snack Bar",     "price": 2.99},
    {"sku": "ITEM002", "name": "Soda Bottle",   "price": 1.99},
    {"sku": "ITEM003", "name": "Chocolate Box", "price": 5.99},
    {"sku": "ITEM004", "name": "Energy Drink",  "price": 3.49},
    {"sku": "ITEM005", "name": "Chips Bag",     "price": 2.49},
    {"sku": "ITEM006", "name": "Candy Pack",    "price": 1.49},
    {"sku": "ITEM007", "name": "Gum Pack",      "price": 0.99},
    {"sku": "ITEM008", "name": "Protein Bar",   "price": 3.99},
    {"sku": "ITEM009", "name": "Water Bottle",  "price": 1.29},
    {"sku": "ITEM010", "name": "Coffee Can",    "price": 2.79},
]
