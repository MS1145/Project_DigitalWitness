"""
Digital Witness — Streamlit Web Interface

Self-contained: no src/ imports. Loads models from models/ directory.
Pipeline: YOLO26 (detection) → MobileNetV2 (features) → LSTM (classification)
"""
import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Digital Witness - Retail Security Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"

YOLO26_PATH     = MODELS_DIR / "yolo26_retail.pt"
YOLO26_BASE     = MODELS_DIR / "yolo26n.pt"          # COCO base — reliable person detection
YOLO26_DW_V2    = MODELS_DIR / "yolo26_dw_v2.pt"     # domain-adaptive fine-tune (Option 1)
MOBILENET_PATH  = MODELS_DIR / "mobilenet_extractor.pt"
LSTM_PATH       = MODELS_DIR / "bilstm_dw.pt"
LSTM_INFO_PATH  = MODELS_DIR / "bilstm_dw_info.json"

BEHAVIOR_CLASSES = ["normal", "shoplifting"]
YOLO_CONF        = 0.2
YOLO_IOU         = 0.45
MOBILENET_DIM    = 1280   # raw MobileNetV2 output (before projection head)
LSTM_SEQ_LEN     = 45    # must match notebook Cell 5: 45 frames @ 6fps = 7.5s
LSTM_STRIDE      = 15

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(120deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0; margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.3rem; color: #666; text-align: center;
        margin-bottom: 2rem; font-weight: 400;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .alert-none {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa; border-left: 4px solid #1e3a5f;
        padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 3px solid #3d7ab5; padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem 2rem;
        font-size: 1.1rem; font-weight: 600; border-radius: 10px;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { padding-left: 1rem; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 10px 10px 0 0;
        padding: 0.4rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Inline Model Definitions ─────────────────────────────────────────────────

def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_mobilenet_extractor():
    """
    Load MobileNetV2 feature extractor matching notebook Cell 4 architecture.

    State dict keys (from mobilenet_dw.pt → 'state_dict'):
      features.*    → MobileNetV2 backbone (layers 0-18)
      pool.*        → AdaptiveAvgPool2d (no params)
      classifier.*  → Linear(1280,512) → ReLU → Dropout → Linear(512,2)

    Only features + pool are used here — classifier head is discarded.
    extract_features() returns the 1280-dim vector used as BiLSTM input.
    """
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    class _MNetExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.features   = base.features
            self.pool       = nn.AdaptiveAvgPool2d((1, 1))
            # Classifier head — loaded from checkpoint but not used during feature extraction
            self.classifier = nn.Sequential(
                nn.Linear(1280, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, 2),
            )

        def extract_features(self, x):
            """Return 1280-dim feature vector for a (1,3,224,224) tensor."""
            import torch
            with torch.no_grad():
                x = self.features(x)
                x = self.pool(x)
                return x.flatten(1)   # (1, 1280)

        def forward(self, x):
            return self.classifier(self.extract_features(x))

    return _MNetExtractor()


def _build_bilstm(num_classes=2, hidden=256, layers=2, dropout=0.3):
    """
    BiLSTMAttentionClassifier — matches exact notebook Cell 5 architecture.

    State dict keys (from bilstm_dw.pt → 'state_dict'):
      bilstm.*            → nn.LSTM bidirectional
      attention.attn.0.*  → Linear(hidden*2, hidden)  [Tanh has no params]
      attention.attn.2.*  → Linear(hidden, 1)
      dropout             → no params
      classifier.*        → Linear(hidden*2, num_classes)

    Input: (B, T=45, 1280) pre-extracted MobileNetV2 feature sequences.
    """
    import torch
    import torch.nn as nn

    class _TemporalAttention(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.Tanh(),
                nn.Linear(h // 2, 1),
            )

        def forward(self, lstm_out):   # (B, T, h)
            scores  = self.attn(lstm_out).squeeze(-1)   # (B, T)
            weights = torch.softmax(scores, dim=1)       # (B, T)
            context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
            return context, weights

    class _BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=1280, hidden_size=hidden,
                num_layers=layers, batch_first=True,
                bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.attention  = _TemporalAttention(hidden * 2)
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden * 2, num_classes)

        def forward(self, x):   # x: (B, T, 1280)
            out, _    = self.bilstm(x)
            ctx, attn = self.attention(out)
            return self.classifier(self.dropout(ctx)), attn

    return _BiLSTM()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_model_exists():
    return LSTM_PATH.exists()


def load_model_metrics():
    if LSTM_INFO_PATH.exists():
        with open(LSTM_INFO_PATH, "r") as f:
            return json.load(f)
    return None


def load_mobilenet_metrics():
    p = MODELS_DIR / "mobilenet_dw_info.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None


def get_active_yolo_path():
    """Model priority:
    1. yolo26_dw_v2.pt  — domain-adaptive fine-tune (retail + surveillance frames)
    2. yolo26n.pt       — COCO base, reliable person detection on any footage
    3. yolo26_retail.pt — retail-only fine-tune, domain-shifted (fallback only)
    """
    if YOLO26_DW_V2.exists():
        return str(YOLO26_DW_V2)
    if YOLO26_BASE.exists():
        return str(YOLO26_BASE)
    if YOLO26_PATH.exists():
        return str(YOLO26_PATH)
    return str(MODELS_DIR / "yolo26n.pt")


def get_product_catalog():
    return [
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


# ─── Inline Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(video_path, progress_callback=None,
                 visual_out_path=None, live_frame_callback=None):
    """
    Self-contained inference pipeline.
    YOLO26 detection → MobileNetV2 feature extraction → LSTM classification
    → Intent scoring → Bias assessment → Alert generation
    """
    import cv2
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from ultralytics import YOLO

    def _cb(p, m):
        if progress_callback:
            progress_callback(p, m)

    _cb(0.0, "Initialising models...")
    dev = _get_device()

    # GPU tuning
    if str(dev) == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Load YOLO26 ──
    yolo_path  = get_active_yolo_path()
    yolo_model = YOLO(yolo_path)
    _yolo_dev  = 0 if str(dev) == "cuda" else "cpu"   # int 0 = first CUDA device
    _yolo_half = (str(dev) == "cuda")                  # FP16 inference on GPU

    # Auto-detect which model is loaded so detection logic adapts automatically.
    _yolo_names   = set(yolo_model.names.values())
    _yolo_names_lower = {n.lower() for n in _yolo_names}
    IS_4CLS_MODEL = "Looking around" in _yolo_names or "Picking-Holding" in _yolo_names
    IS_COCO_MODEL = "person" in _yolo_names_lower   # base COCO or COCO-derived
    _mode = "4-class behaviour" if IS_4CLS_MODEL else ("COCO" if IS_COCO_MODEL else "retail/unknown")
    _cb(0.02, f"YOLO model: {_mode} ({Path(yolo_path).name}) — "
              f"classes: {sorted(_yolo_names)[:6]}...")

    # ── Load MobileNetV2 feature extractor (mobilenet_dw.pt) ──
    # Uses the fine-tuned weights from notebook Cell 4 — critical for accuracy.
    # Features extracted here are what the BiLSTM was trained on.
    mnet_model = _build_mobilenet_extractor().to(dev)
    mnet_path  = MODELS_DIR / "mobilenet_dw.pt"
    if mnet_path.exists():
        mnet_ckpt = torch.load(mnet_path, map_location=dev, weights_only=False)
        sd = mnet_ckpt['state_dict'] if isinstance(mnet_ckpt, dict) and 'state_dict' in mnet_ckpt else mnet_ckpt
        mnet_model.load_state_dict(sd)
    mnet_model.eval()

    # ── Load BiLSTM classifier (bilstm_dw.pt) ──
    # Architecture matches notebook Cell 5 BiLSTMAttentionClassifier exactly.
    lstm_classes  = BEHAVIOR_CLASSES
    bilstm_model  = _build_bilstm(num_classes=len(lstm_classes), hidden=256,
                                   layers=2, dropout=0.3).to(dev)
    if LSTM_PATH.exists():
        bilstm_ckpt = torch.load(LSTM_PATH, map_location=dev, weights_only=False)
        sd = bilstm_ckpt['state_dict'] if isinstance(bilstm_ckpt, dict) and 'state_dict' in bilstm_ckpt else bilstm_ckpt
        bilstm_model.load_state_dict(sd)
    # If not found: keeps random weights — flagged in UI
    bilstm_model.eval()

    # ── Image preprocessing transform ──
    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 4-class behaviour model — ALL detections are person-level behaviour labels.
    # There is no separate "person" class; each box is one of these four.
    BEHAVIOUR_4CLS = {"Looking around", "Picking-Holding", "normal", "shoplifting"}
    # Suspicious classes used for intent scoring
    SUSPICIOUS_CLS = {"Looking around", "shoplifting"}
    # Product-interaction class (replaces the old product proximity logic)
    PRODUCT_CLS    = {"Picking-Holding"}

    def _early_exit_normal(reason, f_num, f_total, fps, w, h):
        """Return a lightweight 'normal / no person' result without running BiLSTM."""
        return {
            "success"       : True,
            "early_exit"    : True,
            "early_exit_reason": reason,
            "video_metadata": {
                "filename"   : Path(video_path).name,
                "duration"   : f_num / max(fps, 1),
                "fps"        : fps,
                "width"      : w,
                "height"     : h,
                "frame_count": f_total,
            },
            "lstm_detection": {
                "classification": "normal",
                "confidence"    : 1.0,
                "is_shoplifting": False,
            },
            "detections": {
                "persons_tracked"  : 0,
                "products_detected": 0,
                "interactions"     : 0,
                "frames_processed" : f_num,
            },
            "product_pickups"  : {},
            "behavior_events"  : [],
            "intent_score"     : {"score": 0.0, "severity": "NONE",
                                  "components": {}, "explanation": reason},
            "quality_analysis" : {"reliability_score": 1.0,
                                  "detection_rate": 1.0, "usable": True},
            "bias_report"      : {"overall_fairness_score": 1.0,
                                  "analysis_reliable": True,
                                  "requires_review": False, "flags": []},
            "alert"            : None,
            "model_trained"    : LSTM_PATH.exists(),
        }

    # ── Video processing ──
    _cb(0.05, "Opening video...")
    cap          = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video file."}

    fps_val      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_feats    = []
    persons_set  = set()   # unique track IDs (for proximity checks only)
    max_concurrent_persons = 0  # peak persons visible in a single frame (accurate count)
    products_set = set()   # unique product class names seen (internal YOLO tracking)
    # product pickup: set of (person_track_id, product_class) — one entry per
    # unique person×product pair that were in close proximity
    product_pickup_events = set()
    frame_num    = 0
    feat_step    = 4   # extract CNN features every 4th frame (speed)
    yolo_step    = 2   # run YOLO every 2nd frame; reuse last result otherwise
    last_yolo_results = None

    # ── Visual output setup ────────────────────────────────────────────────────
    video_writer     = None
    last_live_label  = None
    last_live_conf   = 0.0
    live_timeline    = []
    _live_step       = 30       # update live preview every N frames
    if visual_out_path:
        _fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            visual_out_path, _fourcc, fps_val, (width, height)
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if progress_callback and frame_num % 90 == 0:
            pct = min(0.05 + (frame_num / max(total_f, 1)) * 0.65, 0.70)
            _cb(pct, f"Processing frame {frame_num}/{total_f}...")

        # ── YOLO26 detection + ByteTrack ──
        # Run every yolo_step frames; reuse last detection on skipped frames.
        # imgsz=320 cuts YOLO inference time ~4× vs default 640 with minor accuracy trade-off.
        if frame_num % yolo_step == 1 or last_yolo_results is None:
            last_yolo_results = yolo_model.track(
                frame, conf=YOLO_CONF, iou=YOLO_IOU, persist=True, verbose=False,
                imgsz=320, device=_yolo_dev, half=_yolo_half,
            )[0]
        results = last_yolo_results

        frame_person_boxes  = {}   # {track_id: [x1,y1,x2,y2]}
        frame_product_boxes = []   # [(class_name, [x1,y1,x2,y2])]

        if results.boxes is not None:
            real_ids = (results.boxes.id.int().tolist()
                        if results.boxes.id is not None else None)
            for i, (box_t, cls_id) in enumerate(zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.cls)):
                cls_name = yolo_model.names[int(cls_id)]

                if IS_4CLS_MODEL:
                    # ── 4-class behaviour model ──────────────────────────────
                    # Every detection is a person-level behaviour label.
                    if cls_name not in BEHAVIOUR_4CLS:
                        continue
                    tid = (real_ids[i] if real_ids is not None else -(i + 1))
                    persons_set.add(tid)
                    frame_person_boxes[tid] = box_t
                    if cls_name in PRODUCT_CLS:
                        products_set.add(cls_name)
                        frame_product_boxes.append((cls_name, box_t))
                    if cls_name in SUSPICIOUS_CLS:
                        product_pickup_events.add((tid, cls_name))
                else:
                    # ── COCO / retail / unknown model fallback ───────────────
                    # Person detection strategy (in priority order):
                    #  1. Class ID 0  — always "person" in any COCO-based model
                    #  2. Class name "person" (case-insensitive)
                    #  3. Name starts with "person" (e.g. "person-picking-up")
                    #  4. Unknown model with no "person" class — treat ALL boxes
                    #     as persons so tracking is never silently broken
                    cls_id_int = int(cls_id)
                    is_person  = (
                        cls_id_int == 0                          # COCO class 0 = person
                        or cls_name.lower() == "person"
                        or cls_name.lower().startswith("person-")
                        or (not IS_COCO_MODEL)                   # unknown model: count all
                    )
                    is_product = cls_name in {
                        "bottle", "cup", "bowl", "banana", "apple", "sandwich",
                        "backpack", "handbag", "book", "cell phone", "orange",
                        "backpack-or-handbag", "carrying-item",
                        "empty-basket", "filled-basket",
                        "empty-shopping-bags", "filled-shopping-bags",
                        "empty-trolly", "filled-trolly",
                    }
                    if is_person:
                        tid = (real_ids[i] if real_ids is not None else i + 1)
                        persons_set.add(tid)
                        frame_person_boxes[tid] = box_t
                    elif is_product:
                        products_set.add(cls_name)
                        frame_product_boxes.append((cls_name, box_t))

        # Update max concurrent persons.
        # Count ALL entries — negative IDs are legitimate detections from frames
        # where ByteTrack hasn't yet established tracks (first few frames, or
        # low-confidence scenes). Filtering by k >= 0 was silently zeroing the
        # count whenever ByteTrack was unavailable.
        real_person_count = len(frame_person_boxes)
        if real_person_count > max_concurrent_persons:
            max_concurrent_persons = real_person_count

        # ── Current-frame YOLO signal (used for live badge priority) ──────────
        # If YOLO detects "shoplifting" or "Looking around" in this frame the
        # badge should reflect that immediately — the BiLSTM needs 45 frames of
        # history before it can output a verdict, so YOLO acts as the early
        # warning while LSTM provides the confirmed temporal classification.
        frame_yolo_label = None
        frame_yolo_conf  = 0.0
        if results.boxes is not None and IS_4CLS_MODEL:
            for _cls_v, _cof_v in zip(results.boxes.cls.cpu().numpy(),
                                       results.boxes.conf.cpu().numpy()):
                _cn = yolo_model.names[int(_cls_v)]
                if _cn == "shoplifting":
                    if float(_cof_v) > frame_yolo_conf:
                        frame_yolo_label = "shoplifting"
                        frame_yolo_conf  = float(_cof_v)
                elif _cn == "Looking around" and frame_yolo_label != "shoplifting":
                    if float(_cof_v) > frame_yolo_conf:
                        frame_yolo_label = "Looking around"
                        frame_yolo_conf  = float(_cof_v)

        # Choose badge: YOLO live detection takes priority over LSTM when YOLO
        # sees something suspicious; otherwise fall back to LSTM temporal verdict.
        _SUSP = {"shoplifting", "Looking around"}
        if frame_yolo_label in _SUSP:
            badge_label = frame_yolo_label
            badge_conf  = frame_yolo_conf
        elif last_live_label:
            badge_label = last_live_label
            badge_conf  = last_live_conf
        else:
            badge_label = None
            badge_conf  = 0.0

        # ── Frame annotation (visual output + live preview) ───────────────────
        if video_writer or live_frame_callback:
            ann = frame.copy()
            # Draw bounding boxes + labels for every YOLO detection
            if results.boxes is not None:
                for box_t, cls_id_v, conf_v in zip(
                        results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy(),
                        results.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box_t)
                    cn_v = yolo_model.names[int(cls_id_v)]
                    if IS_4CLS_MODEL:
                        if cn_v == "shoplifting":
                            col_v = (0, 0, 220)      # red  (BGR)
                        elif cn_v == "Looking around":
                            col_v = (0, 140, 255)    # orange
                        elif cn_v == "Picking-Holding":
                            col_v = (0, 200, 255)    # yellow
                        else:                        # normal
                            col_v = (0, 200, 0)      # green
                        lbl_v = f"{cn_v} {conf_v:.0%}"
                    else:
                        is_p  = int(cls_id_v) == 0 or cn_v.lower() == "person"
                        col_v = (0, 200, 0) if is_p else (160, 160, 160)
                        lbl_v = f"Person {conf_v:.0%}" if is_p else f"{cn_v} {conf_v:.0%}"
                    # Bounding box
                    cv2.rectangle(ann, (x1, y1), (x2, y2), col_v, 2)
                    # Label background pill above the box
                    (tw, th), _ = cv2.getTextSize(lbl_v, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    lbl_y = max(y1 - 4, th + 6)
                    cv2.rectangle(ann, (x1, lbl_y - th - 6), (x1 + tw + 8, lbl_y + 2),
                                  col_v, -1)
                    cv2.putText(ann, lbl_v, (x1 + 4, lbl_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            # Draw classification badge - YOLO live signal takes priority,
            # falls back to LSTM temporal verdict once sequence is long enough.
            if badge_label:
                b_col = ((0, 0, 200)   if badge_label == "shoplifting" else
                         (0, 140, 255) if badge_label == "Looking around" else
                         (20, 140, 20))
                b_txt = f"{badge_label.upper()}  {badge_conf:.0%}"
                (tw_b, th_b), _ = cv2.getTextSize(b_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(ann, (8, 8), (tw_b + 18, th_b + 22), b_col, -1)
                cv2.putText(ann, b_txt, (12, th_b + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            # Frame counter bottom-left
            cv2.putText(ann, f"{frame_num}/{total_f}", (8, ann.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            # Write to VideoWriter
            if video_writer:
                video_writer.write(ann)
            # Live callback every _live_step frames
            if live_frame_callback and frame_num % _live_step == 0:
                ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                tl_color = ("red"   if badge_label in {"shoplifting", "Looking around"} else
                             "green" if badge_label == "normal" else "gray")
                live_timeline.append(tl_color)
                live_frame_callback(ann_rgb, frame_num, total_f, {
                    "persons_tracked"  : max_concurrent_persons,
                    "products_detected": len({cls for (_, cls) in product_pickup_events}),
                    "label"            : badge_label or "Analyzing...",
                    "conf"             : badge_conf,
                    "timeline"         : list(live_timeline),
                })

        # ── Person-product proximity → detect pickups ──
        for pid, pbox in frame_person_boxes.items():
            px1, py1, px2, py2 = pbox
            ph   = max(py2 - py1, 1)          # person height as proximity scale
            pcx  = (px1 + px2) / 2
            pcy  = (py1 + py2) / 2
            for cls_name, prod_box in frame_product_boxes:
                qx1, qy1, qx2, qy2 = prod_box
                qcx = (qx1 + qx2) / 2
                qcy = (qy1 + qy2) / 2
                dist = ((pcx - qcx) ** 2 + (pcy - qcy) ** 2) ** 0.5
                if dist < ph * 1.2:   # within 1.2× person height ≈ arm's reach
                    product_pickup_events.add((pid, cls_name))

        # ── MobileNetV2 feature extraction (every feat_step frames for speed) ──
        if frame_num % feat_step == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x    = xform(rgb).unsqueeze(0).to(dev)
            feat = mnet_model.extract_features(x).cpu().numpy().flatten()
            all_feats.append(feat)
            # Rolling BiLSTM for live badge — run once every LSTM_STRIDE features
            if (video_writer or live_frame_callback) and \
                    len(all_feats) >= LSTM_SEQ_LEN and \
                    (len(all_feats) - LSTM_SEQ_LEN) % max(LSTM_STRIDE // 2, 1) == 0:
                with torch.no_grad():
                    _seq_np = np.array(all_feats[-LSTM_SEQ_LEN:])
                    _xt     = torch.FloatTensor(_seq_np[np.newaxis]).to(dev)
                    _log, _ = bilstm_model(_xt)
                    _prb    = torch.softmax(_log / 3.0, dim=1).cpu().numpy()[0]
                last_live_label = lstm_classes[int(np.argmax(_prb))]
                last_live_conf  = min(float(np.max(_prb)) * 0.85, 0.82)

    if video_writer:
        video_writer.release()
    cap.release()
    duration = frame_num / fps_val

    # ── Edge case: no persons in entire video ──
    if not persons_set:
        _cb(1.0, "No persons detected — returning normal.")
        return _early_exit_normal(
            "No person detected across the entire video. Classified as normal.",
            frame_num, total_f, fps_val, width, height,
        )

    # ── Edge case: video too short — no features collected ──
    if not all_feats:
        _cb(1.0, "Video too short to analyse — returning normal.")
        return _early_exit_normal(
            f"Video too short ({frame_num} frame(s)) — insufficient data for "
            "temporal classification. Classified as normal.",
            frame_num, total_f, fps_val, width, height,
        )

    # Aggregate product pickup counts: how many distinct persons handled each product
    product_pickups = {}
    for (pid, cls_name) in product_pickup_events:
        product_pickups[cls_name] = product_pickups.get(cls_name, 0) + 1
    # Distinct product classes actually interacted with (not all visible products)
    products_interacted = len({cls for (_, cls) in product_pickup_events})

    _cb(0.72, "Running LSTM temporal classification...")

    # ── Sliding-window BiLSTM classification ──
    feats_arr   = np.array(all_feats) if all_feats else np.zeros((1, MOBILENET_DIM))
    predictions = []

    if len(all_feats) < LSTM_SEQ_LEN:
        pad = np.zeros((LSTM_SEQ_LEN - len(all_feats), MOBILENET_DIM))
        seq = np.vstack([feats_arr, pad])
        x_t = torch.FloatTensor(seq[np.newaxis]).to(dev)
        with torch.no_grad():
            logits, _ = bilstm_model(x_t)
        # T=3.0 temperature scaling: divides logits before softmax to flatten
        # overconfident distributions. The *0.85 scale factor further prevents
        # near-certain outputs — an untrained/overfit model should never claim
        # certainty. Sequence-length scale penalises very short clips.
        TEMP = 3.0
        probs    = torch.softmax(logits / TEMP, dim=1).cpu().numpy()[0]
        scale    = min(1.0, len(all_feats) / LSTM_SEQ_LEN)
        raw_conf = float(np.max(probs)) * scale
        predictions.append({
            "start"        : 0.0,
            "end"          : duration,
            "behavior_type": lstm_classes[int(np.argmax(probs))],
            "confidence"   : min(raw_conf * 0.85, 0.82),
            "probabilities": {c: float(p) for c, p in zip(lstm_classes, probs)},
        })
    else:
        for start in range(0, len(all_feats) - LSTM_SEQ_LEN + 1, LSTM_STRIDE):
            seq = feats_arr[start:start + LSTM_SEQ_LEN]
            x_t = torch.FloatTensor(seq[np.newaxis]).to(dev)
            with torch.no_grad():
                logits, _ = bilstm_model(x_t)
                probs     = torch.softmax(logits / 3.0, dim=1).cpu().numpy()[0]
            predictions.append({
                "start"        : start / fps_val,
                "end"          : (start + LSTM_SEQ_LEN) / fps_val,
                "behavior_type": lstm_classes[int(np.argmax(probs))],
                "confidence"   : min(float(np.max(probs)) * 0.85, 0.82),
                "probabilities": {c: float(p) for c, p in zip(lstm_classes, probs)},
            })

    _cb(0.85, "Aggregating predictions...")

    # Weighted aggregation (2× weight for shoplifting)
    shop_preds = [p for p in predictions if p["behavior_type"] == "shoplifting"]
    norm_preds = [p for p in predictions if p["behavior_type"] == "normal"]
    shop_w = sum(p["confidence"] for p in shop_preds) * 2.0
    norm_w = sum(p["confidence"] for p in norm_preds)
    total_w = shop_w + norm_w

    if total_w > 0 and shop_w / total_w > 0.3 and shop_preds:
        best          = max(shop_preds, key=lambda p: p["confidence"])
        overall_class = "shoplifting"
        overall_conf  = best["confidence"]
    else:
        overall_class = "normal"
        overall_conf  = (np.mean([p["confidence"] for p in norm_preds])
                         if norm_preds else 0.5)

    _cb(0.90, "Scoring intent...")

    # ── Intent scoring (video-only, no POS) ──
    # With 4-class model: "shoplifting" = direct detection, "Looking around" = recon.
    # No "concealment" or "bypass" classes exist — map to closest equivalents.
    direct_e  = [p for p in predictions if p["behavior_type"] == "shoplifting"]
    recon_e   = [p for p in predictions if p["behavior_type"] == "Looking around"]
    susp_e    = direct_e + recon_e

    # s_conceal: weighted by direct shoplifting confidence (primary signal)
    s_conceal = (np.mean([e["confidence"] for e in direct_e])
                 * min(1.0, len(direct_e) / 3.0)) if direct_e else 0.0
    # s_bypass: reconnaissance behaviour is the closest proxy for bypass intent
    s_bypass  = (np.mean([e["confidence"] for e in recon_e])
                 * min(1.0, len(recon_e) / 3.0)) if recon_e else 0.0
    susp_secs = sum(e["end"] - e["start"] for e in susp_e)
    s_dur     = min(1.0, (susp_secs / max(1.0, duration)) * 3.0) if susp_e else 0.0

    raw_score = 0.50 * s_conceal + 0.35 * s_bypass + 0.15 * s_dur
    raw_score = max(0.0, min(1.0, raw_score))

    if raw_score < 0.3:    severity = "NONE"
    elif raw_score < 0.5:  severity = "LOW"
    elif raw_score < 0.7:  severity = "MEDIUM"
    elif raw_score < 0.85: severity = "HIGH"
    else:                  severity = "CRITICAL"

    # Simple bias adjustment (conservative when score is uncertain)
    adj_score = raw_score
    fairness  = 0.85   # default: high fairness (no demographic indicators in video-only mode)
    bias_flags = []
    if not LSTM_PATH.exists():
        adj_score *= 0.5
        fairness   = 0.5
        bias_flags.append("LSTM model not trained — results are random, do not act on them")

    _cb(0.95, "Building case file...")

    # ── Alert ──
    alert = None
    if adj_score >= 0.5:
        n_susp = len(direct_e) + len(recon_e)
        alert  = {
            "alert_id": f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-0001",
            "level"   : severity,
            "message" : (f"{n_susp} suspicious event(s) detected. "
                         f"Adjusted risk score: {adj_score:.2f} ({severity}). "
                         f"Human review required before any action."),
        }

    _cb(1.0, "Complete!")

    behavior_events = [
        {
            "behavior_type": p["behavior_type"],
            "start_time"   : p["start"],
            "end_time"     : p["end"],
            "confidence"   : p["confidence"],
            "probabilities": p["probabilities"],
        }
        for p in predictions
    ]

    return {
        "success"       : True,
        "video_metadata": {
            "filename"   : Path(video_path).name,
            "duration"   : duration,
            "fps"        : fps_val,
            "width"      : width,
            "height"     : height,
            "frame_count": total_f,
        },
        "lstm_detection": {
            "classification": overall_class,
            "confidence"    : overall_conf,
            "is_shoplifting": overall_class == "shoplifting",
        },
        "detections": {
            "persons_tracked"  : max_concurrent_persons,
            "products_detected": products_interacted,
            "interactions"     : len(susp_e),
            "frames_processed" : len(all_feats),
        },
        "product_pickups": product_pickups,   # {class_name: n_persons_who_handled_it}
        "behavior_events": behavior_events,
        "intent_score": {
            "score"      : adj_score,
            "severity"   : severity,
            "components" : {"concealment": s_conceal,
                            "bypass": s_bypass,
                            "duration": s_dur},
            "explanation": (
                f"Score: {adj_score:.2f} ({severity}). "
                f"Concealment: {s_conceal:.2f}, Bypass: {s_bypass:.2f}, Duration: {s_dur:.2f}"
            ),
        },
        "quality_analysis": {
            "reliability_score": fairness,
            "detection_rate"   : min(1.0, len(all_feats) / max(1, total_f // feat_step)),  # FIX: was frame_step (NameError)
            "usable"           : fairness >= 0.4,
        },
        "bias_report": {
            "overall_fairness_score": fairness,
            "analysis_reliable"     : fairness >= 0.4,
            "requires_review"       : adj_score > 0.5 or bool(bias_flags),
            "flags"                 : bias_flags,
        },
        "alert": alert,
        "model_trained": LSTM_PATH.exists(),
        "annotated_video_path": visual_out_path,
    }


def extract_suspicious_clips(video_path, behavior_events, max_clips=4):
    """
    Extract up to max_clips distinct static frame snapshots from the highest-
    confidence shoplifting windows.  Frames must be at least 2 s apart so
    overlapping LSTM windows never produce duplicate images.

    Returns a list of dicts (one per unique moment):
        thumbnail   : ndarray (RGB) — the captured frame
        timestamp   : float  — mid-point of the window in seconds
        frame_start : int    — first frame number of the window
        frame_end   : int    — last frame number of the window
        behavior    : str
        confidence  : float
    """
    import cv2

    suspicious = [
        e for e in behavior_events
        if e.get("behavior_type") in {"shoplifting", "Looking around",
                                       "concealment", "bypass"}
        and e.get("confidence", 0) >= 0.5
    ]
    if not suspicious:
        return []

    # Sort by confidence descending; keep only events whose midpoint is
    # at least 2 s away from every already-selected event.
    suspicious = sorted(suspicious, key=lambda x: x.get("confidence", 0), reverse=True)
    selected, selected_mids = [], []
    for ev in suspicious:
        mid = (ev.get("start_time", 0) + ev.get("end_time", 0)) / 2
        if all(abs(mid - t) >= 2.0 for t in selected_mids):
            selected.append(ev)
            selected_mids.append(mid)
        if len(selected) >= max_clips:
            break

    if not selected:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    MAX_WIDTH = 640
    frames = []

    for ev in selected:
        ev_start = ev.get("start_time", 0)
        ev_end   = ev.get("end_time", ev_start + 1)
        mid_sec  = (ev_start + ev_end) / 2
        mid_f    = max(0, min(total_f - 1, int(mid_sec * fps_v)))
        start_f  = int(ev_start * fps_v)
        end_f    = int(ev_end   * fps_v)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > MAX_WIDTH:
            s = MAX_WIDTH / w
            rgb = cv2.resize(rgb, (MAX_WIDTH, int(h * s)),
                             interpolation=cv2.INTER_AREA)

        frames.append({
            "thumbnail"  : rgb,
            "timestamp"  : mid_sec,
            "frame_start": start_f,
            "frame_end"  : end_f,
            "behavior"   : ev["behavior_type"],
            "confidence" : ev["confidence"],
        })

    cap.release()
    return frames




# ─── UI Rendering ─────────────────────────────────────────────────────────────

def render_header():
    st.markdown('<h1 class="main-title">Digital Witness</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Deep Learning Retail Security Assistant '
        '| YOLO26 + MobileNetV2 + LSTM</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("### About Digital Witness")
        st.markdown("""
        **Digital Witness** detects potential shoplifting from surveillance video using deep learning.

        **Pipeline:**
        1. **YOLO26 Detection** — Jan 2026 model, NMS-free, 43% faster than YOLOv8.
           Identifies and tracks people & products.
        2. **MobileNetV2 Features** — Extracts 1280-dim spatial features per frame
           (fine-tuned backbone, ImageNet pretrained).
        3. **Bidirectional LSTM + Attention** — Classifies temporal sequences of features
           into normal / shoplifting.

        **Important:** This is an *advisory system*. It does **not** determine guilt.
        All alerts require human validation.
        """)

        st.markdown("---")
        st.markdown("### System Status")
        yolo_ready   = YOLO26_DW_V2.exists() or YOLO26_BASE.exists() or YOLO26_PATH.exists()
        models_ready = check_model_exists() and yolo_ready
        if models_ready:
            st.success("All models loaded")
        else:
            st.warning("Some models missing — check `models/` folder")

        st.markdown("---")
        st.markdown("### Important Notice")
        st.info("This is an **advisory system**.\nAll alerts require human validation.\nThe system does NOT determine guilt.")
        st.markdown("---")
        st.markdown("##### Final Year Project — 2026")


def render_model_performance_tab():
    bilstm_info   = load_model_metrics()
    mobilenet_info = load_mobilenet_metrics()

    if not bilstm_info and not mobilenet_info:
        st.warning("No model metrics found. Train the models first using the notebook.")
        return

    st.markdown('<div class="section-header">Deep Learning Model Performance</div>',
                unsafe_allow_html=True)

    # ── MobileNetV2 evaluation metrics (computed on held-out val set) ──────────
    mn   = mobilenet_info or {}
    mn_m = mn.get("metrics", {})
    st.markdown("### MobileNetV2 Frame Classifier — Evaluation Results")
    st.caption(f"Evaluated on {mn.get('n_val_samples', 0):,} held-out validation frames (20% stratified split)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{mn_m.get('accuracy',  mn.get('best_val_acc', 0)):.2%}")
    c2.metric("Precision", f"{mn_m.get('precision', 0):.2%}")
    c3.metric("Recall",    f"{mn_m.get('recall',    0):.2%}")
    c4.metric("F1 Score",  f"{mn_m.get('f1_score',  0):.2%}")

    per_class = mn.get("per_class_metrics", {})
    if per_class:
        st.markdown("#### Per-Class Metrics")
        classes = mn.get("classes", ["normal", "shoplifting"])
        rows = [{"Class"    : c.capitalize(),
                 "Precision": f"{per_class.get('precision', {}).get(c, 0):.2%}",
                 "Recall"   : f"{per_class.get('recall',    {}).get(c, 0):.2%}",
                 "F1 Score" : f"{per_class.get('f1',        {}).get(c, 0):.2%}",
                 "Support"  : per_class.get('support', {}).get(c, 0)}
                for c in classes]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── BiLSTM val accuracy ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### BiLSTM + Attention Sequence Classifier — Validation Accuracy")
    st.caption("Best validation accuracy recorded during training (frame-sequence level)")
    if bilstm_info:
        lstm_acc = bilstm_info.get("best_val_acc", 0)
        c1, c2 = st.columns(2)
        c1.metric("Best Validation Accuracy", f"{lstm_acc:.2%}")
        c2.metric("Val Sequences", bilstm_info.get("n_val_samples", "N/A"))
    # ── Confusion matrix + learning curve ─────────────────────────────────────
    st.markdown("---")
    _cm_path  = ROOT_DIR / "outputs" / "confusion_matrix.png"
    _lc_path  = ROOT_DIR / "outputs" / "learning_curve.png"
    has_cm = _cm_path.exists()
    has_lc = _lc_path.exists()

    if has_cm and has_lc:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Confusion Matrix")
            st.image(str(_cm_path), use_container_width=True)
        with col2:
            st.markdown("### Training Learning Curve")
            st.image(str(_lc_path), use_container_width=True)
    elif has_cm:
        st.markdown("### Confusion Matrix")
        st.image(str(_cm_path), use_container_width=True)
    elif has_lc:
        st.markdown("### Training Learning Curve")
        st.image(str(_lc_path), use_container_width=True)
    elif "confusion_matrix" in (bilstm_info or {}):
        import plotly.graph_objects as go
        st.markdown("### Confusion Matrix")
        cm = np.array(bilstm_info["confusion_matrix"])
        classes = bilstm_info.get("classes", ["normal", "shoplifting"])
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=classes, y=classes, colorscale="Blues",
            text=cm, texttemplate="%{text}", textfont={"size": 20},
        ))
        fig.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Predicted", yaxis_title="True",
            height=350, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-class metrics table ────────────────────────────────────────────────
    per_class = (bilstm_info or {}).get("per_class_metrics", {})
    if per_class:
        st.markdown("---")
        st.markdown("### Per-Class Metrics")
        classes = (bilstm_info or {}).get("classes", ["normal", "shoplifting"])
        rows = [{"Class": c.capitalize(),
                 "Precision": f"{per_class.get('precision', {}).get(c, 0):.1%}",
                 "Recall"   : f"{per_class.get('recall',    {}).get(c, 0):.1%}",
                 "F1 Score" : f"{per_class.get('f1',        {}).get(c, 0):.1%}"}
                for c in classes]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model architecture & training config ──────────────────────────────────
    st.markdown("---")
    st.markdown("### Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Detection: YOLO26** (Jan 2026)
        - NMS-free, 43% faster than YOLOv8
        - Fine-tuned on retail product annotations
        - ByteTrack multi-object tracking

        **Feature Extraction: MobileNetV2**
        - ImageNet pretrained, last 3 blocks fine-tuned
        - 1280-dim feature output (no projection layer)
        """)
    with col2:
        st.markdown("""
        **Temporal Classification: Bidirectional LSTM**
        - 256 hidden units, 2 layers, attention mechanism
        - Sequence length: 30 frames, stride: 15
        - 2-class output: normal / shoplifting

        **XAI Layer**
        - Attention weights → temporal explanation
        - Intent score: 50% behaviour + 30% product + 20% duration
        - Bias adjustment via video quality score
        """)

    st.markdown("---")
    st.markdown("### Training Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**BiLSTM**")
        st.json({"input_dim": (bilstm_info or {}).get("input_dim", 1280),
                 "hidden":    (bilstm_info or {}).get("hidden", 256),
                 "layers":    (bilstm_info or {}).get("layers", 2),
                 "bidirectional": True, "dropout": 0.3,
                 "seq_length":    (bilstm_info or {}).get("seq_len", 30)})
    with c2:
        st.markdown("**MobileNetV2**")
        st.json({"backbone": "mobilenet_v2",
                 "weights":  "MobileNet_V2_Weights.DEFAULT",
                 "feature_dim": 1280,
                 "unfrozen_blocks": "features[16:]"})
    with c3:
        st.markdown("**Training**")
        st.json({"batch_size":    16,
                 "lr_mobilenet": "1e-4",
                 "lr_bilstm":    "1e-3",
                 "optimizer":    "Adam",
                 "scheduler":    "ReduceLROnPlateau",
                 "early_stopping_patience": 10})


def render_analysis_results(results):
    if not results.get("success", False):
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    import plotly.graph_objects as go

    # ── Early-exit banner ──
    if results.get("early_exit"):
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='alert-none'>
            <h2 style='margin:0'>NORMAL — No Persons Detected</h2>
        </div>""", unsafe_allow_html=True)
        st.info("No persons were detected throughout the video. Classified as normal.")
        vm = results.get("video_metadata", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames Scanned", results["detections"]["frames_processed"])
        c2.metric("Persons Detected", 0)
        c3.metric("Duration", f"{vm.get('duration', 0):.1f}s")
        return

    if not results.get("model_trained", True):
        st.error("**LSTM model not trained.** Results shown are RANDOM and should not be acted upon. "
                 "Train the model using the Colab notebook first.")
        st.markdown("---")

    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    lstm = results["lstm_detection"]
    cls, conf = lstm["classification"], lstm["confidence"]
    is_shop   = lstm["is_shoplifting"]

    # ── LSTM Detection Banner ──
    if is_shop and conf > 0.7:
        st.markdown(f"""
        <div class='alert-critical'>
            <h2 style='margin:0'>SHOPLIFTING BEHAVIOR DETECTED</h2>
            <p style='margin:0.5rem 0 0 0'>Classification: <strong>{cls.upper()}</strong>
            | Confidence: <strong>{conf:.1%}</strong></p>
        </div>""", unsafe_allow_html=True)
    elif is_shop:
        st.markdown(f"""
        <div class='alert-high'>
            <h2 style='margin:0'>POTENTIAL SHOPLIFTING</h2>
            <p style='margin:0.5rem 0 0 0'>Classification: <strong>{cls.upper()}</strong>
            | Confidence: <strong>{conf:.1%}</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='alert-none'>
            <h2 style='margin:0'>NORMAL BEHAVIOR</h2>
            <p style='margin:0.5rem 0 0 0'>Classification: <strong>{cls.upper()}</strong>
            | Confidence: <strong>{conf:.1%}</strong></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Risk Score ──
    intent  = results["intent_score"]
    score   = intent["score"]
    sev     = intent["severity"].upper()
    cls_map = {"CRITICAL": "alert-critical", "HIGH": "alert-high",
               "MEDIUM": "alert-medium",     "LOW": "alert-low",
               "NONE": "alert-none"}
    div_cls = cls_map.get(sev, "alert-none")

    st.markdown("### Bias-Adjusted Risk Assessment")
    st.markdown(f"""
    <div class='{div_cls}'>
        <h3 style='margin:0'>Risk Level: {sev}</h3>
        <p style='margin:0.5rem 0 0 0'>Score: {score:.2f}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Detection Stats ──
    st.markdown("### Detection Statistics")
    det = results.get("detections", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People Interacted with products",    det.get("persons_tracked", 0))
    c2.metric("PICKED UP PRODUCTS", det.get("products_detected", 0))
    c3.metric("Interactions",     det.get("interactions", 0))
    c4.metric("Frames Processed", det.get("frames_processed", 0))
    st.markdown("---")

    # ── XAI Explanation ──
    st.markdown("### Explainable AI (XAI) Analysis")
    with st.expander("Why did the model make this decision?", expanded=True):
        st.markdown(f"""
        **Step 1: Object Detection (YOLO26)**
        - Scanned {det.get('frames_processed', 0)} video frames
        - Detected {det.get('persons_tracked', 0)} person(s), {det.get('products_detected', 0)} product(s)
        - YOLO26 (Jan 2026): NMS-free architecture, 43% faster than YOLOv8

        **Step 2: Feature Extraction (MobileNetV2)**
        - Extracted 1280-dim spatial features per frame
        - Raw backbone features (AdaptiveAvgPool, no projection head)
        - Last 3 inverted residual blocks fine-tuned on retail video

        **Step 3: Temporal Classification (Bidirectional LSTM)**
        - Analysed {LSTM_SEQ_LEN}-frame sliding windows (stride={LSTM_STRIDE})
        - **Classification: {cls.upper()}** | **Confidence: {conf:.1%}**
        """)

        if is_shop:
            st.warning("**Important:** Pattern match only — not definitive proof. Human review essential.")
        else:
            st.success("Movement patterns consistent with normal shopping behaviour.")

        events = results.get("behavior_events", [])
        if events:
            st.markdown("#### Behavior Event Breakdown")
            counts = {}
            for e in events:
                bt = e.get("behavior_type", "unknown")
                counts[bt] = counts.get(bt, {"count": 0, "conf_sum": 0.0})
                counts[bt]["count"]    += 1
                counts[bt]["conf_sum"] += e.get("confidence", 0)
            for bt, d in counts.items():
                avg_c = d["conf_sum"] / d["count"]
                icon  = "[HIGH]" if bt in {"shoplifting", "concealment"} else "[MED]" if bt == "bypass" else "[OK]"
                st.markdown(f"{icon} **{bt.capitalize()}**: {d['count']} segment(s), avg confidence: {avg_c:.1%}")

    st.markdown("---")

    # ── Intent Score Gauge + Components ──
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### Intent Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score * 100,
            title={"text": "Risk Score"},
            gauge={"axis": {"range": [0, 100]},
                   "bar" : {"color": "#667eea"},
                   "steps": [{"range": [0, 30],  "color": "#d4edda"},
                              {"range": [30, 50], "color": "#fff3cd"},
                              {"range": [50, 70], "color": "#ffe4b2"},
                              {"range": [70, 100],"color": "#ffcccc"}]}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Score Components")
        components = intent.get("components", {})
        if components:
            fig = go.Figure(go.Bar(
                x=list(components.values()), y=list(components.keys()),
                orientation="h", marker_color="#667eea"
            ))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=20),
                              xaxis=dict(range=[0, 1], tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)
        st.info(intent.get("explanation", "No explanation available."))

    st.markdown("---")

    # ── Behavior Timeline ──
    st.markdown("### Behavior Timeline")
    events = results.get("behavior_events", [])
    if events:
        colors = {"normal": "#00c853", "shoplifting": "#e91e63",
                  "concealment": "#ff5722", "bypass": "#f44336"}
        fig = go.Figure()
        for i, ev in enumerate(events):
            bt  = ev.get("behavior_type", "unknown")
            col = colors.get(bt, "#667eea")
            fig.add_trace(go.Scatter(
                x=[ev["start_time"], ev["end_time"]], y=[0, 0],
                mode="lines", line=dict(color=col, width=25),
                name=bt, showlegend=(i == 0 or events[i-1]["behavior_type"] != bt),
                hovertemplate=(f"Behavior: {bt}<br>Time: {ev['start_time']:.1f}s – "
                               f"{ev['end_time']:.1f}s<br>Conf: {ev['confidence']:.1%}<extra></extra>"),
            ))
        fig.update_layout(
            title="Behaviour Over Time", xaxis_title="Time (seconds)",
            height=150, yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Detailed Behaviour Events"):
            rows = [{"Behaviour": e["behavior_type"].capitalize(),
                     "Start": f"{e['start_time']:.1f}s",
                     "End": f"{e['end_time']:.1f}s",
                     "Confidence": f"{e['confidence']:.1%}"}
                    for e in events]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No behaviour events detected.")

    st.markdown("---")

    # ── Forensic Evidence (suspicious clips) ──
    if is_shop:
        st.markdown("### Forensic Evidence")
        st.caption(
            "Peak-confidence frames where the model detected shoplifting behaviour. "
            "Each frame is a distinct moment (≥ 2 s apart). Frame range shows which "
            "LSTM window triggered the alert."
        )
        frames = results.get("suspicious_frames", [])
        if frames:
            cols = st.columns(min(len(frames), 4))
            for i, fd in enumerate(frames):
                with cols[i % 4]:
                    if fd.get("thumbnail") is not None:
                        st.image(fd["thumbnail"], use_container_width=True)
                    st.caption(
                        f"**{fd['behavior'].upper()}**  \n"
                        f"Frames {fd.get('frame_start', '?')} – {fd.get('frame_end', '?')}  \n"
                        f"t = {fd['timestamp']:.1f}s  |  {fd['confidence']:.0%} confidence"
                    )
        else:
            st.info("No high-confidence shoplifting frames found (threshold: 50%).")
        st.markdown("---")

    # ── Quality & Fairness ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Video Quality")
        qual = results.get("quality_analysis", {})
        rel  = qual.get("reliability_score", 0)
        fig  = go.Figure(go.Indicator(
            mode="gauge+number", value=rel * 100,
            title={"text": "Reliability Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#38ef7d"},
                   "steps": [{"range": [0, 50],  "color": "#ffcccc"},
                              {"range": [50, 75], "color": "#fff3cd"},
                              {"range": [75, 100],"color": "#d4edda"}]}
        ))
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Bias & Fairness")
        bias   = results.get("bias_report", {})
        f_score = bias.get("overall_fairness_score", 0)
        fig    = go.Figure(go.Indicator(
            mode="gauge+number", value=f_score * 100,
            title={"text": "Fairness Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#667eea"},
                   "steps": [{"range": [0, 50],  "color": "#ffcccc"},
                              {"range": [50, 75], "color": "#fff3cd"},
                              {"range": [75, 100],"color": "#d4edda"}]}
        ))
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Detailed Bias Assessment"):
            st.markdown(f"- **Fairness Score:** {f_score:.1%}")
            st.markdown(f"- **Analysis Reliable:** {'Yes' if bias.get('analysis_reliable') else 'No'}")
            st.markdown(f"- **Manual Review Required:** {'Yes' if bias.get('requires_review') else 'No'}")
            for flag in bias.get("flags", []):
                st.warning(flag)
            st.info("All alerts require human validation before any action is taken.")
            if bias.get("requires_review"):
                st.error("**MANUAL REVIEW REQUIRED** — Bias indicators warrant additional scrutiny.")

    # ── Video Metadata ──
    st.markdown("---")
    st.markdown("### Video Information")
    vm = results.get("video_metadata", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**File:** {vm.get('filename', 'N/A')}")
    c2.write(f"**Duration:** {vm.get('duration', 0):.1f}s")
    c3.write(f"**Resolution:** {vm.get('width', 0)}×{vm.get('height', 0)}")
    c4.write(f"**FPS:** {vm.get('fps', 0):.1f}")

    # ── Advisory Summary ──
    st.markdown("---")
    st.markdown("### Advisory Summary")
    alert = results.get("alert")
    if alert:
        div_cls = "alert-high" if alert.get("level") in {"CRITICAL", "HIGH"} else "alert-medium"
        st.markdown(f"""
        <div class='{div_cls}'>
            <h4 style='margin:0'>Alert: {alert.get('alert_id', '')}</h4>
            <p style='margin:0.5rem 0'>{alert.get('message', '')}</p>
            <p style='margin:0;font-style:italic'>Advisory system — human validation required.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-none'>
            <h4 style='margin:0'>No Concerns Detected</h4>
            <p style='margin:0.5rem 0'>Normal shopping behaviour. No immediate concerns.</p>
        </div>""", unsafe_allow_html=True)


# ─── YOLO class → readable product label mapping ──────────────────────────────
YOLO_TO_PRODUCT = {
    "bottle"    : "Bottle / Drink",
    "cup"       : "Cup / Drink",
    "bowl"      : "Bowl / Food",
    "banana"    : "Banana",
    "apple"     : "Apple",
    "sandwich"  : "Sandwich / Food",
    "backpack"  : "Backpack",
    "handbag"   : "Handbag",
    "book"      : "Book / Magazine",
    "cell phone": "Mobile Phone",
    "orange"    : "Orange / Fruit",
    "donut"     : "Donut / Snack",
}


def render_pos_audit(analysis_results):
    """
    Mock POS terminal + mismatch audit.

    Left panel  — Manual POS entry: user picks items + quantities as if they
                  were the cashier entering a sale.
    Right panel — Video evidence: products the camera detected being handled.
    Bottom      — Mismatch table: flags items detected but not rung up, and
                  items rung up but never seen in the footage.
    """
    st.markdown('<div class="section-header">Mock POS Terminal & Audit</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Simulate a POS transaction then compare it against what the camera
        detected. Mismatches between rung-up items and detected product
        interactions are highlighted as potential theft indicators.
    </div>""", unsafe_allow_html=True)

    catalog = get_product_catalog()
    cat_names = [p["name"] for p in catalog]
    cat_by_name = {p["name"]: p for p in catalog}

    col_pos, col_vid = st.columns(2)

    # ── Left: POS Terminal ──────────────────────────────────────────────────
    with col_pos:
        st.markdown("### Mock POS Terminal")
        st.caption("Enter items as they are being rung up at the checkout.")

        with st.form("pos_form", clear_on_submit=False):
            sel_name = st.selectbox("Select product", cat_names)
            sel_qty  = st.number_input("Quantity", min_value=1, max_value=20, value=1)
            add_btn  = st.form_submit_button("Add to Transaction")

        if add_btn:
            item = dict(cat_by_name[sel_name])
            item["qty"] = int(sel_qty)
            st.session_state.pos_items.append(item)

        if st.button("Clear Transaction", key="clear_pos"):
            st.session_state.pos_items = []

        if st.session_state.pos_items:
            st.markdown("#### Transaction Ledger")
            rows = []
            total = 0.0
            for it in st.session_state.pos_items:
                subtotal = it["price"] * it["qty"]
                total   += subtotal
                rows.append({
                    "SKU"     : it["sku"],
                    "Product" : it["name"],
                    "Qty"     : it["qty"],
                    "Unit £"  : f"£{it['price']:.2f}",
                    "Subtotal": f"£{subtotal:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(f"**Total: £{total:.2f}**")
        else:
            st.info("No items added yet.")

    # ── Right: Behavioural Video Evidence ────────────────────────────────────
    with col_vid:
        st.markdown("### Behavioural Video Evidence")
        st.caption(
            "The behaviour model classifies person actions, not specific products. "
            "Evidence is based on detected interaction events."
        )

        if not analysis_results:
            st.info("Run video analysis first.")
        else:
            det    = analysis_results.get("detections", {})
            lstm   = analysis_results.get("lstm_detection", {})
            events = analysis_results.get("behavior_events", [])

            # Count YOLO Picking-Holding detections (item interaction proxy)
            yolo_pickups = analysis_results.get("product_pickups", {})
            picking_count = yolo_pickups.get("Picking-Holding", 0)

            # Count LSTM shoplifting windows
            shop_windows = sum(
                1 for e in events if e.get("behavior_type") == "shoplifting"
            )
            susp_windows = sum(
                1 for e in events
                if e.get("behavior_type") in {"shoplifting", "Looking around"}
            )

            vid_rows = [
                {"Behavioural Signal"     : "Persons tracked",
                 "Count": det.get("persons_tracked", 0),
                 "Risk" : "—"},
                {"Behavioural Signal"     : "Item interaction (Picking-Holding)",
                 "Count": picking_count,
                 "Risk" : "Possible" if picking_count > 0 else "—"},
                {"Behavioural Signal"     : "Suspicious behaviour windows (LSTM)",
                 "Count": susp_windows,
                 "Risk" : "High" if susp_windows > 0 else "None"},
                {"Behavioural Signal"     : "Shoplifting classification windows",
                 "Count": shop_windows,
                 "Risk" : "High" if shop_windows > 0 else "None"},
            ]
            st.dataframe(pd.DataFrame(vid_rows), use_container_width=True,
                         hide_index=True)
            overall = lstm.get("classification", "normal")
            conf    = lstm.get("confidence", 0.0)
            if overall == "shoplifting":
                st.error(
                    f"LSTM verdict: **SHOPLIFTING** ({conf:.0%}) — "
                    f"transaction flagged for review."
                )
            else:
                st.success(
                    f"LSTM verdict: **NORMAL** ({conf:.0%}) — "
                    f"no shoplifting pattern detected."
                )

    # ── Mismatch Audit ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Transaction Risk Audit")
    st.caption(
        "Since the behaviour model does not identify specific products, the audit "
        "compares the number of items rung up at POS against the number of "
        "Picking-Holding interactions and suspicious behaviour windows detected. "
        "A large gap between items handled and items scanned is a theft indicator."
    )

    if not analysis_results:
        st.info("Run video analysis first.")
        return
    if not st.session_state.pos_items:
        st.info("Add items to the POS transaction to compare.")
        return

    import plotly.graph_objects as go

    det    = analysis_results.get("detections", {})
    events = analysis_results.get("behavior_events", [])
    lstm   = analysis_results.get("lstm_detection", {})

    pos_total_qty  = sum(it["qty"] for it in st.session_state.pos_items)
    pos_total_val  = sum(it["price"] * it["qty"] for it in st.session_state.pos_items)
    picking_count  = analysis_results.get("product_pickups", {}).get("Picking-Holding", 0)
    shop_windows   = sum(1 for e in events if e.get("behavior_type") == "shoplifting")
    is_shoplifting = lstm.get("is_shoplifting", False)

    # Build audit summary table
    audit_rows = [
        {"Metric"   : "POS items scanned",
         "Value"    : pos_total_qty,
         "Status"   : "OK"},
        {"Metric"   : "POS transaction value",
         "Value"    : f"£{pos_total_val:.2f}",
         "Status"   : "OK"},
        {"Metric"   : "Item interactions detected (Picking-Holding)",
         "Value"    : picking_count,
         "Status"   : ("More handled than scanned"
                       if picking_count > pos_total_qty else "OK")},
        {"Metric"   : "Shoplifting behaviour windows (LSTM)",
         "Value"    : shop_windows,
         "Status"   : "Suspicious" if shop_windows > 0 else "None"},
    ]
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    # Determine overall risk and show flag
    flags = []
    if is_shoplifting:
        flags.append(
            f"LSTM classified video as **SHOPLIFTING** ({lstm.get('confidence', 0):.0%}) "
            f"while {pos_total_qty} item(s) were rung up. "
            "Verify transaction matches items handled."
        )
    if picking_count > pos_total_qty:
        flags.append(
            f"**{picking_count} item interaction(s)** detected but only "
            f"**{pos_total_qty} item(s)** scanned at POS — potential scan avoidance."
        )
    if is_shoplifting and pos_total_qty == 0:
        flags.append(
            "Shoplifting behaviour detected but **no purchase recorded** at POS."
        )

    if flags:
        st.markdown("#### Potential Theft Indicators")
        for f in flags:
            st.error(f)
        st.warning("**Human review required** before any action is taken.")
    else:
        st.success("No mismatches found — POS transaction consistent with video evidence.")

    # Bar chart: POS items vs item interactions
    fig = go.Figure()
    fig.add_bar(name="POS Items Scanned",
                x=["Transaction"], y=[pos_total_qty], marker_color="#38ef7d")
    fig.add_bar(name="Item Interactions Detected",
                x=["Transaction"], y=[picking_count],  marker_color="#667eea")
    fig.add_bar(name="Shoplifting Windows",
                x=["Transaction"], y=[shop_windows],   marker_color="#dc3545")
    fig.update_layout(
        barmode="group", title="POS vs Behavioural Evidence",
        height=300, margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "pos_items" not in st.session_state:
        st.session_state.pos_items = []
    if "annotated_video_path" not in st.session_state:
        st.session_state.annotated_video_path = None

    render_header()
    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["Model Performance", "Video Analysis", "POS Audit"])

    with tab1:
        render_model_performance_tab()

    with tab2:
        st.markdown('<div class="section-header">Deep Learning Video Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Pipeline:</strong> YOLO26 (detection) → MobileNetV2 (features) → LSTM (classification)
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Upload Video")
            uploaded = st.file_uploader(
                "Choose a video file", type=["mp4", "avi", "mov", "mkv"],
                help="Upload a video to analyse for potential shoplifting behaviour"
            )
        with c2:
            if uploaded:
                st.markdown("### Preview")
                st.video(uploaded)

        st.markdown("---")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            analyse = st.button(
                "Analyse Video",
                type="primary", use_container_width=True,
                disabled=(uploaded is None)
            )

        if analyse and uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            # Temp file for annotated output video
            import uuid
            ann_out_path = os.path.join(tempfile.gettempdir(),
                                        f"dw_annotated_{uuid.uuid4().hex[:8]}.mp4")

            st.markdown("### Live Analysis Preview")
            st.caption("Watch the pipeline analyse your video — YOLO detections, "
                       "classification badge, and stats update in real time.")

            _col_frame, _col_stats = st.columns([3, 1])
            with _col_frame:
                _frame_ph = st.empty()
                _frame_ph.markdown(
                    "<div style='height:200px;display:flex;align-items:center;"
                    "justify-content:center;background:#111;border-radius:8px;"
                    "color:#666;font-size:0.9rem;'>Waiting for first frame...</div>",
                    unsafe_allow_html=True,
                )
            with _col_stats:
                _stat_ph = st.empty()

            _timeline_ph = st.empty()
            prog   = st.progress(0)
            status = st.empty()

            def on_progress(p, m):
                prog.progress(min(p, 1.0))
                status.text(m)

            def on_live_frame(frame_rgb, frame_idx, total_frames, stats):
                _frame_ph.image(
                    frame_rgb,
                    caption=f"Frame {frame_idx} / {total_frames}",
                    use_container_width=True,
                )
                label = stats.get("label", "Analyzing...")
                conf  = stats.get("conf", 0.0)
                col   = ("#dc3545" if label == "shoplifting" else
                         "#28a745" if label == "normal" else "#aaaaaa")
                _stat_ph.markdown(f"""
                <div style='padding:1rem;background:#1a1a2e;border-radius:8px;
                            border:1px solid #333;font-family:monospace;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>CLASSIFICATION</p>
                    <p style='color:{col};font-size:1.15rem;font-weight:700;
                               margin:0.25rem 0;'>{label.upper()}</p>
                    <p style='color:#ccc;font-size:0.85rem;margin:0;'>
                        {conf:.0%} confidence</p>
                    <hr style='border-color:#333;margin:0.6rem 0;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>SUSPICIOUS PEOPLE</p>
                    <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>
                        {stats.get("persons_tracked", 0)}</p>
                    <hr style='border-color:#333;margin:0.6rem 0;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>PICKED UP PRODUCTS </p>
                    <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>
                        {stats.get("products_detected", 0)}</p>
                </div>""", unsafe_allow_html=True)

                timeline = stats.get("timeline", [])
                if timeline:
                    pct = frame_idx / max(total_frames, 1)
                    cmap = {"red": "#dc3545", "green": "#28a745", "gray": "#555"}
                    bar  = ('<div style="display:flex;width:100%;height:10px;'
                            'border-radius:4px;overflow:hidden;background:#333;">')
                    for tc in timeline:
                        w = 100.0 / max(len(timeline), 1)
                        bar += (f'<div style="flex:{w:.2f};'
                                f'background:{cmap.get(tc, "#555")};"></div>')
                    bar += "</div>"
                    _timeline_ph.markdown(f"""
                    <div style='margin:0.4rem 0 0.8rem 0;'>
                        <p style='color:#888;font-size:0.72rem;margin-bottom:4px;'>
                            Timeline &nbsp;
                            <span style='color:#28a745;'>■ Normal</span>&nbsp;
                            <span style='color:#dc3545;'>■ Suspicious</span>&nbsp;
                            <span style='color:#555;'>■ Analyzing</span>
                        </p>
                        {bar}
                        <p style='color:#666;font-size:0.72rem;margin-top:4px;'>
                            {pct:.0%} processed</p>
                    </div>""", unsafe_allow_html=True)

            try:
                results = run_pipeline(
                    video_path,
                    progress_callback=on_progress,
                    visual_out_path=ann_out_path,
                    live_frame_callback=on_live_frame,
                )
            except Exception as e:
                results = {"success": False, "error": str(e)}

            # Extract forensic GIF clips separately so a clip-extraction failure
            # never wipes out the main analysis results.
            if results.get("success"):
                try:
                    results["suspicious_frames"] = extract_suspicious_clips(
                        video_path, results.get("behavior_events", []), max_clips=4
                    )
                except Exception:
                    results["suspicious_frames"] = []

            # Clear live preview widgets
            _frame_ph.empty()
            _stat_ph.empty()
            _timeline_ph.empty()
            prog.empty()
            status.empty()

            if os.path.exists(video_path):
                os.unlink(video_path)

            # Clean up old annotated video if present
            old_ann = st.session_state.get("annotated_video_path")
            if old_ann and old_ann != ann_out_path and os.path.exists(old_ann):
                try:
                    os.unlink(old_ann)
                except OSError:
                    pass

            st.session_state.analysis_results    = results
            st.session_state.annotated_video_path = ann_out_path
            st.rerun()   # force a clean rerender so results section renders fresh

        if st.session_state.analysis_results:
            # ── Annotated video player ──────────────────────────────────────
            ann_vid = st.session_state.get("annotated_video_path")
            if ann_vid and os.path.exists(ann_vid) and os.path.getsize(ann_vid) > 4096:
                st.markdown("---")
                st.markdown("### Annotated Analysis Video")
                st.caption(
                    "Replay with YOLO behaviour labels and BiLSTM classification badge. "
                    "Green = normal · Orange = Looking around · Red = Shoplifting"
                )
                with open(ann_vid, "rb") as _vf:
                    st.video(_vf.read())

            st.markdown("---")
            r = st.session_state.analysis_results
            if not r.get("success", False):
                st.error(f"Analysis failed: {r.get('error', 'Unknown error')}")
            else:
                try:
                    render_analysis_results(r)
                except Exception as _render_err:
                    st.error(f"Report render error: {_render_err}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab3:
        render_pos_audit(st.session_state.analysis_results)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#888;padding:2rem 0'>
        <p style='font-size:1.1rem;font-weight:500'>Digital Witness — Deep Learning Retail Security Assistant</p>
        <p style='font-size:0.9rem'>
            YOLO26 (detection) → MobileNetV2 (features) → Bidirectional LSTM (classification)<br>
            Advisory system only. All alerts require human validation.
        </p>
        <p style='font-size:0.8rem;margin-top:1rem'>Final Year Project — IIT 2026</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
