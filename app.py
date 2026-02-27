"""
Digital Witness — Streamlit Web Interface

Self-contained: no src/ imports. Loads models from models/ directory.
Pipeline: YOLO26 (detection) → MobileNetV2 (features) → LSTM (classification)
"""
import streamlit as st
import tempfile
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Digital Witness - Retail Security Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"

YOLO26_PATH     = MODELS_DIR / "yolo26_retail.pt"
YOLO26_BASE     = MODELS_DIR / "yolo26n.pt"          # fallback if not fine-tuned
MOBILENET_PATH  = MODELS_DIR / "mobilenet_extractor.pt"
LSTM_PATH       = MODELS_DIR / "lstm_classifier.pt"
LSTM_INFO_PATH  = MODELS_DIR / "lstm_classifier_info.json"

BEHAVIOR_CLASSES = ["normal", "shoplifting"]
YOLO_CONF        = 0.3
YOLO_IOU         = 0.45
MOBILENET_DIM    = 512
LSTM_SEQ_LEN     = 30
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
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 10px 10px 0 0; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Inline Model Definitions ─────────────────────────────────────────────────

def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_lstm_model(input_dim, hidden_dim, num_layers, num_classes, dropout, bidirectional):
    """Build LSTM model with attention (same architecture as notebook Cell 5)."""
    import torch.nn as nn
    import torch

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            attn_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention = nn.Sequential(
                nn.Linear(attn_dim, attn_dim // 2), nn.Tanh(),
                nn.Linear(attn_dim // 2, 1)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(attn_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attn = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn * lstm_out, dim=1)
            return self.classifier(context), attn.squeeze(-1)

    return _Model()


def _build_mobilenet_extractor(feature_dim=512):
    """
    Build MobileNetV2 feature extractor with fixed bugs:
    - weights=MobileNet_V2_Weights.DEFAULT (not deprecated pretrained=True)
    - model.features used directly (not children()[:-1])
    - AdaptiveAvgPool2d before Linear (fixes 62720 ≠ 1280 size mismatch)
    """
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    class _Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.features   = mobilenet.features
            self.pool       = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1280, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return self.projection(x)

    return _Extractor()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_model_exists():
    return LSTM_PATH.exists()


def load_model_metrics():
    if LSTM_INFO_PATH.exists():
        with open(LSTM_INFO_PATH, "r") as f:
            return json.load(f)
    return None


def get_active_yolo_path():
    """Return fine-tuned YOLO26 if available, else base yolo26n.pt."""
    if YOLO26_PATH.exists():
        return str(YOLO26_PATH)
    if YOLO26_BASE.exists():
        return str(YOLO26_BASE)
    return str(MODELS_DIR / "yolo26n.pt")  # ultralytics will attempt to download


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

def run_pipeline(video_path, progress_callback=None):
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

    # ── Load YOLO26 ──
    yolo_path  = get_active_yolo_path()
    yolo_model = YOLO(yolo_path)

    # ── Load MobileNetV2 feature extractor ──
    feat_model = _build_mobilenet_extractor(MOBILENET_DIM)
    if MOBILENET_PATH.exists():
        feat_model.load_state_dict(
            torch.load(MOBILENET_PATH, map_location=dev, weights_only=True)
        )
    feat_model = feat_model.to(dev)
    feat_model.eval()

    # ── Load LSTM classifier ──
    lstm_classes = BEHAVIOR_CLASSES
    lstm_model   = None
    if LSTM_PATH.exists():
        ckpt = torch.load(LSTM_PATH, map_location=dev, weights_only=False)
        cfg  = ckpt.get("config", {})
        lstm_classes = cfg.get("classes", BEHAVIOR_CLASSES)
        lstm_model = _build_lstm_model(
            input_dim=cfg.get("input_dim", MOBILENET_DIM),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 2),
            num_classes=len(lstm_classes),
            dropout=0.3,
            bidirectional=True,
        ).to(dev)
        lstm_model.load_state_dict(ckpt["state_dict"])
        lstm_model.eval()
    else:
        # Untrained model — results will be random, flagged in UI
        lstm_model = _build_lstm_model(
            MOBILENET_DIM, 256, 2, len(lstm_classes), 0.3, True
        ).to(dev)
        lstm_model.eval()

    # ── Image preprocessing transform ──
    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    PRODUCT_CLASSES = {
        "bottle", "cup", "bowl", "banana", "apple", "sandwich",
        "backpack", "handbag", "book", "cell phone", "orange", "donut",
    }

    # ── Video processing ──
    _cb(0.05, "Opening video...")
    cap         = cv2.VideoCapture(video_path)
    fps_val     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_feats   = []
    persons_set = set()
    products_set = set()
    frame_num   = 0
    frame_step  = 3   # sample every 3rd frame for speed on CPU

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % frame_step != 0:
            continue

        if progress_callback and frame_num % 90 == 0:
            pct = min(0.05 + (frame_num / max(total_f, 1)) * 0.65, 0.70)
            _cb(pct, f"Processing frame {frame_num}/{total_f}...")

        # YOLO26 detection + ByteTrack
        results = yolo_model.track(
            frame, conf=YOLO_CONF, iou=YOLO_IOU, persist=True, verbose=False
        )[0]
        if results.boxes is not None:
            ids = (results.boxes.id.int().tolist()
                   if results.boxes.id is not None else [-1] * len(results.boxes))
            for cls_id, tid in zip(results.boxes.cls, ids):
                cls_name = yolo_model.names[int(cls_id)]
                if cls_name == "person":
                    persons_set.add(tid)
                elif cls_name in PRODUCT_CLASSES:
                    products_set.add(cls_name)

        # MobileNetV2 feature extraction
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x    = xform(rgb).unsqueeze(0).to(dev)
        with torch.no_grad():
            feat = feat_model(x).cpu().numpy().flatten()
        all_feats.append(feat)

    cap.release()
    duration = frame_num / fps_val

    _cb(0.72, "Running LSTM temporal classification...")

    # ── Sliding-window LSTM classification ──
    feats_arr   = np.array(all_feats) if all_feats else np.zeros((1, MOBILENET_DIM))
    predictions = []

    if len(all_feats) < LSTM_SEQ_LEN:
        pad = np.zeros((LSTM_SEQ_LEN - len(all_feats), MOBILENET_DIM))
        seq = np.vstack([feats_arr, pad])
        x_t = torch.FloatTensor(seq[np.newaxis]).to(dev)
        with torch.no_grad():
            logits, _ = lstm_model(x_t)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
        scale = len(all_feats) / LSTM_SEQ_LEN
        predictions.append({
            "start"        : 0.0,
            "end"          : duration,
            "behavior_type": lstm_classes[int(np.argmax(probs))],
            "confidence"   : float(np.max(probs)) * scale,
            "probabilities": {c: float(p) for c, p in zip(lstm_classes, probs)},
        })
    else:
        for start in range(0, len(all_feats) - LSTM_SEQ_LEN + 1, LSTM_STRIDE):
            seq = feats_arr[start:start + LSTM_SEQ_LEN]
            x_t = torch.FloatTensor(seq[np.newaxis]).to(dev)
            with torch.no_grad():
                logits, _ = lstm_model(x_t)
                probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predictions.append({
                "start"        : start / fps_val,
                "end"          : (start + LSTM_SEQ_LEN) / fps_val,
                "behavior_type": lstm_classes[int(np.argmax(probs))],
                "confidence"   : float(np.max(probs)),
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
    conceal_e = [p for p in predictions
                 if p["behavior_type"] in {"shoplifting", "concealment"}]
    bypass_e  = [p for p in predictions if p["behavior_type"] == "bypass"]
    susp_e    = [p for p in predictions
                 if p["behavior_type"] in {"shoplifting", "concealment", "bypass"}]

    s_conceal = (np.mean([e["confidence"] for e in conceal_e])
                 * min(1.0, len(conceal_e) / 3.0)) if conceal_e else 0.0
    s_bypass  = max((e["confidence"] for e in bypass_e), default=0.0)
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
        n_susp = len(conceal_e) + len(bypass_e)
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
            "persons_tracked"  : len(persons_set),
            "products_detected": len(products_set),
            "interactions"     : len(susp_e),
            "frames_processed" : len(all_feats),
        },
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
            "detection_rate"   : min(1.0, len(all_feats) / max(1, total_f // frame_step)),
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
    }


def extract_suspicious_frames(video_path, behavior_events, max_frames=4):
    """Extract key frames from the highest-confidence suspicious events."""
    import cv2
    suspicious = [
        e for e in behavior_events
        if e.get("behavior_type") in {"shoplifting", "concealment", "bypass"}
        and e.get("confidence", 0) > 0.5
    ]
    if not suspicious:
        return []
    suspicious = sorted(suspicious, key=lambda x: x.get("confidence", 0), reverse=True)[:max_frames]

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for ev in suspicious:
        mid_t = (ev.get("start_time", 0) + ev.get("end_time", 1)) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_t * fps_v))
        ret, frame = cap.read()
        if ret:
            frames.append({
                "image"     : cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                "timestamp" : mid_t,
                "behavior"  : ev["behavior_type"],
                "confidence": ev["confidence"],
                "start_time": ev.get("start_time", 0),
                "end_time"  : ev.get("end_time", 0),
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
        2. **MobileNetV2 Features** — Extracts 512-dim spatial features per frame
           (1280→512 projection, ImageNet pretrained).
        3. **Bidirectional LSTM + Attention** — Classifies temporal sequences of features
           into normal / shoplifting.

        **Important:** This is an *advisory system*. It does **not** determine guilt.
        All alerts require human validation.
        """)

        st.markdown("---")
        st.markdown("### System Status")

        if check_model_exists():
            st.success("LSTM Model: Trained ✓")
            metrics = load_model_metrics()
            if metrics:
                acc = metrics.get("final_val_acc",
                      metrics.get("metrics", {}).get("accuracy", 0))
                st.metric("Validation Accuracy", f"{acc:.1%}")
                if "metrics" in metrics:
                    m = metrics["metrics"]
                    st.metric("F1 Score", f"{m.get('f1_score', 0):.1%}")
        else:
            st.warning("LSTM Model: Not trained")
            st.info("Train in Colab with **DigitalWitness_Pipeline.ipynb**, then place `lstm_classifier.pt` in `models/`")

        yolo_status = "yolo26_retail.pt ✓" if YOLO26_PATH.exists() else \
                      "yolo26n.pt (base) ✓" if YOLO26_BASE.exists() else "Not found"
        mb_status = "mobilenet_extractor.pt ✓" if MOBILENET_PATH.exists() else "Not fine-tuned (using ImageNet weights)"

        st.markdown("---")
        st.markdown("### Model Files")
        st.code(f"YOLO26 : {yolo_status}\nMobileNetV2: {mb_status}", language=None)

        st.markdown("---")
        st.markdown("### Important Notice")
        st.info("This is an **advisory system**.\nAll alerts require human validation.\nThe system does NOT determine guilt.")
        st.markdown("---")
        st.markdown("##### Final Year Project — 2026")


def render_model_performance_tab():
    metrics = load_model_metrics()

    if not metrics:
        st.warning("No model metrics found. Train the model first using the Colab notebook.")
        st.markdown("---")
        st.markdown("### Training Instructions")
        st.markdown("""
        1. Open **DigitalWitness_Pipeline.ipynb** in Google Colab
        2. Upload `yolo26n.pt` to `Drive/DigitalWitness/`
        3. Add training videos to `Drive/DigitalWitness/dataset/videos/`
        4. Run all cells (Cell 3 → YOLO26, Cell 4 → MobileNetV2, Cell 5 → LSTM)
        5. Download `models/lstm_classifier.pt` and `models/lstm_classifier_info.json`
        6. Place them in `models/` directory here
        """)
        return

    st.markdown('<div class="section-header">Deep Learning Model Performance</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_str = metrics.get("training_date", "N/A")
        st.metric("Training Date", date_str[:10] if len(date_str) >= 10 else date_str)
    with col2:
        st.metric("Train Samples", f"{metrics.get('n_train_samples', 0):,}")
    with col3:
        st.metric("Val Samples",   f"{metrics.get('n_val_samples', 0):,}")

    st.markdown("---")
    st.markdown("### Classification Metrics")
    detailed = metrics.get("metrics", {})
    if detailed:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{detailed.get('accuracy', 0):.1%}")
        c2.metric("Precision", f"{detailed.get('precision', 0):.1%}")
        c3.metric("Recall",    f"{detailed.get('recall', 0):.1%}")
        c4.metric("F1 Score",  f"{detailed.get('f1_score', 0):.1%}")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Train Accuracy", f"{metrics.get('final_train_acc', 0):.1%}")
        c2.metric("Val Accuracy",   f"{metrics.get('final_val_acc', 0):.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
        **Detection: YOLO26** (Jan 2026)
        - NMS-free, 43% faster than YOLOv8
        - Fine-tuned on retail annotations
        - ByteTrack multi-object tracking

        **Feature Extraction: MobileNetV2**
        - ImageNet pretrained weights
        - Last 3 residual blocks fine-tuned
        - 1280 → 512 dim projection

        **Temporal Classification: Bidirectional LSTM**
        - 256 hidden units, 2 layers
        - Attention mechanism
        - 2-class: normal / shoplifting
        """)

    with col2:
        st.markdown("### Confusion Matrix")
        import plotly.graph_objects as go
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            classes = metrics.get("classes", ["normal", "shoplifting"])
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=classes, y=classes, colorscale="Blues",
                text=cm, texttemplate="%{text}", textfont={"size": 20},
            ))
            fig.update_layout(
                title="Predicted vs Actual",
                xaxis_title="Predicted", yaxis_title="True",
                height=300, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run notebook Cell 5 to generate the confusion matrix")

    per_class = metrics.get("per_class_metrics", {})
    if per_class:
        st.markdown("---")
        st.markdown("### Per-Class Metrics")
        classes = metrics.get("classes", ["normal", "shoplifting"])
        rows = [{"Class": c.capitalize(),
                 "Precision": f"{per_class.get('precision', {}).get(c, 0):.1%}",
                 "Recall"   : f"{per_class.get('recall', {}).get(c, 0):.1%}",
                 "F1 Score" : f"{per_class.get('f1', {}).get(c, 0):.1%}"}
                for c in classes]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Training Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**LSTM**")
        st.json({"input_dim": metrics.get("input_dim", 512),
                 "hidden": 256, "layers": 2,
                 "bidirectional": True, "dropout": 0.3,
                 "seq_length": metrics.get("sequence_length", 30)})
    with c2:
        st.markdown("**MobileNetV2**")
        st.json({"backbone": "mobilenet_v2",
                 "weights": "MobileNet_V2_Weights.DEFAULT",
                 "feature_dim": 512,
                 "unfrozen_blocks": "15-18"})
    with c3:
        st.markdown("**Training**")
        st.json({"epochs": metrics.get("epochs", 50),
                 "batch_size": 16,
                 "lr_mobilenet": "1e-4",
                 "lr_lstm": "1e-3",
                 "optimizer": "Adam"})


def render_analysis_results(results):
    if not results.get("success", False):
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    import plotly.graph_objects as go

    if not results.get("model_trained", True):
        st.error("**LSTM model not trained.** Results shown are RANDOM and should not be acted upon. "
                 "Train the model using the Colab notebook first.")
        st.markdown("---")

    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    lstm = results["lstm_detection"]
    cls, conf = lstm["classification"], lstm["confidence"]
    is_shop   = lstm["is_shoplifting"]

    # ── LSTM Detection Banner ──
    st.markdown("### LSTM Deep Learning Detection")
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
    c1.metric("Persons Tracked",  det.get("persons_tracked", 0))
    c2.metric("Products Detected",det.get("products_detected", 0))
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
        - Extracted 512-dim spatial features per frame
        - Using model.features with AdaptiveAvgPool + 1280→512 projection
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
                icon  = "🔴" if bt in {"shoplifting", "concealment"} else "🟡" if bt == "bypass" else "🟢"
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

    # ── Forensic Evidence (suspicious frames) ──
    if is_shop:
        st.markdown("### Forensic Evidence")
        frames = results.get("suspicious_frames", [])
        if frames:
            cols = st.columns(min(len(frames), 4))
            for i, fd in enumerate(frames):
                with cols[i % 4]:
                    st.image(fd["image"],
                             caption=(f"{fd['behavior'].upper()}\n"
                                      f"t={fd['timestamp']:.1f}s | {fd['confidence']:.0%}"),
                             use_container_width=True)
        else:
            st.info("No suspicious frames extracted (brief or low-confidence segments).")
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


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    render_header()
    render_sidebar()

    tab1, tab2 = st.tabs(["📊 Model Performance", "🎥 Video Analysis"])

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
                "Analyse Video (YOLO26 + MobileNetV2 + LSTM)",
                type="primary", use_container_width=True,
                disabled=(uploaded is None)
            )

        if analyse and uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            st.markdown("### Processing")
            prog = st.progress(0)
            status = st.empty()

            def on_progress(p, m):
                prog.progress(min(p, 1.0))
                status.text(m)

            with st.spinner("Running deep learning analysis..."):
                try:
                    results = run_pipeline(video_path, progress_callback=on_progress)
                    if results.get("success"):
                        results["suspicious_frames"] = extract_suspicious_frames(
                            video_path, results.get("behavior_events", []), max_frames=4
                        )
                except Exception as e:
                    results = {"success": False, "error": str(e)}

            if os.path.exists(video_path):
                os.unlink(video_path)

            st.session_state.analysis_results = results
            prog.empty()
            status.empty()

        if st.session_state.analysis_results:
            st.markdown("---")
            render_analysis_results(st.session_state.analysis_results)

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
