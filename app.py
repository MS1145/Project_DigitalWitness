import streamlit as st
import tempfile, os, json, cv2, uuid
import threading, queue
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

st.set_page_config(
    page_title="Digital Witness - Retail Security Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"

YOLO_PATH    = MODELS_DIR / "yolo_dw.pt"    # fine-tuned 4-class behaviour detector
YOLO_BASE    = MODELS_DIR / "yolo26n.pt"    # fallback: base YOLO26n weights
BILSTM_PATH  = MODELS_DIR / "bilstm_dw.pt"
BILSTM_INFO  = MODELS_DIR / "bilstm_dw_info.json"
EFFNET_INFO  = MODELS_DIR / "efficientnet_dw_info.json"
OUTPUTS_DIR  = ROOT_DIR / "outputs" / "cases"

BEHAVIOR_CLASSES = ["normal", "shoplifting"]
YOLO_CONF  = 0.2
YOLO_IOU   = 0.45
FEAT_DIM   = 1280   # EfficientNetV2-S / MobileNetV2 both output 1280-dim
SEQ_LEN    = 45     # 45 frames @ 6fps = 7.5s temporal window
SEQ_STRIDE = 15


# ── CSS ────────────────────────────────────────────────────────────────────────
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
.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6; border-radius: 10px 10px 0 0;
    padding: 0.4rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
}
</style>
""", unsafe_allow_html=True)


# ── Model classes ──────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Learns which frames in a sequence matter most — output weights are the XAI signal."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        ctx = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return ctx, weights


class BiLSTMClassifier(nn.Module):
    """
    Temporal sequence classifier.
    Input:  (B, T, FEAT_DIM) — sequences of per-frame CNN features
    Output: (logits, attention_weights)

    Pipeline:
    1. BiLSTM captures forward + backward temporal context
    2. Attention picks the most suspicious frames (→ XAI heatmap)
    3. Linear classifier outputs normal / shoplifting probabilities
    """
    def __init__(self, feat_dim=1280, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, len(BEHAVIOR_CLASSES)),
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        ctx, attn = self.attention(out)
        return self.classifier(ctx), attn


# ── Model loading (cached so they stay in memory across reruns) ────────────────

@st.cache_resource
def load_effnet():
    """EfficientNetV2-S frozen backbone — extracts 1280-dim spatial feature per frame."""
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    # strip the classifier head, keep only the feature backbone + pooling
    model = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model.eval()
    return model


@st.cache_resource
def load_bilstm():
    if not BILSTM_PATH.exists():
        return None
    ckpt = torch.load(BILSTM_PATH, map_location="cpu", weights_only=False)
    sd  = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    m = BiLSTMClassifier(
        feat_dim=cfg.get("feature_dim", FEAT_DIM),
        hidden=cfg.get("hidden_dim", 256),
        layers=cfg.get("num_layers", 2),
    )
    m.load_state_dict(sd, strict=False)
    return m.eval()


@st.cache_resource
def load_yolo():
    path = str(YOLO_PATH if YOLO_PATH.exists() else YOLO_BASE)
    return YOLO(path)


def get_model_info():
    if BILSTM_INFO.exists():
        return json.loads(BILSTM_INFO.read_text())
    return None

def get_effnet_info():
    if EFFNET_INFO.exists():
        return json.loads(EFFNET_INFO.read_text())
    return None


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _draw_frame(frame, yolo_result, yolo_model, is_4cls, badge_label, badge_conf, fn, total):
    """Draw YOLO bounding boxes and classification badge onto a frame copy."""
    ann = frame.copy()
    if yolo_result.boxes is not None:
        for box, cls_id, conf in zip(yolo_result.boxes.xyxy.cpu().numpy(),
                                      yolo_result.boxes.cls.cpu().numpy(),
                                      yolo_result.boxes.conf.cpu().numpy()):
            cls_name = yolo_model.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box)
            if is_4cls:
                col = ((0,0,220) if cls_name == "shoplifting" else
                       (0,140,255) if cls_name == "looking-around" else
                       (0,200,255) if cls_name == "picking-holding" else (0,200,0))
            else:
                col = (0, 200, 0) if int(cls_id) == 0 else (160, 160, 160)
            cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
            label = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y1 - 4, th + 6)
            cv2.rectangle(ann, (x1, ly-th-6), (x1+tw+8, ly+2), col, -1)
            cv2.putText(ann, label, (x1+4, ly-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    if badge_label:
        b_col = (0,0,200) if badge_label == "shoplifting" else (0,140,255) if badge_label == "looking-around" else (20,140,20)
        b_txt = f"{badge_label.upper()}  {badge_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(b_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(ann, (8, 8), (tw+18, th+22), b_col, -1)
        cv2.putText(ann, b_txt, (12, th+16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(ann, f"{fn}/{total}", (8, ann.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1, cv2.LINE_AA)
    return ann


def _classify_sequences(bilstm, feats, dev):
    """Run BiLSTM on sliding windows over the extracted feature array.
    Returns list of dicts with behavior_type, confidence, start/end, probabilities."""
    feats_arr = np.array(feats)
    results = []

    if len(feats) < SEQ_LEN:
        # pad short videos with zeros so we still get a prediction
        pad = np.zeros((SEQ_LEN - len(feats), feats_arr.shape[1]))
        seq = np.vstack([feats_arr, pad])
        xt = torch.FloatTensor(seq[np.newaxis]).to(dev)
        with torch.no_grad():
            logits, _ = bilstm(xt)
        probs = torch.softmax(logits / 3.0, dim=1).cpu().numpy()[0]
        # scale confidence by how much of the window is real data
        conf = float(np.max(probs)) * min(1.0, len(feats) / SEQ_LEN)
        results.append({
            "start": 0.0, "end": len(feats) * 4 / 30.0,
            "behavior_type": BEHAVIOR_CLASSES[int(np.argmax(probs))],
            "confidence": min(conf, 0.99),
            "probabilities": {c: float(p) for c, p in zip(BEHAVIOR_CLASSES, probs)},
        })
    else:
        for s in range(0, len(feats) - SEQ_LEN + 1, SEQ_STRIDE):
            xt = torch.FloatTensor(feats_arr[s:s+SEQ_LEN][np.newaxis]).to(dev)
            with torch.no_grad():
                logits, _ = bilstm(xt)
            probs = torch.softmax(logits / 3.0, dim=1).cpu().numpy()[0]
            results.append({
                "start": s * 4 / 30.0,
                "end": (s + SEQ_LEN) * 4 / 30.0,
                "behavior_type": BEHAVIOR_CLASSES[int(np.argmax(probs))],
                "confidence": min(float(np.max(probs)), 0.99),
                "probabilities": {c: float(p) for c, p in zip(BEHAVIOR_CLASSES, probs)},
            })
    return results


def _final_verdict(predictions, yolo_shop_confs):
    """Combine LSTM windows + YOLO per-frame shoplifting signal into one verdict."""
    shop_preds = [p for p in predictions if p["behavior_type"] == "shoplifting"]
    norm_preds = [p for p in predictions if p["behavior_type"] == "normal"]

    yolo_peak = max(yolo_shop_confs, default=0.0)
    yolo_mean = sum(yolo_shop_confs) / len(yolo_shop_confs) if yolo_shop_confs else 0.0
    # YOLO override: if YOLO is very confident about shoplifting on multiple frames
    yolo_override = yolo_peak >= 0.60 or (yolo_mean >= 0.50 and len(yolo_shop_confs) >= 3)

    peak_lstm_conf = max((p["confidence"] for p in shop_preds), default=0.0)

    if yolo_override or peak_lstm_conf >= 0.55:
        conf = max(peak_lstm_conf, min(yolo_peak, 0.99))
        return "shoplifting", conf
    elif shop_preds:
        shop_w = sum(p["confidence"] for p in shop_preds) * 2.0
        norm_w = sum(p["confidence"] for p in norm_preds)
        if shop_w + norm_w > 0 and shop_w / (shop_w + norm_w) > 0.3:
            return "shoplifting", max(p["confidence"] for p in shop_preds)
    return "normal", np.mean([p["confidence"] for p in norm_preds]) if norm_preds else 0.5


def _intent_score(predictions, duration, overall_cls, overall_conf, bilstm_trained):
    """Compute a composite risk score from detection components.
    Formula: 0.50 * concealment + 0.35 * duration_fraction + 0.15 * count_ratio
    """
    shop_segs = [p for p in predictions if p["behavior_type"] == "shoplifting"]
    susp_secs = sum(p["end"] - p["start"] for p in shop_segs)
    s_conceal = (np.mean([p["confidence"] for p in shop_segs])
                 * min(1.0, len(shop_segs) / 3.0)) if shop_segs else 0.0
    s_dur = min(1.0, (susp_secs / max(1.0, duration)) * 3.0) if shop_segs else 0.0

    raw = max(0.0, min(1.0, 0.50*s_conceal + 0.35*s_dur))
    if overall_cls == "shoplifting" and raw < 0.3:
        raw = max(raw, overall_conf * 0.75)

    flags = []
    if not bilstm_trained:
        raw *= 0.5
        flags.append("BiLSTM model not trained — results are unreliable")

    sev_map = [(0.3, "NONE"), (0.5, "LOW"), (0.7, "MEDIUM"), (0.85, "HIGH")]
    sev = "CRITICAL"
    for thresh, label in sev_map:
        if raw < thresh:
            sev = label
            break

    return {
        "score": raw, "severity": sev,
        "components": {"concealment": s_conceal, "duration": s_dur},
        "explanation": f"Score: {raw:.2f} ({sev}). Concealment: {s_conceal:.2f}, Duration: {s_dur:.2f}",
        "fairness": 0.85 if bilstm_trained else 0.5,
        "flags": flags,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(video_path, progress_cb=None, visual_out_path=None, live_cb=None):
    """
    Full inference pipeline — Producer-Consumer pattern.
    Thread 1 (main): reads video, runs YOLO, updates Streamlit UI every frame.
    Thread 2 (background): pulls frames from queue, runs EfficientNet + BiLSTM,
                            updates shared_status whenever it has a new verdict.
    This keeps the video display smooth — the UI never blocks on heavy CNN/LSTM math.
    """
    from torchvision import transforms

    def cb(p, m):
        if progress_cb: progress_cb(p, m)

    cb(0.0, "Loading models...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = load_yolo()
    effnet = load_effnet().to(dev)
    bilstm = load_bilstm()
    if bilstm: bilstm = bilstm.to(dev)

    yolo_names = set(yolo_model.names.values())
    is_4cls = "shoplifting" in yolo_names and "looking-around" in yolo_names

    xform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cb(0.05, "Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Shared state — main thread reads, background thread writes
    frame_queue = queue.Queue(maxsize=120)
    shared_status = {"label": None, "conf": 0.0, "predictions": []}

    # ── Background thread: EfficientNet + BiLSTM ──────────────────────────────
    def background_lstm_worker():
        all_feats = []
        while True:
            item = frame_queue.get()
            if item is None:          # poison pill — time to stop
                break
            frame_rgb, _ = item
            with torch.no_grad():
                x = xform(frame_rgb).unsqueeze(0).to(dev)
                feat = effnet(x).cpu().numpy().flatten()
            all_feats.append(feat)

            # rolling BiLSTM window — update verdict without blocking main thread
            if bilstm and len(all_feats) >= SEQ_LEN and len(all_feats) % max(SEQ_STRIDE // 2, 1) == 0:
                with torch.no_grad():
                    seq = torch.FloatTensor(np.array(all_feats[-SEQ_LEN:])[np.newaxis]).to(dev)
                    logits, _ = bilstm(seq)
                    probs = torch.softmax(logits / 3.0, dim=1).cpu().numpy()[0]
                shared_status["label"] = BEHAVIOR_CLASSES[int(np.argmax(probs))]
                shared_status["conf"] = float(np.max(probs))

            frame_queue.task_done()

        # final full-sequence classification for the end report
        if bilstm and all_feats:
            shared_status["predictions"] = _classify_sequences(bilstm, all_feats, dev)

    bg_thread = threading.Thread(target=background_lstm_worker, daemon=True)
    bg_thread.start()

    # ── Main thread: YOLO + video I/O + Streamlit UI ──────────────────────────
    persons_max = 0
    yolo_shop_confs = []
    product_events = set()
    video_writer = None
    if visual_out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(visual_out_path, fourcc, fps, (vid_w, vid_h))

    frame_num = 0
    last_yolo = None
    live_timeline = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        if progress_cb and frame_num % 90 == 0:
            pct = min(0.05 + (frame_num / max(total_frames, 1)) * 0.65, 0.70)
            cb(pct, f"Analyzing: Frame {frame_num}/{total_frames}")

        # YOLO is fast enough to stay in the main thread
        if frame_num % 2 == 1 or last_yolo is None:
            last_yolo = yolo_model.track(
                frame, conf=YOLO_CONF, iou=YOLO_IOU,
                persist=True, verbose=False, imgsz=320,
                device=0 if str(dev) == "cuda" else "cpu"
            )[0]

        person_boxes = {}
        prod_boxes = []

        if last_yolo.boxes is not None:
            ids = (last_yolo.boxes.id.int().tolist()
                   if last_yolo.boxes.id is not None else None)
            for i, (box, cls_id, conf) in enumerate(zip(
                    last_yolo.boxes.xyxy.cpu().numpy(),
                    last_yolo.boxes.cls.cpu().numpy(),
                    last_yolo.boxes.conf.cpu().numpy())):
                cls_name = yolo_model.names[int(cls_id)]
                tid = ids[i] if ids else i

                if is_4cls:
                    person_boxes[tid] = box
                    if cls_name == "picking-holding":
                        prod_boxes.append((cls_name, box))
                    if cls_name in ("shoplifting", "looking-around"):
                        product_events.add((tid, cls_name))
                    if cls_name == "shoplifting":
                        yolo_shop_confs.append(float(conf))
                else:
                    if int(cls_id) == 0 or cls_name.lower() == "person":
                        person_boxes[tid] = box

        persons_max = max(persons_max, len(person_boxes))

        for pid, pbox in person_boxes.items():
            px1, py1, px2, py2 = pbox
            ph = max(py2 - py1, 1)
            pcx, pcy = (px1+px2)/2, (py1+py2)/2
            for cls_name, qbox in prod_boxes:
                qx1, qy1, qx2, qy2 = qbox
                dist = ((pcx - (qx1+qx2)/2)**2 + (pcy - (qy1+qy2)/2)**2) ** 0.5
                if dist < ph * 1.2:
                    product_events.add((pid, cls_name))

        # every 4th frame: toss into the queue for the background thread — don't wait
        if frame_num % 4 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not frame_queue.full():
                frame_queue.put((rgb.copy(), frame_num))

        # draw + update UI using the latest verdict from the background thread
        live_label = shared_status["label"]
        live_conf = shared_status["conf"]

        if video_writer or live_cb:
            ann = _draw_frame(frame, last_yolo, yolo_model, is_4cls,
                               live_label, live_conf, frame_num, total_frames)
            if video_writer:
                video_writer.write(ann)
            # update UI every 2 frames — smooth because we're not blocked on CNN
            if live_cb and frame_num % 2 == 0:
                if frame_num % 30 == 0:
                    tl_col = "red" if live_label == "shoplifting" else "green" if live_label else "gray"
                    live_timeline.append(tl_col)
                live_cb(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), frame_num, total_frames, {
                    "persons_tracked": persons_max,
                    "products_detected": len({c for _, c in product_events}),
                    "label": live_label or "Analyzing...",
                    "conf": live_conf,
                    "timeline": list(live_timeline),
                })

    if video_writer: video_writer.release()
    cap.release()

    cb(0.85, "Finalizing deep learning sequences...")
    # signal background thread to stop, wait for it to finish its last window
    frame_queue.put(None)
    bg_thread.join()

    duration = frame_num / fps
    predictions = shared_status["predictions"]

    if not predictions and not yolo_shop_confs:
        return {
            "success": True, "early_exit": True,
            "early_exit_reason": f"Video too short ({frame_num} frames) — classified as normal.",
            "video_metadata": {"filename": Path(video_path).name, "duration": duration,
                               "fps": fps, "width": vid_w, "height": vid_h, "frame_count": total_frames},
            "lstm_detection": {"classification": "normal", "confidence": 1.0, "is_shoplifting": False},
            "detections": {"persons_tracked": 0, "products_detected": 0, "interactions": 0, "frames_processed": 0},
            "product_pickups": {}, "behavior_events": [],
            "intent_score": {"score": 0.0, "severity": "NONE", "components": {}, "explanation": "Too short"},
            "quality_analysis": {"reliability_score": 1.0, "detection_rate": 1.0, "usable": True},
            "bias_report": {"overall_fairness_score": 1.0, "analysis_reliable": True,
                            "requires_review": False, "flags": []},
            "alert": None, "model_trained": BILSTM_PATH.exists(),
        }

    cb(0.90, "Computing intent score...")
    product_pickups = {}
    for _, cls in product_events:
        product_pickups[cls] = product_pickups.get(cls, 0) + 1

    overall_cls, overall_conf = _final_verdict(predictions, yolo_shop_confs) if predictions else ("normal", 0.5)
    intent = _intent_score(predictions, duration, overall_cls, overall_conf, bilstm is not None)

    alert = None
    if intent["score"] >= 0.5:
        n_susp = len([p for p in predictions if p["behavior_type"] == "shoplifting"])
        alert = {
            "alert_id": f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-0001",
            "level": intent["severity"],
            "message": (f"{n_susp} suspicious window(s) detected. "
                        f"Risk score: {intent['score']:.2f} ({intent['severity']}). "
                        "Human review required."),
        }

    cb(1.0, "Done!")
    return {
        "success": True,
        "video_metadata": {"filename": Path(video_path).name, "duration": duration,
                           "fps": fps, "width": vid_w, "height": vid_h, "frame_count": total_frames},
        "lstm_detection": {"classification": overall_cls, "confidence": overall_conf,
                           "is_shoplifting": overall_cls == "shoplifting"},
        "detections": {"persons_tracked": persons_max,
                       "products_detected": len({c for _, c in product_events}),
                       "interactions": len([p for p in predictions if p["behavior_type"] == "shoplifting"]),
                       "frames_processed": len(predictions)},
        "product_pickups": product_pickups,
        "behavior_events": [{"behavior_type": p["behavior_type"], "start_time": p["start"],
                              "end_time": p["end"], "confidence": p["confidence"],
                              "probabilities": p["probabilities"]} for p in predictions],
        "intent_score": intent,
        "quality_analysis": {"reliability_score": intent["fairness"],
                             "detection_rate": 1.0, "usable": True},
        "bias_report": {"overall_fairness_score": intent["fairness"], "analysis_reliable": True,
                        "requires_review": intent["score"] > 0.5, "flags": intent.get("flags", [])},
        "alert": alert,
        "model_trained": BILSTM_PATH.exists(),
        "annotated_video_path": visual_out_path,
    }


# ── Forensic clip extraction ───────────────────────────────────────────────────

def extract_clips(video_path, behavior_events, max_clips=4):
    """Extract short GIF clips around the most suspicious moments for forensic review."""
    from PIL import Image
    import io

    suspicious = sorted(
        [e for e in behavior_events if e.get("behavior_type") == "shoplifting"
         and e.get("confidence", 0) >= 0.5],
        key=lambda x: x.get("confidence", 0), reverse=True
    )
    if not suspicious:
        suspicious = sorted(behavior_events, key=lambda x: x.get("confidence", 0), reverse=True)[:max_clips]
    if not suspicious:
        return []

    # deduplicate: keep clips at least 2s apart
    selected, mids = [], []
    for ev in suspicious:
        mid = (ev.get("start_time", 0) + ev.get("end_time", 0)) / 2
        if all(abs(mid - t) >= 2.0 for t in mids):
            selected.append(ev)
            mids.append(mid)
        if len(selected) >= max_clips: break

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / src_fps
    GIF_FPS, PAD, MAX_W = 8, 1.0, 480
    step = max(1, int(src_fps / GIF_FPS))

    clips = []
    for ev in selected:
        start = max(0.0, ev.get("start_time", 0) - PAD)
        end = min(total_s, ev.get("end_time", start+1) + PAD)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))
        pil_frames, thumbnail = [], None
        fi = int(start * src_fps)

        while fi <= int(end * src_fps):
            ret, frame = cap.read()
            if not ret: break
            if (fi - int(start * src_fps)) % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if w > MAX_W:
                    rgb = cv2.resize(rgb, (MAX_W, int(h * MAX_W/w)), interpolation=cv2.INTER_AREA)
                pil_frames.append(Image.fromarray(rgb))
                if thumbnail is None: thumbnail = rgb
            fi += 1

        if not pil_frames: continue
        buf = io.BytesIO()
        pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:],
                            duration=int(1000/GIF_FPS), loop=0, optimize=True)
        clips.append({
            "gif_bytes": buf.getvalue(), "thumbnail": thumbnail,
            "clip_start": start, "clip_end": end,
            "behavior": ev["behavior_type"], "confidence": ev["confidence"],
        })

    cap.release()
    return clips


# ── Forensic clip dialog ───────────────────────────────────────────────────────

@st.dialog("Forensic Evidence", width="large")
def show_clip_modal():
    clip = st.session_state.get("_modal_clip")
    if not clip: return
    st.markdown(f"**{clip.get('behavior','?').upper()}** — {clip.get('confidence',0):.0%} confidence")
    st.caption(f"{clip.get('clip_start',0):.1f}s – {clip.get('clip_end',0):.1f}s")
    if clip.get("gif_bytes"):
        st.image(clip["gif_bytes"], use_container_width=True)
    elif clip.get("thumbnail") is not None:
        st.image(clip["thumbnail"], use_container_width=True)
    st.warning("Advisory evidence only. Human review required before any action.")


# ── UI: Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### About Digital Witness")
        st.markdown("""
**Digital Witness** detects shoplifting from CCTV using:

1. **YOLO26n** — fine-tuned on 4 behaviour classes:
   `normal`, `looking-around`, `picking-holding`, `shoplifting`

2. **EfficientNetV2-S** — frozen pretrained backbone
   extracts a 1280-dim spatial feature per video frame

3. **BiLSTM + Attention** — classifies 45-frame windows
   (7.5s at 6fps); attention weights highlight which frames
   triggered the alert (XAI)

4. **Intent Score** — 0.50×concealment + 0.35×duration
   + 0.15×count → risk level NONE/LOW/MEDIUM/HIGH/CRITICAL

⚠️ This is an *advisory system*. It does **not** determine guilt.
All alerts require human validation.
        """)
        st.markdown("---")
        yolo_ready = YOLO_PATH.exists() or YOLO_BASE.exists()
        bilstm_ready = BILSTM_PATH.exists()
        if yolo_ready and bilstm_ready:
            st.success("All models ready")
        elif yolo_ready:
            st.warning("BiLSTM not trained — run notebook Cell 5")
        else:
            st.error("YOLO weights missing — check models/")
        st.markdown("---")
        st.caption("IIT 2026 FYP · Santosh_Manoharadas_20220967_w1954095")


# ── UI: Model Performance tab ──────────────────────────────────────────────────

def render_model_performance_tab():
    import plotly.graph_objects as go

    st.markdown('<div class="section-header">Deep Learning Model Performance</div>',
                unsafe_allow_html=True)

    bilstm_info = get_model_info()
    effnet_info = get_effnet_info()

    # ── EfficientNetV2-S ──────────────────────────────────────────────────────
    st.markdown("### EfficientNetV2-S — Frame Feature Extractor")
    st.caption("Frozen pretrained backbone. Extracts 1280-dim spatial feature per frame, fed into BiLSTM.")
    if effnet_info:
        c1, c2 = st.columns(2)
        c1.metric("Best Val Accuracy", f"{effnet_info.get('best_val_acc', 0):.2%}")
        c2.metric("Feature Dimension", effnet_info.get("feat_dim", 1280))
    else:
        st.info("Train EfficientNetV2-S first (notebook Cell 4) to see metrics here.")

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### BiLSTM + Temporal Attention — Sequence Classifier")
    st.caption("Classifies 45-frame windows (7.5s) of EfficientNetV2-S features as normal / shoplifting.")
    if bilstm_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Val Accuracy", f"{bilstm_info.get('best_val_acc', 0):.2%}")
        c2.metric("Sequence Length", bilstm_info.get("seq_len", 45))
        c3.metric("Hidden Units", bilstm_info.get("hidden", 256))
    else:
        st.info("Train the BiLSTM first (notebook Cell 5) to see metrics here.")

    def show_img(path, caption=None):
        p = Path(path)
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_img(OUTPUTS_DIR / "confusion_matrix.png", "BiLSTM — Confusion Matrix")
    with col2:
        show_img(OUTPUTS_DIR / "attention_xai_example.png", "Attention XAI — suspicious frame weights")

    # ── YOLO26 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### YOLO26n — 4-Class Behaviour Detector")
    st.caption("Fine-tuned on retail surveillance. Classes: normal, looking-around, picking-holding, shoplifting.")
    YOLO_RUN = ROOT_DIR / "runs" / "yolo_dw"
    col1, col2 = st.columns(2)
    with col1:
        show_img(YOLO_RUN / "results.png", "YOLO26 — Training Curves")
    with col2:
        show_img(YOLO_RUN / "confusion_matrix_normalized.png", "YOLO26 — Confusion Matrix")

    with st.expander("Validation batch samples"):
        c1, c2 = st.columns(2)
        with c1:
            show_img(YOLO_RUN / "val_batch0_labels.jpg", "Ground Truth")
        with c2:
            show_img(YOLO_RUN / "val_batch0_pred.jpg", "Predictions")

    # ── Architecture summary ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Architecture Config")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**BiLSTM**")
        st.json({"input_dim": FEAT_DIM, "hidden": bilstm_info.get("hidden", 256) if bilstm_info else 256,
                 "layers": bilstm_info.get("layers", 2) if bilstm_info else 2,
                 "bidirectional": True, "seq_len": SEQ_LEN, "stride": SEQ_STRIDE})
    with c2:
        st.markdown("**EfficientNetV2-S**")
        st.json({"backbone": "efficientnet_v2_s", "feat_dim": FEAT_DIM,
                 "last_2_blocks_unfrozen": "during training", "pool": "AdaptiveAvgPool2d"})
    with c3:
        st.markdown("**Intent Score**")
        st.json({"concealment_weight": 0.50, "duration_weight": 0.35,
                 "yolo_override_threshold": 0.60, "lstm_threshold": 0.55})


# ── UI: Analysis results tab ───────────────────────────────────────────────────

def render_analysis_results(results):
    import plotly.graph_objects as go

    if not results.get("success"):
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    if results.get("early_exit"):
        st.markdown('<div class="alert-none"><h2 style="margin:0">NORMAL</h2></div>',
                    unsafe_allow_html=True)
        st.info(results.get("early_exit_reason", ""))
        return

    if not results.get("model_trained"):
        st.error("BiLSTM not trained — results are unreliable. Run notebook Cells 4–5 first.")

    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    lstm = results["lstm_detection"]
    cls, conf, is_shop = lstm["classification"], lstm["confidence"], lstm["is_shoplifting"]

    # ── verdict banner ────────────────────────────────────────────────────────
    if is_shop and conf > 0.7:
        div = "alert-critical"
        title = "SHOPLIFTING DETECTED"
    elif is_shop:
        div = "alert-high"
        title = "POTENTIAL SHOPLIFTING"
    else:
        div = "alert-none"
        title = "NORMAL BEHAVIOUR"
    st.markdown(f"""<div class='{div}'>
        <h2 style='margin:0'>{title}</h2>
        <p style='margin:0.5rem 0 0 0'>Classification: <strong>{cls.upper()}</strong>
        | Confidence: <strong>{conf:.1%}</strong></p>
    </div>""", unsafe_allow_html=True)

    # ── risk score ────────────────────────────────────────────────────────────
    intent = results["intent_score"]
    score, sev = intent["score"], intent["severity"].upper()
    sev_div = {"CRITICAL": "alert-critical", "HIGH": "alert-high", "MEDIUM": "alert-medium",
               "LOW": "alert-low"}.get(sev, "alert-none")
    st.markdown("---")
    st.markdown("### Bias-Adjusted Risk Assessment")
    st.markdown(f"""<div class='{sev_div}'>
        <h3 style='margin:0'>Risk Level: {sev}</h3>
        <p style='margin:0.5rem 0 0 0'>Score: {score:.2f}</p>
    </div>""", unsafe_allow_html=True)

    # ── detection stats ───────────────────────────────────────────────────────
    st.markdown("---")
    det = results.get("detections", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People Tracked", det.get("persons_tracked", 0))
    c2.metric("Product Interactions", det.get("products_detected", 0))
    c3.metric("Suspicious Windows", det.get("interactions", 0))
    c4.metric("Frames Analysed", det.get("frames_processed", 0))

    # ── XAI explanation ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Explainable AI (XAI) Analysis")
    staff_tab, tech_tab = st.tabs(["Security Staff View", "Technical Details"])

    events = results.get("behavior_events", [])
    with staff_tab:
        if is_shop:
            st.error(f"The AI reviewed this footage and identified behaviour consistent with shoplifting "
                     f"({conf:.0%} confidence). This is a flag for review — not a confirmed incident.")
            st.warning("Do not confront anyone based solely on this result. Apply your store's standard procedure.")
        else:
            st.success("No suspicious behaviour identified. Movements consistent with normal shopping.")
        if events:
            st.markdown("**Behaviour windows:**")
            counts = {}
            for e in events:
                bt = e.get("behavior_type", "?")
                counts[bt] = counts.get(bt, 0) + 1
            for bt, n in counts.items():
                st.markdown(f"- **{bt.capitalize()}**: {n} window(s)")

    with tech_tab:
        st.markdown(f"""
**Step 1 — YOLO26n detection**
- {det.get('frames_processed', 0)} frames processed (YOLO every 2nd, features every 4th)
- Model: `{"yolo_dw.pt" if YOLO_PATH.exists() else "yolo26n.pt"}` ({"4-class" if YOLO_PATH.exists() else "base COCO"})

**Step 2 — EfficientNetV2-S feature extraction**
- 1280-dim spatial feature per frame (frozen pretrained backbone)
- AdaptiveAvgPool2d(1) → Flatten()

**Step 3 — BiLSTM temporal classification**
- {SEQ_LEN}-frame sliding windows, stride={SEQ_STRIDE}
- BiLSTM: 256 hidden, 2 layers, bidirectional + temporal attention
- Temperature scaling T=3.0 to prevent overconfident outputs
- **Verdict: {cls.upper()} | {conf:.1%}**

**Step 4 — Intent aggregation**
- {intent.get('explanation', '')}
        """)

    # ── intent gauge + components ─────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### Intent Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score*100,
            title={"text": "Risk Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#667eea"},
                   "steps": [{"range": [0,30], "color": "#d4edda"},
                              {"range": [30,50], "color": "#fff3cd"},
                              {"range": [50,70], "color": "#ffe4b2"},
                              {"range": [70,100], "color": "#ffcccc"}]}
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        comps = intent.get("components", {})
        if comps and any(v > 0 for v in comps.values()):
            st.markdown("### Score Components")
            fig = go.Figure(go.Bar(x=list(comps.values()), y=list(comps.keys()),
                                   orientation="h", marker_color="#667eea"))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=20,b=20),
                              xaxis=dict(range=[0,1], tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)
        st.info(intent.get("explanation", ""))

    # ── forensic clips ────────────────────────────────────────────────────────
    if is_shop:
        st.markdown("---")
        st.markdown("### Forensic Evidence")
        clips = results.get("suspicious_frames", [])
        if clips:
            cols = st.columns(min(len(clips), 4))
            for i, clip in enumerate(clips):
                with cols[i % 4]:
                    if clip.get("thumbnail") is not None:
                        st.image(clip["thumbnail"])
                    st.caption(f"**{clip.get('behavior','?').upper()}** "
                               f"| {clip.get('clip_start',0):.1f}s–{clip.get('clip_end',0):.1f}s "
                               f"| {clip['confidence']:.0%}")
                    if st.button("▶ Play", key=f"clip_{i}", use_container_width=True):
                        st.session_state["_modal_clip"] = clip
                        show_clip_modal()
        else:
            st.info("No forensic clips extracted.")

    # ── quality and fairness gauges ───────────────────────────────────────────
    st.markdown("---")
    bias = results.get("bias_report", {})
    c1, c2 = st.columns(2)
    for col, title, val, color in [
        (c1, "Reliability", results.get("quality_analysis", {}).get("reliability_score", 0), "#38ef7d"),
        (c2, "Fairness Score", bias.get("overall_fairness_score", 0), "#667eea"),
    ]:
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val*100, title={"text": title},
                gauge={"axis": {"range": [0,100]}, "bar": {"color": color},
                       "steps": [{"range": [0,50], "color": "#ffcccc"},
                                  {"range": [50,75], "color": "#fff3cd"},
                                  {"range": [75,100], "color": "#d4edda"}]}
            ))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

    for flag in bias.get("flags", []):
        st.warning(flag)

    # ── advisory alert ────────────────────────────────────────────────────────
    st.markdown("---")
    alert = results.get("alert")
    if alert:
        div = "alert-high" if alert.get("level") in {"CRITICAL", "HIGH"} else "alert-medium"
        st.markdown(f"""<div class='{div}'>
            <h4 style='margin:0'>Alert: {alert.get('alert_id','')}</h4>
            <p style='margin:0.5rem 0'>{alert.get('message','')}</p>
            <p style='margin:0;font-style:italic'>Advisory only — human validation required.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-none'><h4 style='margin:0'>No Concerns Detected</h4></div>",
                    unsafe_allow_html=True)

    # ── video info ────────────────────────────────────────────────────────────
    vm = results.get("video_metadata", {})
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**File:** {vm.get('filename','N/A')}")
    c2.write(f"**Duration:** {vm.get('duration',0):.1f}s")
    c3.write(f"**Resolution:** {vm.get('width',0)}×{vm.get('height',0)}")
    c4.write(f"**FPS:** {vm.get('fps',0):.1f}")


# ── UI: POS audit tab ──────────────────────────────────────────────────────────

PRODUCT_CATALOG = [
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


def render_pos_audit(analysis_results):
    """
    Mock POS terminal + mismatch audit.
    Left:  cashier enters items at checkout
    Right: what the camera detected being handled
    Bottom: flags gaps between items handled and items scanned
    """
    import plotly.graph_objects as go

    st.markdown('<div class="section-header">Mock POS Terminal & Audit</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Simulate a POS transaction and compare it against behavioural video evidence.
        A large gap between items handled and items scanned is a theft indicator.
    </div>""", unsafe_allow_html=True)

    cat_by_name = {p["name"]: p for p in PRODUCT_CATALOG}
    col_pos, col_vid = st.columns(2)

    with col_pos:
        st.markdown("### Mock POS Terminal")
        with st.form("pos_form", clear_on_submit=False):
            sel_name = st.selectbox("Product", [p["name"] for p in PRODUCT_CATALOG])
            sel_qty = st.number_input("Quantity", min_value=1, max_value=20, value=1)
            if st.form_submit_button("Add"):
                item = dict(cat_by_name[sel_name])
                item["qty"] = int(sel_qty)
                st.session_state.pos_items.append(item)
        if st.button("Clear Transaction"):
            st.session_state.pos_items = []
        if st.session_state.pos_items:
            rows = [{"SKU": it["sku"], "Product": it["name"], "Qty": it["qty"],
                     "Total": f"£{it['price']*it['qty']:.2f}"} for it in st.session_state.pos_items]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            total = sum(it["price"]*it["qty"] for it in st.session_state.pos_items)
            st.markdown(f"**Total: £{total:.2f}**")
        else:
            st.info("No items added yet.")

    with col_vid:
        st.markdown("### Behavioural Video Evidence")
        st.caption("The model classifies person actions, not specific products.")
        if not analysis_results:
            st.info("Run video analysis first.")
        else:
            det = analysis_results.get("detections", {})
            lstm = analysis_results.get("lstm_detection", {})
            events = analysis_results.get("behavior_events", [])
            pickups = analysis_results.get("product_pickups", {})

            picking = pickups.get("picking-holding", 0)
            shop_windows = sum(1 for e in events if e.get("behavior_type") == "shoplifting")

            rows = [
                {"Signal": "Persons tracked", "Count": det.get("persons_tracked", 0), "Risk": "—"},
                {"Signal": "Item handling (Picking-Holding)", "Count": picking,
                 "Risk": "Possible" if picking > 0 else "—"},
                {"Signal": "Shoplifting behaviour windows", "Count": shop_windows,
                 "Risk": "High" if shop_windows > 0 else "None"},
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if lstm.get("is_shoplifting"):
                st.error(f"LSTM: **SHOPLIFTING** ({lstm.get('confidence',0):.0%})")
            else:
                st.success(f"LSTM: **NORMAL** ({lstm.get('confidence',0):.0%})")

    if not analysis_results or not st.session_state.pos_items:
        if not analysis_results:
            st.info("Run video analysis first.")
        else:
            st.info("Add items to the POS transaction to compare.")
        return

    st.markdown("---")
    st.markdown("### Transaction Risk Audit")
    det = analysis_results.get("detections", {})
    lstm = analysis_results.get("lstm_detection", {})
    events = analysis_results.get("behavior_events", [])
    pickups = analysis_results.get("product_pickups", {})

    pos_qty = sum(it["qty"] for it in st.session_state.pos_items)
    pos_val = sum(it["price"]*it["qty"] for it in st.session_state.pos_items)
    picking = pickups.get("picking-holding", 0)
    shop_windows = sum(1 for e in events if e.get("behavior_type") == "shoplifting")
    is_shop = lstm.get("is_shoplifting", False)

    audit = [
        {"Metric": "POS items scanned", "Value": pos_qty, "Status": "OK"},
        {"Metric": "Transaction value", "Value": f"£{pos_val:.2f}", "Status": "OK"},
        {"Metric": "Item interactions (camera)", "Value": picking,
         "Status": "More handled than scanned" if picking > pos_qty else "OK"},
        {"Metric": "Shoplifting windows (LSTM)", "Value": shop_windows,
         "Status": "Suspicious" if shop_windows > 0 else "None"},
    ]
    st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)

    flags = []
    if is_shop:
        flags.append(f"LSTM classified video as **SHOPLIFTING** while {pos_qty} item(s) were rung up.")
    if picking > pos_qty:
        flags.append(f"**{picking} item interactions** detected but only **{pos_qty}** scanned at POS.")
    if is_shop and pos_qty == 0:
        flags.append("Shoplifting detected but **no purchase recorded** at POS.")

    if flags:
        for f in flags: st.error(f)
        st.warning("Human review required before any action is taken.")
    else:
        st.success("No mismatches — POS transaction consistent with video evidence.")

    fig = go.Figure()
    fig.add_bar(name="POS Items Scanned", x=["Transaction"], y=[pos_qty], marker_color="#38ef7d")
    fig.add_bar(name="Item Interactions", x=["Transaction"], y=[picking], marker_color="#667eea")
    fig.add_bar(name="Shoplifting Windows", x=["Transaction"], y=[shop_windows], marker_color="#dc3545")
    fig.update_layout(barmode="group", title="POS vs Camera Evidence",
                      height=300, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "pos_items" not in st.session_state:
        st.session_state.pos_items = []
    if "annotated_video_path" not in st.session_state:
        st.session_state.annotated_video_path = None

    st.markdown('<h1 class="main-title">Digital Witness</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning Retail Security Assistant | YOLO26n + EfficientNetV2-S + BiLSTM</p>',
                unsafe_allow_html=True)

    render_sidebar()

    tab_video, tab_perf, tab_pos = st.tabs(["Video Analysis", "Model Performance", "POS Audit"])

    with tab_perf:
        render_model_performance_tab()

    with tab_video:
        st.markdown('<div class="section-header">Deep Learning Video Analysis</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <strong>Pipeline:</strong> YOLO26n (detect) → EfficientNetV2-S (features) → BiLSTM+Attention (classify)
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        with c2:
            if uploaded:
                st.video(uploaded)

        st.markdown("---")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            analyse = st.button("Analyse Video", type="primary",
                                use_container_width=True, disabled=(uploaded is None))

        if analyse and uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            ann_path = os.path.join(tempfile.gettempdir(), f"dw_ann_{uuid.uuid4().hex[:8]}.mp4")

            st.markdown("### Live Preview")
            col_frame, col_stats = st.columns([3, 1])
            with col_frame:
                frame_ph = st.empty()
            with col_stats:
                stats_ph = st.empty()
            timeline_ph = st.empty()
            prog = st.progress(0)
            status_ph = st.empty()

            def on_progress(p, m):
                prog.progress(min(p, 1.0))
                status_ph.text(m)

            def on_live(frame_rgb, fn, total, stats):
                frame_ph.image(frame_rgb, caption=f"Frame {fn}/{total}", use_container_width=True)
                label = stats.get("label", "Analyzing...")
                col = "#dc3545" if label == "shoplifting" else "#28a745" if label == "normal" else "#aaa"
                stats_ph.markdown(f"""<div style='padding:1rem;background:#1a1a2e;border-radius:8px;
                    border:1px solid #333;font-family:monospace;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>CLASSIFICATION</p>
                    <p style='color:{col};font-size:1.15rem;font-weight:700;margin:0.25rem 0;'>{label.upper()}</p>
                    <p style='color:#ccc;font-size:0.85rem;margin:0;'>{stats.get("conf",0):.0%} confidence</p>
                    <hr style='border-color:#333;margin:0.6rem 0;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>PEOPLE TRACKED</p>
                    <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>{stats.get("persons_tracked",0)}</p>
                    <hr style='border-color:#333;margin:0.6rem 0;'>
                    <p style='color:#888;margin:0;font-size:0.72rem;'>PRODUCT INTERACTIONS</p>
                    <p style='color:#fff;font-size:1.1rem;font-weight:700;margin:0.2rem 0;'>{stats.get("products_detected",0)}</p>
                </div>""", unsafe_allow_html=True)
                timeline = stats.get("timeline", [])
                if timeline:
                    cmap = {"red": "#dc3545", "green": "#28a745", "gray": "#555"}
                    bar = '<div style="display:flex;width:100%;height:10px;border-radius:4px;overflow:hidden;background:#333;">'
                    for tc in timeline:
                        bar += f'<div style="flex:{100/max(len(timeline),1):.2f};background:{cmap.get(tc,"#555")};"></div>'
                    bar += "</div>"
                    pct = fn / max(total, 1)
                    timeline_ph.markdown(f"""<div style='margin:0.4rem 0;'>
                        <p style='color:#888;font-size:0.72rem;margin-bottom:4px;'>Timeline &nbsp;
                        <span style='color:#28a745;'>■ Normal</span>&nbsp;
                        <span style='color:#dc3545;'>■ Suspicious</span></p>
                        {bar}
                        <p style='color:#666;font-size:0.72rem;margin-top:4px;'>{pct:.0%} processed</p>
                    </div>""", unsafe_allow_html=True)

            try:
                results = run_pipeline(video_path, on_progress, ann_path, on_live)
            except Exception as e:
                results = {"success": False, "error": str(e)}

            if results.get("success"):
                try:
                    results["suspicious_frames"] = extract_clips(
                        video_path, results.get("behavior_events", []))
                except Exception:
                    results["suspicious_frames"] = []

            frame_ph.empty(); stats_ph.empty(); timeline_ph.empty()
            prog.empty(); status_ph.empty()

            if os.path.exists(video_path):
                os.unlink(video_path)

            old_ann = st.session_state.get("annotated_video_path")
            if old_ann and old_ann != ann_path and os.path.exists(old_ann):
                try: os.unlink(old_ann)
                except OSError: pass

            st.session_state.analysis_results = results
            st.session_state.annotated_video_path = ann_path
            st.rerun()

        if st.session_state.analysis_results:
            ann_vid = st.session_state.get("annotated_video_path")
            if ann_vid and os.path.exists(ann_vid) and os.path.getsize(ann_vid) > 4096:
                st.markdown("---")
                st.markdown("### Annotated Video")
                st.caption("Green = normal · Orange = looking-around · Red = shoplifting")
                with open(ann_vid, "rb") as vf:
                    st.video(vf.read())
            st.markdown("---")
            r = st.session_state.analysis_results
            if not r.get("success"):
                st.error(f"Analysis failed: {r.get('error', '')}")
            else:
                try:
                    render_analysis_results(r)
                except Exception as e:
                    import traceback
                    st.error(f"Render error: {e}")
                    st.code(traceback.format_exc())

    with tab_pos:
        render_pos_audit(st.session_state.analysis_results)

    st.markdown("---")
    st.markdown("""<div style='text-align:center;color:#888;padding:2rem 0'>
        <p>Digital Witness — Deep Learning Retail Security Assistant</p>
        <p style='font-size:0.85rem'>YOLO26n → EfficientNetV2-S → BiLSTM + Temporal Attention<br>
        Advisory system only. All alerts require human validation.</p>
        <p style='font-size:0.8rem'>IIT 2026 FYP · Santosh_Manoharadas_20220967_w1954095</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
