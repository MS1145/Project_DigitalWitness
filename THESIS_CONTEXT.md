# Digital Witness — Complete Thesis Context Document

**Student:** Santosh Manoharadas | W1954095 / 20220967  
**Project:** Digital Witness — Bias-Aware, Explainable Retail Security Assistant  
**Institution:** IIT (Final Year Project, 2025–2026)  
**Deadline:** 30 March 2026  
**Current Status:** Version 3.0 (dev3.0 branch) — production-grade single-rule classification implemented

---

## INSTRUCTIONS FOR THE AI THESIS ASSISTANT

This document is the single source of truth for the Digital Witness project. When helping write the thesis, draw exclusively from the facts stated here. Do not invent model performance numbers, do not assume standard architectures — the exact architectures, hyperparameters, and design decisions are documented in full below. The thesis should be written in first-person academic voice. The student's name is Santosh Manoharadas. The project is an undergraduate Final Year Project. Key themes to emphasise: explainability-first design, bias awareness, advisory-only output, the three-model cascade pipeline, and the POS-behaviour fusion approach.

---

## 1. PROBLEM STATEMENT

Current AI-powered retail security systems operate as **opaque black boxes**. They cannot reliably distinguish between:
- Intentional shoplifting
- Accidental behaviour (e.g. a customer forgetting to scan an item)
- Vulnerable behaviour (e.g. elderly confusion, children misunderstanding)

This causes:
- False accusations and legal liability
- Erosion of customer trust
- Ethical concerns around automated surveillance
- No audit trail for human operators to review decisions

Additionally, existing systems treat **behavioural video evidence** and **POS transaction data** as entirely separate silos, making it impossible to determine whether items physically handled were billed.

---

## 2. PROJECT VISION AND DESIGN PHILOSOPHY

Digital Witness is designed as a **Blameless AI Assistant** — an advisory tool that supports, not replaces, human decision-making.

**Core design principle:** The system NEVER determines guilt. It generates an intent risk score with an explainable evidence trail. All alerts require human validation before any action.

**Three non-negotiable constraints built into the system:**
1. No facial recognition or identity inference — analysis is body-pose only
2. No personal data stored — all processing is stateless per video
3. Every output includes a mandatory human-review disclaimer

**XAI as primary goal:** The project explicitly frames Explainable AI (XAI) as the primary design goal, not detection accuracy. A high-accuracy black box is explicitly rejected in favour of a lower-accuracy transparent system that a security officer can actually reason about and challenge.

---

## 3. SYSTEM ARCHITECTURE OVERVIEW

### 3.1 Three-Phase Model Cascade

The inference pipeline runs three sequential deep learning models:

```
Video Input
    │
    ▼
[Stage 1] YOLO26n — Per-frame behaviour detection
           4 classes: shoplifting | Looking around | Picking-Holding | normal
           ByteTrack multi-object tracking
           Runs every 2nd frame (yolo_step=2), reuses detection on skipped frames
           imgsz=320 (4× speed vs default 640)
    │
    ├── yolo_frame_labels[] ← (frame_num, dominant_class, confidence)
    ├── yolo_shop_confs[] ← peak shoplifting confidence per frame
    ├── product_pickup_events ← set of (person_track_id, product_class)
    │
    ▼
[Stage 2] MobileNetV2 — Spatial feature extraction
           Runs every 4th frame (feat_step=4) for speed
           Input: 224×224 RGB person crop
           Output: 1280-dimensional feature vector
           Uses ImageNet weights: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    │
    ├── all_feats[] ← list of 1280-dim vectors
    │
    ▼
[Stage 3] BiLSTM + Temporal Attention — Sequence classification
           Input: sliding windows of 45 consecutive feature vectors
           Stride: 15 features (LSTM_STRIDE=15)
           Temporal coverage: 45 frames × 4 (feat_step) = 180 video frames ≈ 6s at 30fps
           Temperature scaling T=3.0 (softens overconfident distributions)
           Output: per-window probability over {normal, shoplifting}
           ADVISORY ONLY — does not affect final verdict
    │
    ▼
[Final Rule] Single Peak Threshold Classification
           yolo_peak = max shoplifting confidence across all frames
           ≥ 70% → SHOPLIFTING  (red banner)
           50–69% → NEEDS REVIEW (orange banner)
           < 50%  → NORMAL      (green banner)
```

### 3.2 Why This Architecture

**YOLO as primary signal:** YOLO sees every frame and makes real-time per-frame decisions. Its shoplifting confidence is the most direct signal available. The BiLSTM adds temporal context but was found to have a training distribution bias toward "normal" (training data imbalance), making it unreliable as a primary signal.

**BiLSTM as advisory / XAI:** The attention weights from the BiLSTM show *which frames* in a 7.5-second window most influenced the classification. This is the XAI output — a temporal explanation of which moments were suspicious. It is shown to the operator for transparency but does not affect the verdict.

**Single peak threshold (final rule):** Multiple iterations of more complex rules (segment voting, majority voting, sustained-frame rules) were trialled and found to produce inconsistency between what the live preview showed and what the final verdict said. The single-peak rule is the simplest possible rule that is also the most transparent to explain to a security officer: "if the model was ever more than 70% confident it saw shoplifting, the verdict is shoplifting."

---

## 4. MODEL ARCHITECTURES (DETAILED)

### 4.1 YOLO26n (Stage 1)

**Model family:** YOLOv8 nano variant (ultralytics library)  
**Base weights:** yolo26n.pt (COCO-pretrained)  
**Fine-tuned variant:** yolo26_dw_v2.pt — domain-adaptive fine-tune on 24-class Roboflow retail annotation dataset  
**Training strategy:** Backbone frozen (freeze=10) — only the detection head learns retail-specific classes

**Model priority at inference (automatic fallback):**
1. `yolo26_dw_v2.pt` — domain-adaptive fine-tune (preferred)
2. `yolo26n.pt` — COCO base (reliable person detection)
3. `yolo26_retail.pt` — retail-only fine-tune (legacy fallback)

**Inference hyperparameters:**
- `YOLO_CONF = 0.2` (confidence threshold — deliberately low to capture ambiguous poses)
- `YOLO_IOU = 0.45` (IoU threshold for NMS)
- `imgsz = 320` (inference resolution — 4× faster than default 640 with minor accuracy tradeoff)
- FP16 half-precision on GPU, full float32 on CPU

**4-class behaviour model:**
| Class | Priority | Meaning |
|-------|----------|---------|
| shoplifting | 3 (highest) | Direct shoplifting gesture detected |
| Looking around | 2 | Suspicious reconnaissance behaviour |
| Picking-Holding | 1 | Product interaction / potential concealment |
| normal | 0 (lowest) | Normal shopping behaviour |

**Per-frame dominant label selection:**  
When multiple bounding boxes exist in a single frame, the highest-priority class wins. Ties broken by confidence score. Priority: shoplifting > Looking around > Picking-Holding > normal.

**ByteTrack multi-object tracking:** Assigns persistent track IDs across frames. Used to count unique persons and associate product interactions with specific individuals.

**COCO / unknown model fallback:**  
If the fine-tuned model is unavailable, the system falls back to COCO behaviour: class 0 = person (detected for tracking), COCO product classes (bottle, cup, backpack, etc.) detected for interaction tracking. No 4-class behaviour signal in this mode — only person count and product proximity.

**Bounding box colour coding (live preview):**
- Red (#dc0000) — shoplifting ≥ 50%
- Green (0,200,0) — shoplifting < 50% (shown as "normal" in label)
- Orange/cyan (0,140,255) — Looking around
- Teal (0,200,255) — Picking-Holding

### 4.2 MobileNetV2 Feature Extractor (Stage 2)

**Base architecture:** torchvision MobileNetV2 (pretrained on ImageNet — MobileNet_V2_Weights.DEFAULT)

**Modified architecture:**
```python
class MobileNetV2Extractor(nn.Module):
    def __init__(self):
        self.features   = mobilenet_v2.features      # 18-layer inverted residual blocks
        self.pool       = nn.AdaptiveAvgPool2d((1,1)) # global average pooling → (1,1)
        self.classifier = nn.Sequential(             # classification head (loaded but unused)
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )
    
    def extract_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)   # → (1, 1280)
```

**During inference:** Only `extract_features()` is called. The classifier head is loaded from checkpoint but never used during pipeline inference.

**Training setup (Cell 4 in notebook):**
- Input: 224×224 RGB person crops, sampled at FPS_TARGET = 6fps
- Data split: 80% train / 10% validation / 10% test (stratified by class, fixed seed=42)
- `WeightedRandomSampler` used to handle class imbalance
- ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Reported performance: val_acc = 97.7%** (from notebook header)

**Feature extraction frequency:** Every 4th video frame (`feat_step=4`). This gives approximately 7.5 features/second at 30fps source video, matching the BiLSTM's temporal granularity.

### 4.3 BiLSTM + Temporal Attention (Stage 3)

**Architecture:**
```python
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self):
        self.bilstm = nn.LSTM(
            input_size  = 1280,       # MobileNetV2 output dim
            hidden_size = 256,        # hidden units per direction
            num_layers  = 2,          # stacked LSTM
            batch_first = True,
            bidirectional = True,     # → effective hidden = 512
            dropout = 0.3,            # between LSTM layers
        )
        self.attention  = TemporalAttention(512)   # 512 = hidden*2
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, 2)        # → {normal, shoplifting}
```

**Temporal Attention mechanism:**
```python
class TemporalAttention(nn.Module):
    def __init__(self, h):
        self.attn = nn.Sequential(
            nn.Linear(h, h // 2),   # 512 → 256
            nn.Tanh(),
            nn.Linear(h // 2, 1),   # 256 → 1 (scalar attention score per timestep)
        )
    
    def forward(self, lstm_out):   # lstm_out: (B, T=45, 512)
        scores  = self.attn(lstm_out).squeeze(-1)   # (B, T)
        weights = torch.softmax(scores, dim=1)       # normalised attention distribution
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # weighted sum
        return context, weights   # weights are the XAI output
```

**The attention weights are the XAI contribution:** Each weight value represents what fraction of the model's decision was attributed to that specific 4-frame window. Frames with weights above the uniform baseline (1/45 ≈ 2.2%) indicate moments of heightened suspicious activity. This is visualised as a bar chart in the notebook (Cell 7).

**Sequence parameters:**
- `LSTM_SEQ_LEN = 45` — 45 consecutive feature vectors per window (= 45×4 = 180 video frames ≈ 6 seconds at 30fps)
- `LSTM_STRIDE = 15` — step between windows (50% overlap when divided into 45-frame windows)

**Temperature scaling:**  
`T = 3.0` applied at softmax: `probs = softmax(logits / 3.0)`. This flattens overconfident distributions. The BiLSTM was found to produce very high-confidence outputs even when the signal was weak, so temperature scaling prevents false certainty. Short clips also receive a `scale = min(1.0, len(all_feats) / LSTM_SEQ_LEN)` penalty.

**Short clip handling:**  
If `len(all_feats) < LSTM_SEQ_LEN`, the feature array is zero-padded on the right to reach length 45. The padding frames are silent (all zeros) which the LSTM interprets as absent context, automatically reducing confidence via the scale factor.

**Training setup (Cell 5):**
- Data split: same 80/10/10 strategy as MobileNetV2, test set held out until Cell 7
- `WeightedRandomSampler` for class balance
- Input sequences: 45-frame windows of 1280-dim feature vectors
- **Reported performance: val_acc = 88.6%** (from notebook header)

**Known limitation:** The BiLSTM has a training distribution bias toward classifying everything as "normal." This was discovered empirically: debug output showed 24/24 LSTM windows returning "normal" at 97.5% confidence on a video that YOLO flagged as shoplifting (4 segments ≥ 50%, peak 64%). This is why the BiLSTM was demoted to advisory-only status.

---

## 5. CLASSIFICATION RULE (CURRENT — VERSION 3.0)

### 5.1 The Single Peak Threshold Rule

After iterating through multiple classification approaches (described in Section 9), the final rule is:

```
yolo_peak = max(shoplifting_confidence across all video frames)

if yolo_peak >= 0.70:
    verdict = "SHOPLIFTING"   → red banner, full security action panel
elif yolo_peak >= 0.50:
    verdict = "NEEDS REVIEW"  → orange banner, human review required
else:
    verdict = "NORMAL"        → green banner, compact summary only
```

**Rationale:** A single number — the peak shoplifting confidence — is the simplest possible rule. It is directly explainable to any non-technical security officer: "The model saw a moment where it was more than 70% confident that shoplifting was happening." No voting, no segment counting, no accumulated conditions.

### 5.2 Intent Score (Severity within Shoplifting verdict)

Once a verdict is determined, an additional risk severity score (0.0–1.0) is computed for the SHOPLIFTING and NEEDS REVIEW cases:

```python
# For shoplifting/review cases:
_n_segs    = count of 30-frame segments with max_shop_conf >= 0.50
_sustained = min(_n_segs / 5.0, 1.0)    # 5+ qualifying segments → full score
raw_score  = min(yolo_peak * (0.5 + 0.5 * _sustained), 1.0)
s_conceal  = yolo_peak
s_bypass   = 0.0                         # bypass scoring disabled (LSTM advisory only)
s_dur      = _sustained
```

**Severity thresholds:**
| Score | Severity |
|-------|----------|
| < 0.30 | NONE |
| 0.30–0.49 | LOW |
| 0.50–0.69 | MEDIUM |
| 0.70–0.84 | HIGH |
| ≥ 0.85 | CRITICAL |

**Original intent scoring formula (from notebook Cell 10, used in earlier versions):**
```
score = 0.40 × behaviour_score     (BiLSTM output)
      + 0.30 × concealment_score   (YOLO concealment classes)
      + 0.20 × pos_mismatch_score  (POS discrepancy)
      + 0.10 × duration_score      (temporal persistence)
```
This formula is documented in the notebook as the research-phase design. The live application now uses the simplified YOLO-peak-based formula above, with POS integration handled separately in the POS Audit tab.

---

## 6. SEGMENT TIMELINE (XAI EVIDENCE)

The video is post-processed into 30-frame segments to provide a structured evidence timeline for the XAI output.

**Segment computation:**
```python
YOLO_SEG_FRAMES = 30   # one segment per 30 frames

for each segment:
    entries       = all (class, confidence) pairs for frames in this segment
    dominant      = highest-priority class (using _YOLO_PRIORITY weights)
    avg_conf      = mean confidence across all detections in segment
    max_shop_conf = max shoplifting confidence in segment (key field for evidence)
```

**Segment is used for:**
- XAI scatter chart (confidence over time) — marks each ≥50% shoplifting frame
- "Shoplifting Segments (≥50%)" metric card — shows how many of N segments had shoplifting signal
- Intent score `_n_segs` calculation (sustained signal indicator)
- YOLO class distribution bar chart in the timeline section

**Scatter chart reference lines:**
- 70% (red dashed) → SHOPLIFTING threshold
- 50% (orange dotted) → NEEDS REVIEW threshold

---

## 7. POS AUDIT MODULE

### 7.1 Design

The POS (Point of Sale) audit is a **mock terminal** that simulates real-world POS-behaviour data fusion. The operator enters items as they would appear in a POS transaction, and the system compares this against what the camera detected.

**The mismatch analysis works at the behavioural level, not product-instance level.** The behaviour model classifies person actions (Picking-Holding, shoplifting, Looking around) rather than identifying specific product SKUs. The system therefore compares:
- **Camera signals:** "Picking-Holding events", "LSTM shoplifting windows", "persons tracked"
- **POS records:** itemised transaction entered by operator

### 7.2 Product Catalog

The mock catalog contains 10 product entries (SKU, name, unit price):

| SKU | Product | Price |
|-----|---------|-------|
| ITEM001 | Snack Bar | £2.99 |
| ITEM002 | Soda Bottle | £1.99 |
| ITEM003 | Chocolate Box | £5.99 |
| ITEM004 | Energy Drink | £3.49 |
| ITEM005 | Chips Bag | £2.49 |
| ITEM006 | Candy Pack | £1.49 |
| ITEM007 | Gum Pack | £0.99 |
| ITEM008 | Protein Bar | £3.99 |
| ITEM009 | Water Bottle | £1.29 |
| ITEM010 | Coffee Can | £2.79 |

### 7.3 PersonProductTracker (from notebook Cell 8)

The `PersonProductTracker` class tracks behaviour-related state for a single person across frames:

**Fields tracked:**
- `direct_shoplifting_frames` — frames where class "shoplifting" detected
- `looking_around_frames` — frames where class "Looking around" detected
- `picking_holding_frames` — frames where class "Picking-Holding" detected
- Product state transitions (e.g. filled-basket → empty-basket without checkout)

**Proximity detection:** Person-product interaction is detected when the Euclidean distance between bounding box centres is less than 1.2× the person's bounding box height (arm's reach proxy).

### 7.4 Mismatch Detection

Three mismatch categories are flagged:
1. **UNSCANNED** — item detected by camera but not in POS transaction
2. **UNTOUCHED** — item in POS transaction but no camera interaction detected
3. **MATCH** — item in both POS and camera evidence

### 7.5 Session State

POS items are stored in Streamlit session state (`st.session_state.pos_items`) as a list of dictionaries `{sku, name, price, qty}`. This persists across reruns within the same browser session but resets when the page is fully reloaded.

---

## 8. LIVE PREVIEW SYSTEM

The live preview streams annotated frames to the browser during video processing, before the final verdict is known.

**Performance settings:**
- `_live_step = 15` — update preview every 15 frames (≈2fps at 30fps source)
- Preview downscaled to max 480px wide before sending to browser (reduces WebSocket payload ~4× for typical 1080p footage)
- `cv2.INTER_AREA` interpolation for downscaling (best for pixel averaging)

**Live badge gating:** The label shown on the live preview badge during processing uses YOLO confidence ≥ 50% as gate. If YOLO is below threshold, the badge falls back to the last known LSTM rolling classification.

**Rolling BiLSTM for live badge:** While frames are being processed, the BiLSTM is run every `LSTM_STRIDE // 2` features (stride 7) on the last 45 features, giving a rolling verdict that updates the badge during long videos.

**Bounding box consistency rule:** Shoplifting boxes are only drawn red when YOLO confidence ≥ 50%. Below 50%, the box is drawn green and the label shows "normal XX%" — this prevents the live preview showing a red shoplifting box when the final verdict will be NORMAL.

**Live timeline bar:** A colour-coded strip is built frame-by-frame: red for suspicious frames, green for normal. This gives a visual "heatmap" of when in the video suspicious behaviour was detected.

---

## 9. DESIGN EVOLUTION — CLASSIFICATION RULE ITERATIONS

The following iterations were tried before arriving at the current single-peak rule. This history is important for the thesis methodology section.

### Iteration 1 — Simple 55% threshold
A single frame with shoplifting confidence ≥ 55% → shoplifting verdict. **Problem:** One ambiguous frame could trigger a shoplifting verdict.

### Iteration 2 — Majority vote with 45% gate
BiLSTM majority vote. If >50% of LSTM windows voted shoplifting AND confidence ≥ 45% → shoplifting. **Problem:** BiLSTM always voted normal (training distribution imbalance).

### Iteration 3 — Sustained frames gate
Require ≥ 3 consecutive frames with shoplifting ≥ 55%. **Problem:** Normal videos with brief ambiguous poses still triggered.

### Iteration 4 — YOLO-only with segment voting
Remove BiLSTM from verdict. 30-frame segments, if any segment has peak ≥ 50% → shoplifting. **Problem:** Segments with brief 30% detections were being counted; inconsistency between live preview and final verdict.

### Iteration 5 — Segment threshold 50%
Segments with peak ≥ 50% only. **Problem:** Still inconsistency. Live preview showed red boxes but final output was normal (because bounding box colouring used 50% gate but verdict used segment counting with different logic).

### Iteration 6 — Current: Single peak threshold
`yolo_peak = max(all shoplifting frames)`. Three-way verdict: ≥70% SHOPLIFTING, 50–69% NEEDS REVIEW, <50% NORMAL. **Result:** Consistent, transparent, directly explainable.

---

## 10. XAI OUTPUT COMPONENTS

The system generates five distinct XAI components:

### 10.1 Detection Evidence Scatter Chart
- X-axis: time in seconds
- Y-axis: shoplifting confidence
- Marker colour: red ≥70%, orange 50–69%
- Reference lines at 70% and 50%
- Shows the temporal distribution of suspicious frames

### 10.2 Decision Reasoning Expander
Expandable section in plain English explaining:
- The decision rule (peak threshold)
- Peak confidence observed and resulting verdict
- Number of frames ≥50% and their temporal spread
- All three pipeline steps with counts

### 10.3 BiLSTM Advisory Context
Advisory (non-verdict) display showing:
- Total LSTM windows analysed
- Fraction classified as shoplifting by LSTM
- Per-behaviour-type window count and average confidence
- Explicit labelling: "advisory only — does not affect verdict"

### 10.4 Behaviour Timeline Chart
Gantt-style chart with two tracks:
- LSTM Windows track (per-window classification over time)
- YOLO Frames track (per-segment classification over time)
Both shown with colour coding by class for visual temporal comparison.

### 10.5 Forensic Case File (from notebook)
Structured JSON audit trail containing:
- UUID case_id
- ISO timestamp
- Video metadata (path, duration, fps, resolution, frame count)
- Per-window LSTM predictions
- Intent score components (decomposed)
- Bias flags
- POS comparison result
- Alert ID
- Mandatory human-review disclaimer
- Reference: GDPR Article 22 (automated decision-making)

---

## 11. BIAS AND FAIRNESS MODULE

**Fairness score:** Default 0.85 (high fairness — no demographic indicators available in video-only mode). Reduces to 0.50 if LSTM model is not trained (random weights).

**Bias flags generated when:**
- LSTM model not trained (`LSTM_PATH` does not exist)
- Score is uncertain (near threshold)

**Design principle:** The system operates exclusively on body pose and movement. No facial features, skin colour, age estimation, or other demographic proxies are used. This is documented as a deliberate design choice to minimise algorithmic bias.

**Fairness display:** Gauge chart showing fairness score 0–100%, with colour bands (red <50%, amber 50–75%, green >75%).

**From notebook Cell 10:** The bias-aware quality adjustment was designed as: if quality/confidence is low, scale down the intent score to prevent high-severity alerts from low-quality footage. `adj_score = raw_score * quality_multiplier`.

---

## 12. UI STRUCTURE (STREAMLIT)

The application is a single-page Streamlit app with a tab-based layout.

### Tab 1 — Model Performance
- BiLSTM/MobileNet training metrics display
- Confusion matrix visualisation (Plotly heatmap if JSON data available, else image file)
- Training learning curve (loss/accuracy over epochs)
- Loaded from: `models/bilstm_dw_info.json`, `models/mobilenet_dw_info.json`

### Tab 2 — Video Analysis
**Sub-sections:**

**Upload panel:**
- Supported formats: MP4, AVI, MOV, MKV
- Progress bar with per-stage status messages (10 stages, 0%→100%)
- Live preview streamed during processing
- Live stats panel: persons tracked, products detected, current label + confidence

**Analysis Results (shown after processing):**

For SHOPLIFTING:
- Red banner: "SHOPLIFTING DETECTED" with peak confidence
- Security Action Panel: recommended immediate steps + evidence checklist
- Advisory warning: mandatory before action

For NEEDS REVIEW:
- Orange banner: "NEEDS REVIEW" with peak confidence (50–69% range)
- Ambiguous signal warning

For NORMAL:
- Green banner: "NORMAL BEHAVIOR"
- Compact summary (4 metric cards: persons, products, frames, duration)
- Early return — no XAI sections shown for normal videos

**XAI sections (shown for shoplifting/review only):**
1. Detection Evidence (scatter chart + 4 metric cards)
2. Explainable AI — Decision Reasoning (expandable expander)
3. Risk Assessment (colour-coded severity banner)
4. Detection Statistics (4 metric cards)
5. Intent Score (gauge chart + component breakdown Gantt chart)
6. Behaviour Timeline (dual-track Gantt chart)
7. YOLO Class Distribution (segment breakdown metrics)
8. Forensic Evidence (GIF clips of suspicious moments)
9. Quality & Fairness (two gauge charts)
10. Video Information (metadata)
11. Advisory Summary (alert card)
12. Debug Info (expandable JSON dump for troubleshooting)

### Tab 3 — POS Audit
Two-panel layout:
- Left: Mock POS Terminal (form with selectbox + quantity input)
- Right: Behavioural Video Evidence (camera-detected signals)
- Bottom: Mismatch analysis table

### Tab 4 — Training Guide
Step-by-step Colab notebook usage instructions.

### Sidebar
- System status (YOLO / MobileNet / BiLSTM availability with ✓/✗)
- Model info (accuracy metrics from JSON if available)
- Quick tips

---

## 13. TECHNICAL STACK

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10–3.12 | Runtime |
| Streamlit | ≥1.32.0 | Web UI framework |
| OpenCV | ≥4.8.0 | Video I/O, frame annotation |
| PyTorch | (GPU auto-detected) | BiLSTM and MobileNetV2 inference |
| torchvision | (with PyTorch) | MobileNetV2 architecture + ImageNet weights |
| Ultralytics | ≥8.3.0 | YOLO inference + ByteTrack tracking |
| lapx | ≥0.5.2 | ByteTrack dependency |
| NumPy | ≥1.24.0 | Array operations |
| Pandas | ≥2.0.0 | POS transaction tables |
| Plotly | ≥5.15.0 | Interactive charts (scatter, Gantt, gauge) |
| Matplotlib | ≥3.7.0 | Training visualisations in notebook |
| scikit-learn | ≥1.3.0 | Evaluation metrics in notebook |
| ReportLab | ≥4.0.0 | PDF report generation |

### Compute
- GPU: CUDA auto-detected. If available: FP16 YOLO inference, `torch.backends.cudnn.benchmark=True`
- CPU fallback: Full float32, same results with slower inference
- Inference hardware tested on: Windows 10, Intel CPU, NVIDIA GPU (optional)

### Deployment
- Entry point: `run.py` → `streamlit run app.py`
- Port: 8501 (default)
- `--server.headless=true` for headless server deployment
- Containerised: `.devcontainer/` configuration included for VS Code Dev Containers

---

## 14. TRAINED MODELS (FILES)

| File | Description |
|------|-------------|
| `models/yolo26n.pt` | COCO-pretrained YOLOv8 nano base weights |
| `models/yolo26_dw_v2.pt` | Domain-adaptive fine-tune on 24-class retail dataset (preferred) |
| `models/yolo26_retail.pt` | Retail-only fine-tune (legacy fallback) |
| `models/mobilenet_dw.pt` | MobileNetV2 binary classifier checkpoint |
| `models/mobilenet_dw_info.json` | Training metrics: accuracy, confusion matrix, learning curve |
| `models/mobilenet_test_split.json` | Held-out test set paths + labels (for reproducible evaluation) |
| `models/bilstm_dw.pt` | BiLSTM + Attention classifier checkpoint |
| `models/bilstm_dw_info.json` | Training metrics: accuracy, confusion matrix, learning curve |
| `models/bilstm_test_split.json` | Held-out test set (for Cell 7 evaluation) |

---

## 15. DATA AND TRAINING

### Training Dataset
- **Video sources:** Retail surveillance footage (normal and shoplifting behaviours)
- **Annotation tool:** Roboflow (24-class retail annotation)
- **Frame extraction:** 6fps (FPS_TARGET) from behaviour videos
- **Person crops:** Extracted using YOLO bounding boxes, resized to 224×224

### Data Splits (all with seed=42 for reproducibility)
- 80% training / 10% validation / 10% test (stratified by class)
- Test set held out completely — only used in Cell 6 (MobileNetV2) and Cell 7 (BiLSTM)

### Class Imbalance Handling
- `WeightedRandomSampler` in both Cell 4 and Cell 5
- Ensures each training batch sees approximately balanced class representation

### Data Augmentation
- Standard ImageNet normalisation applied at inference and training
- Additional augmentations applied during MobileNetV2 training (standard torchvision transforms)

---

## 16. PERFORMANCE METRICS (REPORTED)

| Model | Metric | Value |
|-------|--------|-------|
| MobileNetV2 | Validation Accuracy | **97.7%** |
| BiLSTM | Validation Accuracy | **88.6%** |
| MobileNetV2 (Phase 1, RandomForest) | Accuracy | 80.0% |
| MobileNetV2 (Phase 1, RandomForest) | Precision | 80.0% |
| MobileNetV2 (Phase 1, RandomForest) | Recall | 80.0% |
| MobileNetV2 (Phase 1, RandomForest) | F1 | 80.0% |
| MobileNetV2 (Phase 1, RandomForest) | CV Accuracy | 81.5% ± 1.7% |

Note: The Phase 1 metrics (80%) are from the original Random Forest classifier on MediaPipe 21-feature hand-pose data. The Phase 2 (current) MobileNetV2 97.7% and BiLSTM 88.6% are from deep learning models trained on the full surveillance video dataset.

**Top 5 features from Phase 1 Random Forest (by importance):**
1. Right elbow angle (mean)
2. Body velocity (mean)
3. Left hand-body distance (mean)
4. Hand height relative to shoulders
5. Body displacement

---

## 17. PHASE 1 vs PHASE 2 COMPARISON

This is important for the thesis to articulate the system's evolution.

### Phase 1 (MVP — original design, now superseded)
- **Pose estimation:** MediaPipe — 33 body landmarks per frame
- **Feature extraction:** 21 hand/body features manually engineered
- **Classifier:** scikit-learn Random Forest
- **Detection:** Binary classification per sliding window (30-frame, stride 15)
- **Accuracy:** 80.0%
- **Architecture:** `src/` module directory (separate files per component)

### Phase 2 (Current — single self-contained app.py)
- **Object detection:** YOLO26n with ByteTrack (replaces MediaPipe)
- **Feature extraction:** MobileNetV2 1280-dim feature vectors (replaces manual features)
- **Classifier:** BiLSTM + Temporal Attention (replaces Random Forest)
- **Detection:** Three-stage cascade + single-peak rule
- **Accuracy:** 97.7% (MobileNetV2) / 88.6% (BiLSTM)
- **Architecture:** Self-contained `app.py` — no `src/` imports

### Key architectural difference
Phase 1 used MediaPipe for pose estimation (33 keypoints → manually engineered 21 features → Random Forest). This was intentionally lightweight and interpretable. Phase 2 replaces this with an end-to-end deep learning pipeline that achieves higher accuracy at the cost of requiring GPU for reasonable speed.

---

## 18. KNOWN LIMITATIONS

These should be discussed honestly in the thesis limitations section.

1. **BiLSTM training distribution imbalance:** The BiLSTM was observed to classify nearly all input as "normal" at very high confidence. Root cause: the training dataset likely has significantly more normal frames than shoplifting frames. Mitigation: demoted to advisory-only; `WeightedRandomSampler` added in training notebook.

2. **YOLO confidence calibration:** Low confidence threshold (YOLO_CONF=0.2) means borderline poses are captured, but also means normal poses at unusual angles may briefly trigger low-confidence shoplifting detections. The 50% classification gate prevents these from affecting the verdict.

3. **No real product identification:** The system detects product *interaction behaviours* (Picking-Holding) but cannot identify which specific product was handled. The POS audit comparison is therefore at the behavioural level, not the product-instance level.

4. **Mock POS only:** The current POS integration is a manual-entry simulation. Real integration would require webhook or API connection to a live POS system.

5. **Single-camera, single-customer limitation:** The system processes one video at a time. Multi-camera, multi-customer real-time processing is Phase 3 (not yet implemented).

6. **No temporal persistence across sessions:** Each video analysis is stateless. The system cannot track a customer across multiple store visits or across different cameras.

7. **Speed:** At 30fps source video with full annotation, processing is approximately 0.5–2× real-time depending on GPU availability. Not suitable for true real-time streaming without dedicated GPU.

---

## 19. ETHICAL FRAMEWORK

This section is explicitly important for the thesis.

### Built-in ethical constraints
1. **No facial recognition** — body pose only, no identity inference possible
2. **No demographic profiling** — age, gender, race are not used as features
3. **No personal data storage** — processing is stateless, no footage retained
4. **Advisory only** — every output has a mandatory human-review disclaimer
5. **Explainable by design** — every verdict includes a plain-English reasoning section

### GDPR relevance
The forensic case file (from notebook Cell 11) explicitly references GDPR Article 22 (automated decision-making). The system is designed so that no automated decision directly affects a person — all outputs are advisory inputs to a human decision.

### Fairness score mechanism
The 0–1 fairness score is displayed on every result. It is penalised when:
- The ML model is untrained (random weights → results are meaningless)
- External bias factors are present (logged as bias_flags)

### Vulnerable population handling (planned — Phase 2/3)
The README documents planned edge case handling:
- Children: detect child-sized poses, require adult association for alerts
- Elderly/confused customers: slower movement patterns, extended dwell time tolerance
- Items placed back before checkout: put-back gesture tracking

---

## 20. ACADEMIC CONTRIBUTIONS (ORIGINAL)

For the thesis contribution section, these are the novel elements:

1. **Behavioural and transactional data fusion** — combining YOLO behaviour signals with POS transaction records for a unified mismatch indicator. No equivalent system found in retail AI literature as of the project period.

2. **Structured forensic case file as first-class model output** — a per-incident JSON audit trail (case_id, timestamp, decomposed scores, bias flags, POS comparison, mandatory human-review clause). Notebook Cell 11 notes: "No reviewed paper generates a structured per-incident audit trail as a first-class model output" (confirmed by Perplexity literature search, March 2026).

3. **Decomposed intent scoring** — a four-component score (behaviour, concealment, POS mismatch, duration) that shows the operator exactly which factor drove the alert. Single-number risk scores provide no actionable information; decomposed scores enable targeted human review.

4. **Bias-aware quality adjustment** — integrating a fairness/quality multiplier into the scoring pipeline to prevent low-quality footage from generating high-severity alerts. The fairness score is reported alongside the intent score.

5. **Advisory-first design pattern** — the explicit architectural choice to make the system incapable of taking autonomous action. Every code path ends with a human-review requirement. This is a design contribution to human-in-the-loop AI.

6. **Three-way classification (SHOPLIFTING / NEEDS REVIEW / NORMAL)** — introducing a middle "ambiguous" category rather than forcing binary classification. This reduces false positive burden on security staff.

---

## 21. SYSTEM LIMITATIONS VS. PHASE 2 ROADMAP

| Feature | Current State | Phase 2/3 Target |
|---------|--------------|-----------------|
| Video processing | Offline (upload + analyse) | Live RTSP camera feed |
| POS integration | Manual mock entry | Real-time POS webhook/API |
| Multi-person | Tracked per video | Cross-camera identity persistence |
| Speed | ~0.5–2× real-time | ≥15fps real-time |
| Alerts | In-app display | Push/SMS/email to manager |
| Explainability | YOLO timeline + BiLSTM attention | SHAP for transactional data |
| Edge cases | Basic handling | Vulnerable group detection |
| Deployment | Local/Streamlit Cloud | Low-cost edge hardware |

---

## 22. GLOSSARY (FOR THESIS WRITING)

| Term | Definition |
|------|-----------|
| XAI | Explainable AI — AI outputs that include human-readable reasoning |
| YOLO | You Only Look Once — real-time object detection architecture |
| YOLOv8n | YOLOv8 nano — smallest/fastest variant of YOLOv8 |
| BiLSTM | Bidirectional LSTM — processes sequences forward AND backward |
| LSTM | Long Short-Term Memory — RNN variant that retains long-range context |
| ByteTrack | Multi-object tracking algorithm that assigns persistent IDs across frames |
| MobileNetV2 | Efficient CNN architecture designed for mobile/embedded inference |
| Temporal Attention | Learned weighting of sequence positions — produces interpretable frame importance |
| Temperature Scaling | Dividing logits by T>1 before softmax — reduces overconfident predictions |
| Intent Score | Weighted risk score 0.0–1.0 computed from behaviour + duration + POS signals |
| POS | Point of Sale — checkout terminal where transactions are recorded |
| SKU | Stock Keeping Unit — unique product identifier |
| Sliding Window | Moving sub-sequence of fixed length extracted from a longer sequence |
| GDPR Article 22 | EU regulation on automated decision-making affecting individuals |
| Human-in-the-Loop | System design where a human must review before any action is taken |
| feat_step | Feature extraction interval — MobileNetV2 runs every feat_step frames |
| yolo_step | YOLO inference interval — runs every yolo_step frames |
| WeightedRandomSampler | PyTorch sampler that up-samples minority class to balance training batches |

---

## 23. SUGGESTED THESIS CHAPTER STRUCTURE

Based on the system as built, a natural thesis structure would be:

1. **Introduction** — Problem statement (opaque retail AI, false accusations, POS-video silos), research questions, contributions overview
2. **Literature Review** — Existing retail surveillance systems, YOLO-based detection, LSTM temporal modelling, XAI in CV, bias in AI surveillance, GDPR implications
3. **System Design** — Advisory-first philosophy, three-stage pipeline architecture, three-way classification rationale, POS integration design
4. **Implementation** — YOLO fine-tuning methodology, MobileNetV2 transfer learning, BiLSTM + attention architecture, training data and splits, single-peak classification rule
5. **XAI Components** — Scatter chart evidence, decision reasoning, BiLSTM attention weights, temporal Gantt chart, forensic case file
6. **Evaluation** — MobileNetV2 97.7% val accuracy, BiLSTM 88.6% val accuracy, qualitative evaluation of XAI outputs, POS mismatch detection examples
7. **Bias and Fairness** — Design choices (no facial recognition), fairness scoring mechanism, decomposed intent score transparency, GDPR Article 22 compliance approach
8. **Discussion** — Classification rule evolution (9 iterations to single-peak), BiLSTM limitation (training distribution bias), comparison of Phase 1 (Random Forest 80%) vs Phase 2 (BiLSTM 88.6%)
9. **Conclusion and Future Work** — Current limitations, Phase 2/3 roadmap (real-time, multi-camera, vulnerable group handling)

---

*Document generated: 2026-04-12. Based on codebase at commit state: branch dev3.0. Primary source: app.py (~2180 lines), DigitalWitness_Pipeline.ipynb, README.md.*
