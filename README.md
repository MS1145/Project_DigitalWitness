# Digital Witness — Branch: `dev2`

---

## Approach

> **Three-stage cascade pipeline for retail shoplifting detection, with a single-rule final verdict based on YOLO peak confidence.**
> BiLSTM is advisory/explainability only — the final classification uses one transparent threshold, not a voting mechanism.
>
> **Stack:** YOLO26n (per-frame behavior) → MobileNetV2 (spatial features) → BiLSTM + Attention (temporal XAI) → Single Peak Rule (verdict)

> **Advisory system only. Does not determine guilt. All alerts require human review.**

---

## Pipeline Overview

```
Raw Video
   │
   ▼
[YOLO26n]  ── every 2nd frame ──▶  behavior class (4 classes) + ByteTrack IDs
   │                                         │
   │                                         ▼
   │                               yolo_peak = max shoplifting conf across all frames
   │
   ▼
[MobileNetV2]  ── every 4th frame ──▶  1280-dim spatial feature vector
   │
   ▼
[BiLSTM + Temporal Attention]  ── sliding window (45 frames) ──▶  advisory confidence + attention weights (XAI)
   │
   ▼
[Single Peak Rule]  ──▶  SHOPLIFTING / NEEDS REVIEW / NORMAL
   │
   ▼
[Intent Score + POS Audit + Case File]
```

---

## Final Verdict Rule (v3.0)

The verdict is determined by one explicit threshold — no voting, no accumulated conditions:

| `yolo_peak` | Verdict | UI |
|---|---|---|
| >= 0.70 | **SHOPLIFTING** | Red banner + security action panel |
| 0.50 – 0.69 | **NEEDS REVIEW** | Orange banner |
| < 0.50 | **NORMAL** | Green banner |

> `yolo_peak` = highest shoplifting confidence YOLO assigned to any single frame in the video.

This rule was chosen after 5 previous iterations caused inconsistency between the live preview and the final verdict. One threshold is auditable and explainable to non-technical security staff.

---

## Models

| Model | Role | Input | Output | Test Accuracy |
|---|---|---|---|---|
| YOLO26n (`yolo26_dw_v2.pt`) | Per-frame behavior detection | 320×320 frame | Bounding boxes + 4-class behavior | mAP50 >= 0.75 |
| MobileNetV2 (`mobilenet_dw.pt`) | Spatial feature extraction | 224×224 frame | 1280-dim vector | **99.53%** (9,363 samples) |
| BiLSTM + Attention (`bilstm_dw.pt`) | Temporal reasoning / XAI | 45 x 1280-dim window | Advisory probability + attention weights | **96.3%** (27 sequences) |

### YOLO Model Priority (automatic fallback)

| Priority | File | Description |
|---|---|---|
| 1 (preferred) | `yolo26_dw_v2.pt` | Domain-adaptive fine-tune on retail dataset |
| 2 (fallback) | `yolo26n.pt` | COCO base — reliable person detection |
| 3 (legacy) | `yolo26_retail.pt` | Earlier retail-only fine-tune |

### YOLO Behavior Classes

| Class | Priority | Description |
|---|---|---|
| `shoplifting` | 3 | Direct concealment gesture |
| `Looking around` | 2 | Suspicious reconnaissance |
| `Picking-Holding` | 1 | Product interaction / potential concealment |
| `normal` | 0 | Standard shopping behavior |

### Model Performance Detail

| Model | Metric | Value |
|---|---|---|
| MobileNetV2 | Val Accuracy | 97.7% |
| MobileNetV2 | Test Accuracy | 99.53% |
| MobileNetV2 | Shoplifting Precision | 99.87% |
| MobileNetV2 | Shoplifting Recall | 97.62% |
| BiLSTM | Val Accuracy | 88.6% |
| BiLSTM | Test Accuracy | 96.3% |
| BiLSTM | Shoplifting Precision | 96.56% |
| BiLSTM | Shoplifting Recall | 100% |

---

## BiLSTM Architecture

```
Input: 45 x 1280-dim feature sequence  (45 frames = ~6s at 30fps)
   |
BiLSTM(hidden=256, layers=2, bidirectional=True, dropout=0.3)
   -> 512-dim output per timestep
   |
Temporal Attention: Linear(512->256) -> Tanh -> Linear(256->1) -> Softmax
   -> attention weight per frame  (XAI signal — shows which frames drove the decision)
   -> context vector = weighted sum of LSTM outputs
   |
Classifier: Linear(512 -> 2)  ->  {normal, shoplifting}
```

- Temperature scaling `T=3.0` softens overconfident outputs
- Sliding window stride: 15 features (50% overlap)
- BiLSTM verdict is **advisory only** — final verdict comes from YOLO peak rule

---

## Intent Score

Calculated only when verdict is SHOPLIFTING or NEEDS REVIEW:

```
_sustained = min(count of 30-frame segments with shop_conf >= 0.50 / 5.0, 1.0)
raw_score  = min(yolo_peak x (0.5 + 0.5 x _sustained), 1.0)
```

| Score | Severity |
|---|---|
| < 0.30 | NONE |
| 0.30 – 0.49 | LOW |
| 0.50 – 0.69 | MEDIUM |
| 0.70 – 0.84 | HIGH |
| >= 0.85 | CRITICAL |

---

## Training Pipeline (Notebook — 7 Cells)

| Cell | Task | Key Details |
|---|---|---|
| 1 | Dependencies | PyTorch, OpenCV, ultralytics, torchvision |
| 2 | Config | All hyperparameters + paths; GPU detection; SMOKE_TEST mode |
| 3 | YOLO fine-tuning | Freeze backbone (`freeze=22`), train detection head only; label smoothing 0.1 |
| 4 | MobileNetV2 training | 6 FPS frame extraction; WeightedRandomSampler; 70/15/15 stratified split |
| 5 | BiLSTM training | Feature sequences from MobileNetV2; early stopping; WeightedRandomSampler |
| 6 | MobileNetV2 evaluation | 9,363 held-out frames; confusion matrix + per-class metrics |
| 7 | BiLSTM evaluation + XAI | 27 held-out sequences; attention weight bar chart |

---

## Runtime Architecture (app.py)

Two threads run concurrently to keep UI responsive:

```
Main Thread (Video I/O + UI)          Background Thread (CNN + BiLSTM)
------------------------------------  -----------------------------------
Read frame                            Pop frame from queue
Run YOLO (every 2nd frame)   ---->    MobileNetV2 -> 1280-dim feature
Draw annotations                      Update rolling BiLSTM window
Push frame to queue (every 4th)       BiLSTM predict + attention weights
Update Streamlit live badge           Write to shared_status dict (non-blocking)
```

10 pipeline stages: model init → video load → YOLO → MobileNetV2 → BiLSTM live → proximity detection → BiLSTM sliding window → intent score → alert → clip extraction.

---

## UI — 4 Tabs (Streamlit)

| Tab | Contents |
|---|---|
| **Model Performance** | MobileNetV2 + BiLSTM metrics, confusion matrices, learning curves |
| **Video Analysis** | Upload, live preview, annotated output, risk banner, XAI confidence chart, behavior timeline, forensic GIF clips |
| **POS Audit** | Product catalog, operator enters transaction, system flags UNSCANNED / UNTOUCHED / MATCH |
| **Case File** | Structured JSON audit trail with all analysis data + GDPR Article 22 disclaimer |

---

## Training Data

| Split | Source | Volume |
|---|---|---|
| Normal | UCF-Crime dataset | ~32K frames @ 6fps |
| Shoplifting | Kaggle dataset | ~61K frames @ 6fps |
| YOLO fine-tune | Roboflow 24-class retail | train / valid / test splits |

Class balancing via `WeightedRandomSampler` in all training loops.

---

## Project Files

```
Project_DigitalWitness/
├── app.py                           # Streamlit dashboard + full inference pipeline
├── run.py                           # Entry point — launches Streamlit (port 8501)
├── DigitalWitness_Pipeline.ipynb    # Full training notebook (7 cells)
├── requirements.txt                 # Python dependencies
├── THESIS_CONTEXT.md                # Full architectural decisions + iteration history
├── models/
│   ├── yolo26n.pt                   # COCO base (5.3 MB)
│   ├── yolo26_dw_v2.pt              # Domain-adaptive fine-tune (5.2 MB)
│   ├── mobilenet_dw.pt              # Trained MobileNetV2 (12 MB)
│   ├── bilstm_dw.pt                 # Trained BiLSTM (19 MB)
│   └── *_info.json / *_eval.json    # Metrics metadata
├── data/
│   ├── dataset/                     # Roboflow YOLO annotations
│   ├── videos/{normal,shoplifting}/ # Source videos
│   └── sequences/                   # Pre-computed feature sequences
├── frames/{normal,shoplifting}/     # Extracted frames @ 6fps
└── outputs/
    ├── cases/                       # JSON case files
    └── clips/                       # Forensic GIF extracts
```

---

## Key Dependencies

| Library | Purpose |
|---|---|
| PyTorch 2.5.1+cu121 | Training and inference |
| Ultralytics | YOLO26n |
| torchvision | MobileNetV2 backbone |
| OpenCV | Frame extraction, annotation |
| Streamlit | Web dashboard |
| scikit-learn | Metrics, stratified splitting |

**Hardware target:** RTX 3070 Laptop (8.6 GB VRAM) — CPU fallback supported.

---

## dev2 vs dev — Key Differences

| Aspect | `dev2` (this branch) | `dev` |
|---|---|---|
| Feature extractor | MobileNetV2 (12 MB) | EfficientNetV2-S (81 MB) |
| Final verdict | Single YOLO peak threshold | BiLSTM majority + YOLO override |
| BiLSTM role | Advisory / XAI only | Primary verdict signal |
| BiLSTM model size | 19 MB | 97 MB |
| YOLO input size | 320×320 (faster) | 640×640 |
| UI tabs | 4 (includes Case File) | 3 |

---

*IIT 2026 Final Year Project — advisory system only. Human review required before any action.*
