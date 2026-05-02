# Digital Witness — Branch: `dev`

---

## Approach

> **Multi-stage deep learning pipeline for retail shoplifting detection from CCTV footage.**
> The system flags suspicious behavior for human review — it does not determine guilt or enforce action autonomously.
>
> **Stack:** YOLO26n (frame-level detection) → EfficientNetV2-S (spatial features) → BiLSTM + Attention (temporal classification) → Intent Score (risk aggregation)

---

## Pipeline Overview

```
Raw Video
   │
   ▼
[YOLO26n]  ──── every 2nd frame ────▶  bounding boxes + behavior class (4 classes)
   │                                         │
   │                                         ▼
   │                               ByteTrack multi-person tracking
   │
   ▼
[EfficientNetV2-S]  ── every 4th frame ──▶  1280-dim spatial feature vector
   │
   ▼
[BiLSTM + Temporal Attention]  ── sliding window (45 frames / 7.5s) ──▶  shoplifting probability
   │
   ▼
[Intent Score]  ──▶  risk level (NONE → LOW → MEDIUM → HIGH → CRITICAL)
   │
   ▼
[POS Audit]  ──▶  cross-validate items handled vs. items scanned
```

---

## Models

| Model | Role | Input | Output | Val Accuracy |
|---|---|---|---|---|
| YOLO26n (`yolo_dw.pt`) | Behavior detection per frame | 640×640 video frame | Bounding boxes + class + confidence | mAP50 ≥ 0.75 |
| EfficientNetV2-S (`efficientnet_dw.pt`) | Spatial feature extraction | 224×224 RGB frame | 1280-dim vector | **99.53%** |
| BiLSTM + Attention (`bilstm_dw.pt`) | Temporal sequence classification | 45-frame window of 1280-dim features | Shoplifting probability + attention weights | **94.39%** |

### YOLO Behavior Classes

| Class | Description |
|---|---|
| `normal` | Standard shopping behavior |
| `looking-around` | Browsing / scanning environment |
| `picking-holding` | Handling products |
| `shoplifting` | Suspicious concealment behavior |

### BiLSTM Performance (held-out test set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.86 | 1.00 | 0.92 |
| Shoplifting | 1.00 | 0.92 | 0.96 |

---

## Intent Score Formula

```
Intent Score = 0.50 × concealment + 0.35 × duration_fraction + 0.15 × count_ratio
```

| Score Range | Risk Level |
|---|---|
| 0.00 – 0.30 | NONE |
| 0.30 – 0.50 | LOW |
| 0.50 – 0.70 | MEDIUM |
| 0.70 – 0.85 | HIGH |
| 0.85 – 1.00 | CRITICAL |

**YOLO Override:** If YOLO peak confidence ≥ 0.60, or mean ≥ 0.50 across ≥ 3 frames → forced shoplifting classification.

---

## Training Data

| Split | Source | Frames |
|---|---|---|
| Normal | UCF-Crime dataset (capped 500 videos) | ~32K frames |
| Shoplifting | Kaggle kipshidze dataset | ~61K frames |
| YOLO fine-tune | Roboflow `shopliftingvideo+handpocket` | train / valid / test splits |

- **Extraction rate:** 6 FPS (captures concealment events ≥ 0.17s)
- **Class balancing:** `WeightedRandomSampler` used in all training loops
- **Feature cache:** `/data/feat_cache.pt` (~200 MB) stores pre-computed 1280-dim vectors — gives ~64× speedup to BiLSTM training

---

## Training Pipeline (Notebook Cells)

| Cell | Task |
|---|---|
| 1 | Environment setup — PyTorch / CUDA check |
| 2 | All hyperparameters + paths in one config block |
| 3 | YOLO26n fine-tuning (30 epochs, freeze backbone first 10) |
| 4 | Frame extraction → pseudo-label via YOLO → EfficientNetV2-S (25 epochs) |
| 5 | BiLSTM training on cached features (20 epochs, early stopping patience=7) |
| 6 | Evaluation — confusion matrix, XAI attention plot |
| 7 | POS integration — generate mock transaction data |
| 8 | End-to-end smoke test on a single video |

**Hardware target:** RTX 3070 Laptop (8.6 GB VRAM) — YOLO ~15 min, EfficientNet ~15 min, BiLSTM ~10 min.

---

## Runtime Architecture

Two threads run concurrently to keep the UI responsive:

```
Main Thread                         Background Thread
──────────────────────────────      ──────────────────────────────
Read frame from video               Pop frame from queue
Run YOLO (every 2nd frame)    ──▶   EfficientNetV2 → 1280-dim feature
Draw bounding boxes                 Update rolling BiLSTM window
Push frame to queue (every 4th)     BiLSTM predict + attention weights
Update Streamlit UI                 Write result to shared_status dict
```

---

## UI (Streamlit — `app.py`)

| Tab | Contents |
|---|---|
| Video Analysis | Upload video, live frame preview, person/product tracking stats, annotated output, forensic GIF clips |
| Model Performance | Val metrics, training curves, confusion matrices, architecture JSON |
| POS Audit | Mock POS transactions, handled-vs-scanned item comparison, mismatch flags |

---

## Project Files

```
Project_DigitalWitness/
├── app.py                        # Streamlit dashboard (~1250 lines)
├── DigitalWitness_Pipeline.ipynb # Full training notebook (8 cells)
├── run.py                        # Streamlit launcher (port 8501)
├── pos_integration.py            # Mock POS database
├── models/
│   ├── yolo26n.pt                # Base YOLO26n (5.3 MB)
│   ├── yolo_dw.pt                # Fine-tuned YOLO (5.2 MB)
│   ├── efficientnet_dw.pt        # Trained EfficientNetV2-S (81 MB)
│   ├── bilstm_dw.pt              # Trained BiLSTM (97 MB)
│   └── *.json                    # Model metadata + pseudo-labels
├── data/
│   ├── dataset/                  # Roboflow YOLO data + data.yaml
│   ├── frames/{normal,shoplifting}/  # 93K extracted frames
│   └── feat_cache.pt             # Pre-computed CNN features (~200 MB)
└── outputs/cases/                # Output cases + visualizations
```

---

## Key Dependencies

| Library | Purpose |
|---|---|
| PyTorch 2.5.1+cu121 | Model training & inference |
| Ultralytics | YOLO26n |
| torchvision | EfficientNetV2-S backbone |
| OpenCV | Frame extraction, annotation |
| Streamlit | Web dashboard |
| scikit-learn | Metrics, data splitting |

---

*IIT 2026 Final Year Project — advisory system only. Human review required before any action.*
