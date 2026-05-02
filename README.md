# Digital Witness — Branch: `dev3` / `main`

---

## Approach

> **Final, tested, working version. Clean modular pipeline — YOLO26 for detection, MobileNetV2 for spatial features, BiLSTM + Attention for temporal reasoning.**
> One explicit decision rule: if YOLO peak frame confidence >= 70%, verdict is SHOPLIFTING. BiLSTM is advisory and XAI only.
>
> **Stack:** YOLO26 (detect + track) → MobileNetV2 (1280-dim features) → BiLSTM + Attention (temporal) → Peak Confidence Rule (verdict)

> **Advisory system only. Does not determine guilt. All alerts require human review.**

---

## Pipeline Overview

```
Raw Video
   │
   ▼
[YOLO26]  ── every 2nd frame ──▶  4-class behavior + ByteTrack IDs
   │                                      │
   │                                      ▼
   │                            track persons + product pickups (proximity)
   │                            yolo_peak = max shoplifting conf across all frames
   │
   ▼
[MobileNetV2]  ── every 4th frame ──▶  1280-dim spatial feature vector
   │
   ▼
[BiLSTM + Attention]  ── sliding window (45 frames, stride 15) ──▶  advisory label + attention weights
   │
   ▼
[Decision Rule]  yolo_peak >= 0.70  ──▶  SHOPLIFTING
                 yolo_peak <  0.70  ──▶  NORMAL
   │
   ▼
[IntentScorer]  ──▶  risk score (0.0 – 1.0)
   │
   ▼
[BiasAssessor]  ──▶  fairness-adjusted score
   │
   ▼
[AlertGenerator + ClipExtractor + POS Audit]
```

---

## Decision Rule (Exact)

```python
if yolo_peak >= shop_threshold:   # shop_threshold = 0.70
    verdict = "shoplifting",  confidence = min(yolo_peak, 0.99)
else:
    verdict = "normal",       confidence = avg_normal_event_confidence
```

`yolo_peak` = highest shoplifting confidence YOLO assigned to any single frame.
BiLSTM output does **not** affect the verdict — it drives the XAI explanation only.

---

## Model Performance

| Model | Test Set | Accuracy | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|---|---|
| YOLO26 4-class | 247 images | 90.3% | 85.7% | 84.8% | 89.7% | 0.972 |
| MobileNetV2 | 9,363 frames | 99.5% | 99.5% | 99.97% | 99.3% | 0.9999 |
| BiLSTM + Attention | 27 sequences | 96.3% | 96.3% | 100% | 92.9% | 1.000 |

---

## Models

| File | Size | Role |
|---|---|---|
| `yolo26_dw_v2.pt` | 5.3 MB | Domain-adaptive fine-tune — preferred |
| `yolo26n.pt` | 5.5 MB | COCO base — automatic fallback |
| `yolo26_retail.pt` | 5.4 MB | Legacy retail fine-tune |
| `mobilenet_dw.pt` | 11.7 MB | Fine-tuned MobileNetV2 |
| `bilstm_dw.pt` | 19.4 MB | 2-layer BiLSTM + Attention |

### YOLO Behavior Classes

| Class | Description |
|---|---|
| `shoplifting` | Direct concealment gesture |
| `looking-around` | Suspicious reconnaissance |
| `picking-holding` | Product interaction |
| `normal` | Standard shopping behavior |

### BiLSTM Architecture

```
Input: (B, 45, 1280)
   |
BiLSTM(hidden=256, layers=2, bidirectional=True, dropout=0.3)
   -> 512-dim output per timestep
   |
Temporal Attention (Bahdanau-style softmax)
   -> attention weight per frame  (XAI signal)
   -> context vector = weighted sum
   |
Linear(512 -> 2)  ->  {normal, shoplifting}
Temperature scaling: T = 3.0
```

---

## Intent Score

| Path | Formula |
|---|---|
| Shoplifting | `peak_conf × (0.5 + 0.5 × min(suspicious_segments / 5, 1.0))` |
| Normal | `0.50 × concealment + 0.35 × bypass + 0.15 × duration` |

| Score | Severity |
|---|---|
| < 0.30 | NONE |
| 0.30 – 0.49 | LOW |
| 0.50 – 0.69 | MEDIUM |
| 0.70 – 0.84 | HIGH |
| >= 0.85 | CRITICAL |

Alert fires when adjusted score >= 0.50. If BiLSTM is untrained, score is halved (× 0.50) and fairness_score drops to 0.50.

---

## Key Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| `yolo_conf` | 0.20 | Detection confidence floor (low = better recall) |
| `yolo_iou` | 0.45 | NMS IoU threshold |
| `yolo_step` | 2 | Run YOLO every 2nd frame |
| `feat_step` | 4 | Extract features every 4th frame |
| `lstm_seq_len` | 45 | Frames per BiLSTM input window |
| `lstm_stride` | 15 | Sliding window stride (50% overlap) |
| `shop_threshold` | 0.70 | YOLO peak cutoff for SHOPLIFTING verdict |
| `imgsz` | 320 | YOLO input size (43% faster than 640) |

---

## Functional Test Suite (FR01 – FR09)

`functional_test_runner.py` validates the live system end-to-end:

| Test | What it checks |
|---|---|
| FR01 | Video upload accepted — pipeline runs, metadata extracted |
| FR02 | YOLO detects persons or behavior segments |
| FR03 | ByteTrack assigns integer track IDs |
| FR04 | POS mismatch flagged when picked_count > scanned_count |
| FR05 | Intent score in [0.0, 1.0], severity in valid set |
| FR06 | BiasAssessor halves score when BiLSTM is untrained |
| FR07 | PipelineResult contains all required fields |
| FR08 | JSON case file written to outputs/cases/ |
| FR09 | ClipExtractor produces >= 1 GIF with gif_bytes |

---

## UI — 4 Tabs (Streamlit)

| Tab | Contents |
|---|---|
| **Video Analysis** | Upload, live frame preview with YOLO boxes + classification badge, run button |
| **Store Security** | Risk level banner, detection stats, up to 4 forensic GIF clips, advisory alert |
| **Technical Analysis** | 5-step XAI: YOLO evidence → MobileNetV2 features → BiLSTM verdict → intent score breakdown → suspicious timeline |
| **POS Audit** | Mock POS item entry, behavioral evidence, UNSCANNED / UNTOUCHED / MATCH mismatch flags |

---

## Pipeline Stages (pipeline.py)

| Stage | What happens |
|---|---|
| 1–3 | Load YOLO, MobileNetV2, BiLSTM |
| 4 | Open video — extract FPS, resolution, frame count |
| 5 | Per-frame loop: YOLO every 2nd frame, features every 4th, live BiLSTM badge, annotate output |
| 6 | Sliding-window BiLSTM over full feature sequence |
| 7 | Build 30-frame YOLO segments (dominant class per block) |
| 8 | Apply decision rule → `yolo_peak >= 0.70` → verdict |
| 9 | IntentScorer → raw risk score |
| 10 | BiasAssessor → fairness-adjusted score + quality analysis |
| 11 | AlertGenerator → emit AlertRecord if adj_score >= 0.50 |
| 12 | ClipExtractor → animated GIFs of top suspicious windows |
| 13 | Return PipelineResult |

---

## Project Structure

```
Project_DigitalWitness/
├── app.py                         # Streamlit entry — session state, tab routing
├── run.py                         # Launch helper (python run.py)
├── requirements.txt
├── functional_test_runner.py      # FR01–FR09 test suite
├── DigitalWitness_Pipeline.ipynb  # Training notebook
├── evaluate_models.ipynb          # Model evaluation notebook
│
├── core/
│   ├── config.py                  # ModelPaths, PipelineParams, all constants
│   ├── model_definitions.py       # MobileNetExtractor, BiLSTMAttentionClassifier (nn.Module)
│   ├── model_registry.py          # Model file discovery + validation
│   ├── result_types.py            # Typed dataclasses: PipelineResult, AlertRecord, etc.
│   ├── pipeline.py                # AnalysisPipeline orchestration (main entry)
│   ├── video_processor.py         # VideoFrameIterator + YOLO frame annotation
│   ├── feature_extractor.py       # MobileNetV2 wrapper
│   ├── behaviour_classifier.py    # BiLSTM sliding-window inference
│   ├── intent_scorer.py           # Composite risk calculation
│   ├── bias_assessor.py           # Fairness adjustments
│   ├── alert_generator.py         # Rule-based alert generation
│   └── clip_extractor.py          # Animated GIF forensic clips
│
├── ui/
│   ├── analysis_view.py           # Store Security + Technical Analysis tabs
│   ├── pos_audit_view.py          # POS audit tab
│   ├── components.py              # Header, sidebar
│   └── styles.py                  # CSS injection
│
├── models/                        # Weights (.pt) + metric JSON files
├── outputs/
│   ├── eval/                      # Confusion matrices, ROC curves, all_models_eval.json
│   └── cases/                     # JSON case audit files
└── data/
    └── videos/{normal,shoplifting}/  # 90 normal + 30 shoplifting clips
```

---

## How to Run

```bash
pip install -r requirements.txt
python run.py
# Open http://localhost:8501
```

Requires Python 3.10–3.12. Model weights in `models/*.pt` must be present before running.

---

## Key Dependencies

| Library | Purpose |
|---|---|
| PyTorch (separate install) | BiLSTM training and inference |
| torchvision | MobileNetV2 backbone |
| ultralytics >= 8.3 | YOLO26 |
| lapx >= 0.5 | ByteTrack tracking |
| OpenCV 4.8+ | Frame extraction, annotation |
| Streamlit >= 1.32 | Web UI |
| ReportLab >= 4.0 | PDF report generation |
| scikit-learn 1.3+ | Metrics, evaluation |

---

## Branch Evolution

| Aspect | `ipd` | `dev2` | `dev3` / `main` |
|---|---|---|---|
| Code structure | Modular `src/` | Single `app.py` | Modular `core/` + `ui/` |
| CNN backbone | MobileNetV3-Small (576-dim) | MobileNetV2 (1280-dim) | MobileNetV2 (1280-dim) |
| LSTM window | 30 frames | 45 frames | 45 frames |
| Final verdict | BiLSTM majority | YOLO peak >= 0.70 | YOLO peak >= 0.70 |
| Intent score weights | 4-component | yolo_peak × sustained | yolo_peak × sustained |
| Testing | None | None | FR01–FR09 functional suite |
| Evaluation notebooks | None | None | evaluate_models.ipynb |
| Total code size | ~6,000 lines | ~1,250 lines | ~2,000 lines |
| Model total size | ~189 MB | ~38 MB | ~42 MB |

---

*IIT 2026 Final Year Project — advisory system only. Human review required before any action.*
