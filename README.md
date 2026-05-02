# Digital Witness — Retail Security Assistant

A deep learning system that detects potential shoplifting behaviour in retail surveillance video. Built as a Final Year Project (BSc Computer Science, 2026).

## What It Does

Upload a surveillance video clip. The pipeline runs three models in sequence and returns a verdict (normal / shoplifting), a risk score, forensic GIF clips of suspicious moments, and a POS transaction audit.

**It is an advisory system.** Every alert requires human validation before any action is taken.

## Pipeline Overview

```
Video → YOLO26 (detect + track) → MobileNetV2 (feature extract) → BiLSTM+Attention (classify)
                                                                          ↓
                                                    Intent Score → Bias Check → Alert
```

1. **YOLO26** (fine-tuned, 4-class) — detects and tracks persons and products frame-by-frame using ByteTrack. Classes: `looking-around`, `picking-holding`, `normal`, `shoplifting`.
2. **MobileNetV2** (fine-tuned) — extracts a 1280-dimensional spatial feature vector every 4th frame.
3. **BiLSTM + Temporal Attention** — classifies overlapping 45-frame windows of features into `normal` or `shoplifting`.

**Decision rule:** YOLO peak frame confidence ≥ 70% → SHOPLIFTING verdict.

## Model Performance

| Model | Test Size | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| YOLO 4-class | 247 images | 90.3% | 85.7% | 0.972 |
| MobileNetV2 | 9,363 frames | 99.5% | 99.6% | 0.9999 |
| BiLSTM+Attention | 27 sequences | 96.3% | 96.3% | 1.000 |

## How to Run

```bash
pip install -r requirements.txt
python run.py
# Open http://localhost:8501
```

Requires Python 3.10–3.12. PyTorch must be installed separately (see `requirements.txt`).

Model weights (`models/*.pt`) must be present before running.

## Project Structure

```
app.py                  — Streamlit entry point
run.py                  — Launch helper
core/
  pipeline.py           — End-to-end orchestration
  video_processor.py    — OpenCV frame iteration + YOLO tracking
  feature_extractor.py  — MobileNetV2 wrapper
  behaviour_classifier.py — BiLSTM sliding-window inference
  intent_scorer.py      — Composite risk scoring
  bias_assessor.py      — Fairness / reliability adjustment
  alert_generator.py    — Rule-based alert creation
  clip_extractor.py     — Animated GIF forensic clips
  model_definitions.py  — PyTorch nn.Module definitions
  model_registry.py     — Model file discovery
  config.py             — All constants and paths
  result_types.py       — Typed pipeline output objects
ui/
  analysis_view.py      — Results rendering (Store Security + Technical Analysis tabs)
  pos_audit_view.py     — Mock POS terminal and mismatch audit
  components.py         — Header and sidebar
  styles.py             — Application CSS
models/                 — Model weights (.pt) and metric JSON files
outputs/eval/           — Confusion matrices, ROC curves
functional_test_runner.py — FR01–FR09 functional test suite
```

## Advisory Notice

This system is a research prototype. It does not determine guilt. All outputs require human review before any real-world action.
