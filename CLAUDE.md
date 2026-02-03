# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Witness is a **deep learning** retail security assistant that detects potential shoplifting by correlating behavioral video analysis with POS transaction data. It uses a **YOLO → CNN → LSTM** pipeline for end-to-end behavior analysis.

**Key principle:** The system does NOT determine guilt. It provides intent risk assessments with explainable evidence for human operators to review.

## Deep Learning Pipeline

```
Video → YOLO → CNN → LSTM → Intent Score → Alert
         ↓       ↓      ↓
     Detection  Features  Classification
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| **YOLO** | YOLOv8 | Object detection & tracking (persons, products) |
| **CNN** | ResNet18 | Spatial feature extraction from frames |
| **LSTM** | Bidirectional LSTM + Attention | Temporal behavior classification |

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Unified Entry Point (`run.py`)

```bash
python run.py              # Run demo mode (simulated data)
python run.py --ui         # Launch web interface
python run.py --train      # Train LSTM model on video dataset
python run.py <video.mp4>  # Analyze specific video
python run.py --help       # Show help
```

### Train Model
```bash
python run.py --train
```

Training process:
- Processes videos from `data/training/normal/` and `data/training/shoplifting/`
- Extracts CNN features (ResNet18) from video frames
- Trains bidirectional LSTM classifier with attention
- Saves model to `models/lstm_classifier.pt`

## Architecture

### 14-Step Analysis Pipeline (`src/main.py`)

1. **Initialize Models** - Load YOLO, CNN, LSTM
2. **YOLO Detection** - Detect persons and products in frames
3. **Object Tracking** - Track objects across frames (ByteTrack)
4. **Interaction Detection** - Detect person-product interactions
5. **CNN Features** - Extract spatial features per frame
6. **LSTM Classification** - Classify behavior sequences
7. **Quality Analysis** - Assess video quality for reliability
8. **Load POS Data** - Parse JSON transactions
9. **Cross-Check** - Compare detected products vs billed items
10. **Bias-Aware Scoring** - Calculate fairness-adjusted intent score
11. **Edge Case Handling** - Handle occlusion, low confidence, etc.
12. **Generate Alert** - Create advisory alert if threshold exceeded
13. **Build Case File** - JSON audit trail with all evidence
14. **Forensic Package** - PDF report, annotated clips, screenshots

### Module Structure

```
src/
├── config.py              # All configuration constants
├── main.py                # Pipeline orchestration
│
├── ui/                    # Streamlit Web Interface
│   └── app.py             # Web dashboard (run via: python run.py --ui)
│
├── detection/             # YOLO Object Detection
│   ├── yolo_detector.py   # YOLOv8 person/product detection
│   └── tracker.py         # Multi-object tracking
│
├── models/                # Deep Learning Models
│   ├── cnn_feature_extractor.py  # ResNet18 spatial features
│   ├── lstm_classifier.py        # Bidirectional LSTM + attention
│   ├── deep_pipeline.py          # YOLO → CNN → LSTM orchestration
│   └── train_deep_model.py       # Training script
│
├── video/                 # Video Processing
│   ├── loader.py          # VideoLoader context manager
│   ├── clip_extractor.py  # Forensic clip extraction
│   └── annotated_clip_generator.py  # Clips with overlays
│
├── pose/                  # Pose Estimation (for quality analysis)
│   ├── estimator.py       # MediaPipe pose detection
│   └── behavior_classifier.py  # BehaviorEvent dataclass
│
├── pos/                   # POS Transaction Handling
│   ├── data_loader.py     # JSON parser
│   └── mock_generator.py  # Test data generation
│
├── analysis/              # Analysis & Scoring
│   ├── cross_checker.py   # Video-to-POS reconciliation
│   ├── intent_scorer.py   # Multi-factor risk scoring
│   ├── bias_aware_scorer.py  # Bias-adjusted scoring
│   ├── bias_detector.py   # Fairness metrics
│   ├── edge_case_handler.py  # Edge case handling
│   ├── quality_analyzer.py   # Video quality analysis
│   └── alert_generator.py    # Advisory alert creation
│
└── output/                # Output Generation
    ├── case_builder.py    # Case file serialization
    ├── report_generator.py   # PDF forensic reports
    └── evidence_compiler.py  # Forensic package assembly
```

### Key Configuration (`src/config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `YOLO_CONF_THRESHOLD` | 0.5 | Minimum detection confidence |
| `CNN_BACKBONE` | resnet18 | CNN architecture |
| `CNN_FEATURE_DIM` | 512 | Feature vector dimension |
| `LSTM_HIDDEN_DIM` | 256 | LSTM hidden state size |
| `LSTM_NUM_LAYERS` | 2 | Number of LSTM layers |
| `LSTM_SEQUENCE_LENGTH` | 30 | Frames per sequence |
| `ALERT_THRESHOLD` | 0.5 | Minimum score for alert |

### Data Flow

- **Training Data:** Videos in `data/training/normal/` and `data/training/shoplifting/`
- **Input Videos:** Video files in `data/videos/`, POS JSON in `data/pos/`
- **Models:**
  - YOLO: `models/yolov8n.pt` (auto-downloaded)
  - CNN: ResNet18 (pretrained ImageNet)
  - LSTM: `models/lstm_classifier.pt` (trained)
- **Output:**
  - Case files in `outputs/cases/`
  - Forensic clips in `outputs/clips/`
  - PDF reports in `outputs/reports/`
  - Forensic packages in `outputs/forensic_packages/`

### POS JSON Format
```json
{
  "transactions": [{
    "transaction_id": "TXN001",
    "timestamp": "2026-01-19T20:17:00",
    "items": [{"sku": "ITEM001", "name": "Product", "quantity": 1, "price": 2.99}],
    "total": 2.99,
    "payment_method": "card"
  }]
}
```

## Key Features

| Feature | Description |
|---------|-------------|
| **YOLO Detection** | Detects persons, products, bags, bottles, etc. |
| **Multi-Object Tracking** | Maintains object identity across frames |
| **Interaction Detection** | Detects pickup, hold, conceal, approach |
| **CNN Features** | Extracts 512-dim spatial features per frame |
| **LSTM Classification** | Classifies: normal, pickup, concealment, bypass |
| **Attention Mechanism** | Highlights important temporal segments |
| **Bias Detection** | Identifies potential bias in predictions |
| **Quality Analysis** | Assesses video quality for reliability |
| **Edge Case Handling** | Handles occlusion, low confidence |
| **Forensic Reports** | PDF reports with evidence |
| **Annotated Clips** | Video clips with pose overlays |

## Design Patterns

- **Context Managers:** `VideoLoader`, `PoseEstimator`, `DeepPipeline` use `with` statements
- **Lazy Initialization:** Deep learning models load only when needed
- **Dataclasses:** Type-safe structures (Detection, IntentPrediction, etc.)
- **Centralized Config:** All constants in `src/config.py`
- **Explainability First:** All scores include component breakdowns
- **Human-in-Loop:** Alerts always require human review

## Requirements

- Python 3.8+
- GPU recommended (CUDA) for faster inference
- ~4GB RAM minimum
- Dependencies: PyTorch, Ultralytics (YOLO), OpenCV, MediaPipe

## Current Limitations

- Pre-recorded videos only (no live feed)
- Mock POS data (no real POS integration)
- 4 behavior classes (normal, pickup, concealment, bypass)
- Single-machine processing
