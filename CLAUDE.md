# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Witness is a bias-aware, explainable retail security assistant that detects potential shoplifting by correlating behavioral video analysis with POS transaction data. It is an **advisory system** that supports human decision-making - all alerts require human validation.

**Key principle:** The system does NOT determine guilt. It provides intent risk assessments with explainable evidence for human operators to review.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Unified Entry Point (`run.py`)

All commands go through `run.py`:

```bash
python run.py --ui               # Launch web interface (recommended)
python run.py --train            # Train model on video dataset
python run.py                    # CLI demo mode
python run.py path/to/video.mp4  # Analyze specific video (CLI)
python run.py --help             # Show help
```

### Web Interface (Recommended)
```bash
python run.py --ui
```

Opens Streamlit dashboard at `http://localhost:8501` with:
- Video upload interface
- Real-time processing progress
- Visual results display (charts, timelines)
- Risk level alerts with explanations

### Train Model
```bash
python run.py --train
```

Training process:
- Processes videos from `data/training/normal/` and `data/training/shoplifting/`
- Extracts pose features using MediaPipe
- Trains a Random Forest classifier (binary: normal vs shoplifting)
- Generates confusion matrix and feature importance visualizations
- Saves model to `models/behavior_classifier.pkl`

### Direct Module Access (Advanced)
```bash
python -m src.main                          # Main pipeline
python -m src.pose.train_classifier         # Synthetic data training (fallback)
streamlit run app.py                        # Direct Streamlit
```

## Architecture

### 10-Step Analysis Pipeline (`src/main.py`)

1. **Load Video** - Extract frames, get metadata via OpenCV
2. **Pose Estimation** - MediaPipe landmarks (33 points per frame)
3. **Feature Extraction** - Sliding windows (30 frames, stride 15) → 21 features
4. **Behavior Classification** - RandomForest predicts: normal, pickup, concealment, bypass
5. **Load POS Data** - Parse JSON transactions (or generate mock data)
6. **Cross-Check** - Compare detected products vs billed items
7. **Intent Score** - Weighted multi-factor risk calculation (0.0-1.0)
8. **Generate Alert** - Threshold-based (default ≥0.5) advisory alert
9. **Extract Clips** - Forensic video evidence around suspicious events
10. **Build Case File** - JSON audit trail with all evidence

### Module Structure

```
src/
├── config.py              # All configuration constants
├── main.py                # Pipeline orchestration
├── video/                 # Video I/O and clip extraction
│   ├── loader.py          # VideoLoader context manager
│   └── clip_extractor.py  # Forensic clip generation
├── pose/                  # Pose estimation and behavior ML
│   ├── estimator.py       # MediaPipe pose detection
│   ├── feature_extractor.py  # 21-feature vector from pose sequences
│   ├── behavior_classifier.py  # RandomForest classification
│   └── train_classifier.py     # Training script
├── pos/                   # POS transaction handling
│   ├── data_loader.py     # JSON parser
│   └── mock_generator.py  # Test data generation
├── analysis/              # Core analysis logic
│   ├── cross_checker.py   # Video-to-POS reconciliation
│   ├── intent_scorer.py   # Multi-factor risk scoring
│   └── alert_generator.py # Advisory alert creation
└── output/
    └── case_builder.py    # Case file serialization
```

### Key Configuration (`src/config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SLIDING_WINDOW_SIZE` | 30 frames | Feature extraction window |
| `SLIDING_WINDOW_STRIDE` | 15 frames | Window overlap |
| `BEHAVIOR_CLASSES` | normal, pickup, concealment, bypass | ML targets |
| `WEIGHT_DISCREPANCY` | 0.4 | Intent score weight |
| `WEIGHT_CONCEALMENT` | 0.3 | Intent score weight |
| `WEIGHT_BYPASS` | 0.2 | Intent score weight |
| `WEIGHT_DURATION` | 0.1 | Intent score weight |
| `ALERT_THRESHOLD` | 0.5 | Minimum score to generate alert |

### Data Flow

- **Training Data:** Videos in `data/training/normal/` and `data/training/shoplifting/`
- **Input Videos:** Video files in `data/videos/`, POS JSON in `data/pos/`
- **Model:** Trained classifier in `models/behavior_classifier.pkl`
- **Output:** Case files in `outputs/cases/`, forensic clips in `outputs/clips/`

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

## Design Patterns

- **Context Managers:** `VideoLoader` and `PoseEstimator` use `with` statements for resource cleanup
- **Dataclasses:** Type-safe structures throughout (VideoMetadata, BehaviorEvent, Transaction, etc.)
- **Centralized Config:** All constants in `src/config.py`
- **Explainability First:** All scores include component breakdowns and human-readable explanations
- **Human-in-Loop:** Alerts always marked `requires_human_review: True`

## MVP Limitations

- Pre-recorded videos only (no live feed)
- Mock POS data for testing
- 4 behavior classes only
- No bias mitigation (measurement only)
- Single-machine processing
