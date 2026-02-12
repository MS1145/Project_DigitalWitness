# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Digital Witness is a deep learning retail security assistant that detects potential shoplifting by analyzing behavioral patterns in video. It uses a **YOLO -> CNN -> LSTM** pipeline.

**Key principle:** The system does NOT determine guilt. It provides intent risk assessments with explainable evidence for human operators to review.

## Deep Learning Pipeline

```
Video -> YOLO -> CNN -> LSTM -> Intent Score -> Alert
          |       |      |
      Detection  Features  Classification
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| **YOLO** | YOLO26n | Object detection (persons, products) - 43% faster CPU inference |
| **CNN** | MobileNetV3-Small | Spatial feature extraction (576-dim) - lightweight, fast |
| **LSTM** | Bidirectional + Attention | Temporal behavior classification |

### Optimizations
- Frame skipping (process every Nth frame)
- Motion detection (skip static frames)
- Person gate (skip frames without people)
- Scene validation (detect non-retail/wrong footage early)

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run Application
```bash
python run.py              # Launch web interface (default)
python run.py --train      # Train LSTM model
python run.py --evaluate   # Evaluate model metrics
```

## Project Structure

```
Project_DigitalWitness/
├── run.py                 # Entry point
├── src/
│   ├── config.py          # All configuration
│   ├── main.py            # Pipeline orchestration
│   ├── ui/
│   │   └── app.py         # Streamlit web interface
│   ├── detection/
│   │   ├── yolo_detector.py
│   │   └── tracker.py
│   ├── models/
│   │   ├── cnn_feature_extractor.py
│   │   ├── lstm_classifier.py
│   │   ├── deep_pipeline.py
│   │   └── train_deep_model.py
│   ├── analysis/
│   │   ├── intent_scorer.py
│   │   ├── bias_aware_scorer.py
│   │   ├── alert_generator.py
│   │   └── quality_analyzer.py
│   ├── video/
│   │   ├── loader.py
│   │   └── clip_extractor.py
│   ├── pos/
│   │   ├── data_loader.py
│   │   └── mock_generator.py
│   └── output/
│       ├── case_builder.py
│       └── report_generator.py
├── models/
│   ├── lstm_classifier.pt # Trained LSTM
│   └── DigitalWitness_Optimized_Training.ipynb  # Colab training notebook
├── data/
│   ├── training/
│   │   ├── normal/        # Normal behavior videos
│   │   └── shoplifting/   # Shoplifting videos
│   └── videos/            # Test videos
└── outputs/
    ├── cases/
    ├── clips/
    └── reports/
```

## Key Configuration (`src/config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `YOLO_MODEL` | yolo26n.pt | Latest lightweight YOLO model |
| `CNN_BACKBONE` | mobilenet_v3_small | Feature extraction backbone |
| `CNN_FEATURE_DIM` | 576 | Feature vector size |
| `LSTM_HIDDEN_DIM` | 256 | LSTM hidden state |
| `LSTM_SEQUENCE_LENGTH` | 30 | Frames per sequence |
| `FRAME_SKIP` | 3 | Process every Nth frame |
| `ALERT_THRESHOLD` | 0.5 | Alert trigger threshold |

## Training

The notebook `models/DigitalWitness_Optimized_Training.ipynb` trains the LSTM classifier:
- Downloads UCF-Crime dataset via Kaggle API
- Compares CNN backbones (MobileNetV3, EfficientNetV2, SqueezeNet)
- Extracts CNN features with conditional processing
- Trains bidirectional LSTM with attention
- Includes checkpointing for Colab disconnects
- Saves model to `models/lstm_classifier.pt`

**Note:** YOLO26n and CNN (MobileNetV3) use pretrained weights (auto-downloaded). Only the LSTM is trained on your dataset.

## Classes

| Class | Description |
|-------|-------------|
| `normal` | Regular shopping behavior |
| `shoplifting` | Suspicious/theft behavior |

## Requirements

- Python 3.8+
- PyTorch, Ultralytics (YOLO), OpenCV, torchvision
- GPU recommended for faster inference (works on CPU with optimizations)
