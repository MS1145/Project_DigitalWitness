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
| **YOLO** | YOLOv8n | Object detection (persons, products) |
| **CNN** | ResNet18 | Spatial feature extraction (512-dim) |
| **LSTM** | Bidirectional + Attention | Temporal behavior classification |

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
│   ├── yolov8n.pt         # YOLO weights
│   ├── lstm_classifier.pt # Trained LSTM
│   └── DigitalWitness_Training.ipynb
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
| `YOLO_CONF_THRESHOLD` | 0.3 | Detection confidence |
| `CNN_FEATURE_DIM` | 512 | Feature vector size |
| `LSTM_HIDDEN_DIM` | 256 | LSTM hidden state |
| `LSTM_SEQUENCE_LENGTH` | 30 | Frames per sequence |
| `ALERT_THRESHOLD` | 0.5 | Alert trigger threshold |

## Training

The notebook `models/DigitalWitness_Training.ipynb` trains the LSTM classifier:
- Extracts CNN features from videos
- Trains bidirectional LSTM with attention
- Saves model to `models/lstm_classifier.pt`

**Note:** YOLOv8 uses pretrained weights (auto-downloaded). Only the LSTM is trained on your dataset.

## Classes

| Class | Description |
|-------|-------------|
| `normal` | Regular shopping behavior |
| `shoplifting` | Suspicious/theft behavior |

## Requirements

- Python 3.8+
- PyTorch, Ultralytics (YOLO), OpenCV, MediaPipe
- GPU recommended for faster inference
