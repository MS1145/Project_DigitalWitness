# Digital Witness

**Bias-Aware, Explainable Retail Security Assistant**

A computer vision system that detects potential shoplifting behavior by analyzing body pose patterns in surveillance video and correlating findings with Point-of-Sale (POS) transaction data. This is an **advisory system** designed to support human decision-making - all alerts require human validation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

---

## Important Notice

> **This system does NOT determine guilt.** It provides intent risk assessments with explainable evidence for human operators to review. Final decisions must always be made by qualified personnel.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Product Vision](#product-vision)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Phase 2: Real-Time System](#phase-2-real-time-decision-support-system)
- [Roadmap](#roadmap)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Ethical Considerations](#ethical-considerations)
- [Technical Requirements](#technical-requirements)
- [License](#license)

---

## Problem Statement

Current AI-powered retail security systems operate as opaque black boxes. They are unable to reliably distinguish between intentional shoplifting and accidental or vulnerable behaviour such as children misunderstanding purchases or elderly customers making mistakes. This results in false accusations, erosion of customer trust, ethical concerns, and potential legal liability.

Additionally, existing systems often treat behavioural evidence and POS transactions as separate silos, making it difficult to determine whether items physically taken by a customer were actually billed.

---

## Product Vision

Digital Witness is designed as a **Blameless AI Assistant** that supports — not replaces — human decision-making in retail security.

The system analyses a customer's **physical interaction with products** and correlates it with **POS transaction data** to determine whether all products taken by the customer have been billed.

The system does **not** determine guilt or confirm shoplifting. Instead, it provides:

- An intent risk assessment
- An Intent Risk Level (Level 1–5) indicating severity and confidence
- Short digital forensic video clips highlighting suspicious moments
- Clear, human-readable explanations supporting review

All alerts are **advisory**. Final accountability and decisions — including whether to assist a customer, dismiss an alert, or escalate to store management — always remain with a **human operator**.

---

## Key Features

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Pose-Based Behavior Analysis** | Uses MediaPipe to extract 33 body landmarks and analyze movement patterns |
| **Machine Learning Classification** | Random Forest classifier trained on real video data |
| **POS Data Correlation** | Cross-references detected behaviors with billing records |
| **Explainable AI** | Provides clear reasoning for all assessments with confidence scores |
| **Interactive Web UI** | Streamlit-based dashboard for video upload and analysis |
| **Human-in-the-Loop** | All alerts are advisory and require human validation |
| **Digital Forensics** | Extracts video clips of key moments for review |

### Design Principles

Digital Witness intentionally builds upon **established open-source AI models** rather than developing low-level computer vision systems from scratch:

- Pre-trained pose estimation models (MediaPipe) for anonymised behavioural analysis
- Established explainability patterns for transparent reasoning
- No facial recognition or identity inference

**Original contributions:**
- Behavioural and transactional data fusion
- Intent risk inference across full customer journeys
- Explainable digital forensic evidence generation
- Bias-aware, human-in-the-loop decision support design

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam or video files for testing

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Project_DigitalWitness.git
   cd Project_DigitalWitness
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)
   ```bash
   python run.py --train
   ```

---

## Usage

All commands go through the unified `run.py` entry point:

```bash
python run.py --ui               # Launch web interface (recommended)
python run.py --train            # Train model on video dataset
python run.py                    # Run CLI demo mode
python run.py path/to/video.mp4  # Analyze specific video (CLI)
python run.py --help             # Show help
```

### Web Interface (Recommended)

```bash
python run.py --ui
```

This opens the Streamlit dashboard at `http://localhost:8501`

### Command Line Analysis

```bash
# Analyze a specific video
python run.py path/to/video.mp4

# Run demo mode (simulated data)
python run.py
```

### Web UI Guide

#### Tab 1: Model Performance

View model metrics including:
- Accuracy, Precision, Recall, F1-Score
- Cross-validation results
- Confusion matrix visualization
- Feature importance rankings

#### Tab 2: Video Analysis

1. **Upload Video**: Select a video file (MP4, AVI, MOV, MKV)
2. **Configure POS Data**: Use the transaction editor to simulate billing scenarios
   - Normal Purchase: All items billed
   - Partial Billing: Some items missing
   - No Billing: No transaction recorded
3. **Analyze**: Click "Analyze Video" to run the pipeline
4. **Review Results**:
   - Risk level indicator (HIGH/MEDIUM/LOW/NONE)
   - Intent score gauge
   - Behavior timeline visualization
   - Advisory summary

---

## How It Works

### 10-Step Analysis Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Load Video  │───▶│    Pose      │───▶│   Feature    │
│              │    │  Estimation  │    │  Extraction  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐           ▼
│  Load POS    │───▶│ Cross-Check  │◀───┌──────────────┐
│    Data      │    │              │    │  Behavior    │
└──────────────┘    └──────┬───────┘    │Classification│
                           │            └──────────────┘
                           ▼
                    ┌──────────────┐
                    │ Intent Score │
                    │ Calculation  │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   Generate   │  │   Extract    │  │    Build     │
  │    Alert     │  │    Clips     │  │  Case File   │
  └──────────────┘  └──────────────┘  └──────────────┘
```

**Steps:**
1. **Load Video** - Extract frames and metadata using OpenCV
2. **Pose Estimation** - MediaPipe extracts 33 body landmarks per frame
3. **Feature Extraction** - Sliding windows (30 frames) generate 21 features
4. **Behavior Classification** - Random Forest predicts: `normal` or `shoplifting`
5. **Load POS Data** - Parse transaction records
6. **Cross-Check** - Compare detected interactions with billing
7. **Intent Score** - Calculate weighted risk score (0.0-1.0)
8. **Generate Alert** - Create advisory if score ≥ threshold
9. **Extract Clips** - Save forensic video segments
10. **Build Case File** - Generate audit trail

### Feature Set (21 Features)

- Hand velocities and positions
- Body displacement and movement
- Arm angles and postures
- Pose detection confidence
- Hand-to-body distances

### Intent Scoring Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Discrepancy | 40% | Items detected but not billed |
| Concealment | 30% | Hiding behavior detected |
| Bypass | 20% | Checkout avoidance patterns |
| Duration | 10% | Length of suspicious activity |

### Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| NONE | < 0.3 | Normal behavior |
| LOW | 0.3 - 0.5 | Minor anomalies |
| MEDIUM | 0.5 - 0.7 | Review recommended |
| HIGH | ≥ 0.7 | Immediate review required |

---

## Project Structure

```
Project_DigitalWitness/
├── run.py                      # Unified entry point (--ui, --train, analysis)
├── app.py                      # Streamlit web UI
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for cloud)
│
├── src/                        # Source code
│   ├── config.py               # Configuration constants
│   ├── main.py                 # Pipeline orchestration
│   │
│   ├── video/                  # Video processing
│   │   ├── loader.py           # Video loading utilities
│   │   └── clip_extractor.py   # Forensic clip extraction
│   │
│   ├── pose/                   # Pose estimation & ML
│   │   ├── estimator.py        # MediaPipe pose detection
│   │   ├── feature_extractor.py    # 21-feature extraction
│   │   ├── behavior_classifier.py  # Random Forest classifier
│   │   └── train_video_classifier.py  # Training pipeline
│   │
│   ├── pos/                    # POS data handling
│   │   ├── data_loader.py      # JSON transaction parser
│   │   └── mock_generator.py   # Test data generation
│   │
│   ├── analysis/               # Core analysis
│   │   ├── cross_checker.py    # Video-POS reconciliation
│   │   ├── intent_scorer.py    # Risk scoring engine
│   │   └── alert_generator.py  # Advisory alert creation
│   │
│   └── output/
│       └── case_builder.py     # Case file generation
│
├── models/                     # Trained models
│   ├── behavior_classifier.pkl
│   └── behavior_classifier_info.json
│
├── data/                       # Data directories
│   ├── training/
│   │   ├── normal/             # Normal behavior videos
│   │   └── shoplifting/        # Shoplifting behavior videos
│   ├── videos/                 # Input videos for analysis
│   └── pos/                    # POS transaction files
│
├── outputs/                    # Generated outputs
│   ├── cases/                  # Case files (JSON)
│   └── clips/                  # Extracted video clips
│
└── docs/                       # Documentation
    └── IMPLEMENTATION_PLAN.md  # Phase 2 implementation details
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.0% |
| Precision | 80.0% |
| Recall | 80.0% |
| F1-Score | 80.0% |
| CV Accuracy | 81.5% ± 1.7% |

### Top Features by Importance

1. Right elbow angle (mean)
2. Body velocity (mean)
3. Left hand-body distance (mean)
4. Hand height relative to shoulders
5. Body displacement

---

## Phase 2: Real-Time Decision Support System

The system will operate as a **real-time, blameless decision-support assistant** by continuously analysing customer behaviour using pose-based interaction tracking while simultaneously ingesting live POS transaction events.

### Real-Time Behavioural Analysis

| Behaviour | Description | Detection Method |
|-----------|-------------|------------------|
| **Product Pickup** | Customer picks up item from shelf | Hand-to-shelf pose, arm extension patterns |
| **Concealment Gestures** | Hand-to-pocket, hand-to-bag movements | Pose trajectory analysis, gesture classification |
| **Prolonged Holding** | Extended product holding without checkout progression | Time-based tracking with location awareness |
| **Exit Behaviour** | Movement toward exit without completing transaction | Zone-based tracking, trajectory prediction |

### Live POS Integration

- Stream POS transaction events in real-time
- Cross-check whether all physically interacted products are billed
- Detect billing discrepancies as they occur
- Support barcode scan event correlation

### Enhanced Risk Levels & Notifications

| Risk Level | Threshold | Action |
|------------|-----------|--------|
| Level 1 (Minimal) | 0.0 - 0.2 | No notification, logged only |
| Level 2 (Low) | 0.2 - 0.4 | Logged, periodic review |
| Level 3 (Medium) | 0.4 - 0.6 | Soft alert to operator dashboard |
| Level 4 (High) | 0.6 - 0.8 | Immediate notification to manager |
| Level 5 (Critical) | 0.8 - 1.0 | Urgent alert with full forensic package |

### Forensic Package Contents

When risk thresholds are exceeded, the system attaches a forensic package:

- **Product Pickup Clips** - Video segments showing item interaction
- **Billing Area Interaction** - Footage near POS/checkout zones
- **Concealment Evidence** - Clips of suspicious gestures
- **Exit Footage** - Final moments before leaving store
- **Discrepancy Explanation** - Human-readable summary

### Edge Case Handling

| Edge Case | Handling Strategy |
|-----------|-------------------|
| **Accidental Missed Scans** | Allow grace period; correlate with customer behaviour at checkout |
| **Barcode Failures** | Detect repeated scan attempts; do not flag if manual entry follows |
| **Personal Items Mistaken as Products** | Track item origin; items brought into store vs. picked from shelf |
| **Children Handling Items** | Detect child-sized poses; require adult association for alerts |
| **Elderly or Confused Customers** | Slower movement patterns; extended dwell time tolerance |
| **Items Placed Back Before Checkout** | Track put-back gestures; remove from interaction list |
| **Multiple Customers Interacting** | Associate interactions with specific pose tracks |
| **Crowded Scenarios** | Confidence-weighted alerts; avoid false positives |

### Real-Time Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Live Camera   │────▶│  Frame Buffer    │────▶│  Pose Estimator │
│   Feed (RTSP)   │     │  (Ring Buffer)   │     │  (MediaPipe)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│   POS System    │────▶│  Event Queue     │     ┌─────────────────┐
│   (Webhook/API) │     │  (Redis/Memory)  │────▶│  Fusion Engine  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              ▼
                        │  Alert Manager   │◀────┌─────────────────┐
                        │  (WebSocket)     │     │  Risk Scorer    │
                        └────────┬─────────┘     └─────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │  Dashboard   │   │  Manager App │   │  Forensic    │
      │  (Streamlit) │   │  (Push/SMS)  │   │  Storage     │
      └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Roadmap

### Version 1.0 (MVP - Completed)

- [x] Pre-recorded video analysis
- [x] Mock POS data processing
- [x] Basic behaviour classification
- [x] Streamlit demo interface
- [x] Explainable results

### Version 2.0 (Real-Time System - In Progress)

- [ ] Live video feed processing (RTSP)
- [ ] Real-time POS event streaming
- [ ] Enhanced gesture detection
- [ ] Store manager notification system
- [ ] Forensic package generation
- [ ] Edge case handling
- [ ] Multi-person tracking

### Version 3.0 (Production Ready)

- [ ] Sequence-based learning models (GRU / CNN-LSTM)
- [ ] SHAP-based explanations for transactional data
- [ ] Bias mitigation strategies
- [ ] Differentiated response logic for vulnerable groups
- [ ] Low-cost edge deployment
- [ ] Performance benchmarking
- [ ] Exportable forensic and accountability reports

---

## Configuration

Key settings in `src/config.py`:

```python
# Analysis settings
SLIDING_WINDOW_SIZE = 30      # Frames per analysis window
SLIDING_WINDOW_STRIDE = 15    # Window overlap
ALERT_THRESHOLD = 0.5         # Minimum score for alert

# Pose detection
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Intent scoring weights
WEIGHT_DISCREPANCY = 0.4
WEIGHT_CONCEALMENT = 0.3
WEIGHT_BYPASS = 0.2
WEIGHT_DURATION = 0.1
```

---

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```

### Using Ngrok

```bash
ngrok http 8501
```

---

## Ethical Considerations

This system is designed with ethical AI principles:

| Principle | Implementation |
|-----------|----------------|
| **No Facial Recognition** | Analysis based solely on body pose |
| **No Identity Tracking** | No personal data stored |
| **Explainable Decisions** | All scores include reasoning |
| **Human Oversight** | Alerts are advisory only |
| **Bias Awareness** | System acknowledges limitations |
| **Vulnerable Protection** | Special handling for children, elderly |

---

## Technical Requirements

### Phase 1 (MVP) - Current

- Python 3.8+
- 4GB RAM minimum
- CPU sufficient for demo

### Phase 2 (Real-Time)

**Hardware:**
- Camera: IP camera with RTSP support (720p minimum, 1080p recommended)
- Processing: NVIDIA GPU (GTX 1060 or better) for real-time inference
- Storage: SSD with minimum 500GB for forensic clip retention
- Network: Gigabit Ethernet for low-latency video streaming

**Software:**
- Python 3.9+
- MediaPipe for pose estimation
- OpenCV for video processing
- Redis or in-memory queue for event buffering
- WebSocket server for real-time alerts
- Streamlit for operator dashboard

**Integration:**
- POS System: REST API or webhook endpoint for transaction events
- Notification: Email, SMS (Twilio), or push notification service
- Storage: Local filesystem or cloud storage (S3) for forensic packages

---

## Success Criteria

### Phase 1 (MVP) - Completed

- [x] Video can be uploaded and processed end-to-end
- [x] Product interactions are detected
- [x] POS data is cross-checked against physical actions
- [x] Alerts are generated based on risk levels
- [x] Explanations and forensic clips are available
- [x] Human validation is required for all final decisions

### Phase 2 (Real-Time System)

- [ ] Live camera feed processed at ≥15 FPS
- [ ] POS events ingested within 500ms
- [ ] Behaviour detection accuracy ≥80%
- [ ] False positive rate <10% for high-risk alerts
- [ ] Forensic clips generated within 5 seconds
- [ ] Manager notification within 10 seconds of threshold breach
- [ ] Edge cases handled without false accusations
- [ ] System maintains 99% uptime

---

## Technologies Used

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: scikit-learn (Random Forest)
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: NumPy, Pandas

---

## License

This project is developed for educational purposes as part of a Final Year Project at IIT.

---

## Acknowledgments

- MediaPipe by Google for pose estimation
- UCF Crime Dataset for training data inspiration
- Streamlit for the web framework

---

## Contact

For questions or feedback about this project, please open an issue on GitHub.

---

<p align="center">
  <b>Digital Witness</b> - Final Year Project 2026<br>
  <i>An advisory system for retail security that prioritizes explainability and human oversight.</i>
</p>
