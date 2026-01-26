# Digital Witness

**Bias-Aware, Explainable Retail Security Assistant**

A computer vision system that detects potential shoplifting behavior by analyzing body pose patterns in surveillance video and correlating findings with Point-of-Sale (POS) transaction data. This is an **advisory system** designed to support human decision-making - all alerts require human validation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

---

## Key Features

- **Pose-Based Behavior Analysis**: Uses MediaPipe to extract 33 body landmarks and analyze movement patterns
- **Machine Learning Classification**: Random Forest classifier trained on real video data (80% accuracy)
- **POS Data Correlation**: Cross-references detected behaviors with billing records
- **Explainable AI**: Provides clear reasoning for all assessments with confidence scores
- **Interactive Web UI**: Streamlit-based dashboard for video upload and analysis
- **Human-in-the-Loop**: All alerts are advisory and require human validation

---

## Important Notice

> **This system does NOT determine guilt.** It provides intent risk assessments with explainable evidence for human operators to review. Final decisions must always be made by qualified personnel.

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
   python train.py
   ```

---

## Usage

### Launch Web UI (Recommended)

```bash
python demo.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Command Line Analysis

```bash
# Analyze a specific video
python run.py path/to/video.mp4

# Run in demo mode
python run.py
```

---

## Web UI Guide

### Tab 1: Model Performance

View model metrics including:
- Accuracy, Precision, Recall, F1-Score
- Cross-validation results
- Confusion matrix visualization
- Feature importance rankings

### Tab 2: Video Analysis

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

## Project Structure

```
Project_DigitalWitness/
├── app.py                      # Streamlit web UI
├── demo.py                     # Demo launcher script
├── run.py                      # CLI runner
├── train.py                    # Model training script
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
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

---

## How It Works

### 10-Step Analysis Pipeline

1. **Load Video** - Extract frames and metadata using OpenCV
2. **Pose Estimation** - MediaPipe extracts 33 body landmarks per frame
3. **Feature Extraction** - Sliding windows (30 frames) generate 21 features:
   - Hand velocities and positions
   - Body displacement and movement
   - Arm angles and postures
   - Pose detection confidence
4. **Behavior Classification** - Random Forest predicts: `normal` or `shoplifting`
5. **Load POS Data** - Parse transaction records
6. **Cross-Check** - Compare detected interactions with billing
7. **Intent Score** - Calculate weighted risk score (0.0-1.0)
8. **Generate Alert** - Create advisory if score ≥ threshold
9. **Extract Clips** - Save forensic video segments
10. **Build Case File** - Generate audit trail

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

## Technologies Used

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: scikit-learn (Random Forest)
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: NumPy, Pandas

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

## Configuration

Key settings in `src/config.py`:

```python
SLIDING_WINDOW_SIZE = 30      # Frames per analysis window
SLIDING_WINDOW_STRIDE = 15    # Window overlap
ALERT_THRESHOLD = 0.5         # Minimum score for alert
POSE_MIN_DETECTION_CONFIDENCE = 0.5
```

---

## Ethical Considerations

This system is designed with ethical AI principles:

- **No Facial Recognition**: Analysis based solely on body pose
- **No Identity Tracking**: No personal data stored
- **Explainable Decisions**: All scores include reasoning
- **Human Oversight**: Alerts are advisory only
- **Bias Awareness**: System acknowledges limitations

---

## Limitations

- Pre-recorded video only (no live streaming)
- Single person detection
- 2-class classification (normal/shoplifting)
- Requires adequate lighting and camera angle
- Mock POS data for demonstration

---

## Future Improvements

- [ ] Multi-person tracking
- [ ] Real-time video stream support
- [ ] Additional behavior classes
- [ ] Bias mitigation strategies
- [ ] Mobile app integration
- [ ] Cloud-based processing

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
