"""
Configuration constants for Digital Witness system.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Video paths
VIDEO_INPUT_DIR = DATA_DIR / "videos"
DEFAULT_VIDEO_PATH = VIDEO_INPUT_DIR / "sample.mp4"

# POS data paths
POS_DATA_DIR = DATA_DIR / "pos"
DEFAULT_POS_PATH = POS_DATA_DIR / "transactions.json"

# Training data paths
TRAINING_DATA_DIR = DATA_DIR / "training"
TRAINING_NORMAL_DIR = TRAINING_DATA_DIR / "normal"
TRAINING_SHOPLIFTING_DIR = TRAINING_DATA_DIR / "shoplifting"

# Model paths
BEHAVIOR_MODEL_PATH = MODELS_DIR / "behavior_classifier.pkl"

# MediaPipe configuration
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Feature extraction configuration
SLIDING_WINDOW_SIZE = 30  # frames
SLIDING_WINDOW_STRIDE = 15  # frames

# Behavior classes
BEHAVIOR_CLASSES = ["normal", "pickup", "concealment", "bypass"]

# Intent scoring thresholds
INTENT_THRESHOLD_LOW = 0.3
INTENT_THRESHOLD_MEDIUM = 0.5
INTENT_THRESHOLD_HIGH = 0.7
INTENT_THRESHOLD_CRITICAL = 0.85

# Intent scoring weights
WEIGHT_DISCREPANCY = 0.4
WEIGHT_CONCEALMENT = 0.3
WEIGHT_BYPASS = 0.2
WEIGHT_DURATION = 0.1

# Severity levels
SEVERITY_LEVELS = {
    "LOW": (0.0, INTENT_THRESHOLD_LOW),
    "MEDIUM": (INTENT_THRESHOLD_LOW, INTENT_THRESHOLD_MEDIUM),
    "HIGH": (INTENT_THRESHOLD_MEDIUM, INTENT_THRESHOLD_HIGH),
    "CRITICAL": (INTENT_THRESHOLD_HIGH, 1.0)
}

# Clip extraction configuration
CLIP_BUFFER_BEFORE = 3.0  # seconds before event
CLIP_BUFFER_AFTER = 3.0   # seconds after event
CLIP_OUTPUT_DIR = OUTPUTS_DIR / "clips"

# Alert configuration
ALERT_THRESHOLD = INTENT_THRESHOLD_MEDIUM  # Generate alert if score >= this

# Output configuration
CASE_OUTPUT_DIR = OUTPUTS_DIR / "cases"
