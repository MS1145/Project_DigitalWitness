"""
Central configuration for Digital Witness system.

All tunable parameters and paths are defined here to ensure consistency
across modules and to facilitate hyperparameter adjustment.

Configuration Categories:
-------------------------
1. PATH CONFIGURATION - File system paths for data, models, outputs
2. FEATURE EXTRACTION - Sliding window parameters for temporal analysis
3. BEHAVIOR CLASSIFICATION - ML model classes
4. INTENT SCORING - Weights and thresholds for risk assessment
5. DEEP LEARNING - YOLO, CNN, LSTM hyperparameters
6. BIAS DETECTION - Fairness and bias mitigation parameters
7. FORENSIC OUTPUT - Report and evidence package settings

Tuning Guidelines:
------------------
- Lower thresholds = more sensitive (more alerts, more false positives)
- Higher thresholds = less sensitive (fewer alerts, might miss incidents)
- Weights should sum to 1.0 for proper score normalization
- Adjust CNN_INPUT_SIZE if using different backbone architectures
"""
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

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

# Model paths (deep learning models stored in MODELS_DIR)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

SLIDING_WINDOW_SIZE = 30   # Frames per feature window (~1 sec at 30fps)
SLIDING_WINDOW_STRIDE = 15 # Overlap between windows (50% overlap)

# ============================================================================
# BEHAVIOR CLASSIFICATION
# ============================================================================

# Classes the ML model can predict (order matters for label encoding)
BEHAVIOR_CLASSES = ["normal", "pickup", "concealment", "bypass"]

# ============================================================================
# INTENT SCORING
# ============================================================================

# Severity thresholds (score ranges)
INTENT_THRESHOLD_LOW = 0.3
INTENT_THRESHOLD_MEDIUM = 0.5
INTENT_THRESHOLD_HIGH = 0.7
INTENT_THRESHOLD_CRITICAL = 0.85

# Component weights for weighted sum (must sum to 1.0)
WEIGHT_DISCREPANCY = 0.4   # POS mismatch is strongest signal
WEIGHT_CONCEALMENT = 0.3   # Hiding behavior
WEIGHT_BYPASS = 0.2        # Avoiding checkout
WEIGHT_DURATION = 0.1      # Time spent in suspicious state

# Score-to-severity mapping (min, max) ranges
SEVERITY_LEVELS = {
    "LOW": (0.0, INTENT_THRESHOLD_LOW),
    "MEDIUM": (INTENT_THRESHOLD_LOW, INTENT_THRESHOLD_MEDIUM),
    "HIGH": (INTENT_THRESHOLD_MEDIUM, INTENT_THRESHOLD_HIGH),
    "CRITICAL": (INTENT_THRESHOLD_HIGH, 1.0)
}

# ============================================================================
# FORENSIC CLIP EXTRACTION
# ============================================================================

CLIP_BUFFER_BEFORE = 3.0  # Seconds of context before event
CLIP_BUFFER_AFTER = 3.0   # Seconds of context after event
CLIP_OUTPUT_DIR = OUTPUTS_DIR / "clips"

# ============================================================================
# ALERT GENERATION
# ============================================================================

# Minimum intent score required to trigger an alert
ALERT_THRESHOLD = INTENT_THRESHOLD_MEDIUM

# ============================================================================
# OUTPUT PATHS
# ============================================================================

CASE_OUTPUT_DIR = OUTPUTS_DIR / "cases"

# ============================================================================
# QUALITY ANALYSIS
# ============================================================================

QUALITY_MIN_DETECTION_CONFIDENCE = 0.5  # Minimum detection confidence for quality
QUALITY_MIN_DETECTION_RATE = 0.6        # Minimum ratio of frames with detections

# ============================================================================
# EDGE CASE HANDLING
# ============================================================================

EDGE_CASE_MIN_CONFIDENCE = 0.5       # Minimum confidence to trust prediction
EDGE_CASE_MIN_QUALITY = 0.4          # Minimum quality to avoid quality flag
EDGE_CASE_AMBIGUITY_THRESHOLD = 0.2  # Max probability gap for ambiguity

# ============================================================================
# DEEP LEARNING MODEL CONFIGURATION
# ============================================================================

# YOLO Object Detection
# Use YOLOv8s (small) for better accuracy - nano misses detections in multi-person scenes
YOLO_MODEL_PATH = MODELS_DIR / "yolov8s.pt"
YOLO_CONF_THRESHOLD = 0.3            # Lower threshold for better recall (was 0.5)
YOLO_IOU_THRESHOLD = 0.45            # Slightly lower for crowded scenes (was 0.5)
YOLO_CLASSES = ["person", "bottle", "cup", "food", "bag"]  # Retail-relevant

# CNN Feature Extraction
CNN_BACKBONE = "resnet18"            # Backbone architecture
CNN_FEATURE_DIM = 512                # Output feature dimension
CNN_PRETRAINED = True                # Use ImageNet pretrained weights
CNN_INPUT_SIZE = (224, 224)          # Input image size

# LSTM Temporal Classification
LSTM_HIDDEN_DIM = 256                # LSTM hidden state dimension
LSTM_NUM_LAYERS = 2                  # Number of LSTM layers
LSTM_SEQUENCE_LENGTH = 30            # Frames per sequence
LSTM_DROPOUT = 0.3                   # Dropout probability

# Intent Classes for Deep Model
INTENT_CLASSES = ["normal", "pickup", "concealment", "bypass"]

# ============================================================================
# BIAS DETECTION
# ============================================================================

BIAS_SENSITIVITY = 0.5               # How sensitive to potential bias (0-1)
BIAS_DETECTION_RATE_MIN = 0.6        # Minimum acceptable detection rate
BIAS_CONFIDENCE_VARIANCE_MAX = 0.1   # Maximum acceptable confidence variance

# ============================================================================
# FORENSIC OUTPUT
# ============================================================================

FORENSIC_PACKAGES_DIR = OUTPUTS_DIR / "forensic_packages"
FORENSIC_REPORTS_DIR = OUTPUTS_DIR / "reports"

# ============================================================================
# ANNOTATION COLORS (BGR format for OpenCV)
# ============================================================================

ANNOTATION_COLORS = {
    "normal": (0, 200, 0),          # Green
    "pickup": (0, 200, 255),        # Yellow-Orange
    "concealment": (0, 128, 255),   # Orange
    "bypass": (0, 0, 255),          # Red
    "shoplifting": (0, 0, 200)      # Dark Red
}

# ============================================================================
# VULNERABLE GROUP DETECTION
# ============================================================================

# Child detection thresholds (based on pose height ratios)
CHILD_HEIGHT_RATIO_MAX = 0.6        # Max height ratio vs frame height
CHILD_MOVEMENT_TOLERANCE = 1.5      # Multiplier for allowable erratic movement

# Elderly detection thresholds
ELDERLY_MOVEMENT_SPEED_MIN = 0.3    # Slow movement threshold
ELDERLY_DWELL_TIME_TOLERANCE = 2.0  # Multiplier for acceptable dwell time

# Vulnerable group handling
VULNERABLE_CONFIDENCE_REDUCTION = 0.3   # Reduce confidence for vulnerable alerts
VULNERABLE_REQUIRES_REVIEW = True       # Always require human review
