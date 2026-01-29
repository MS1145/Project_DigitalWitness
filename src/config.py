"""
Central configuration for Digital Witness system.

All tunable parameters and paths are defined here to ensure consistency
across modules and to facilitate hyperparameter adjustment.
"""
import os
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

# Model paths
BEHAVIOR_MODEL_PATH = MODELS_DIR / "behavior_classifier.pkl"

# ============================================================================
# POSE ESTIMATION (MediaPipe)
# ============================================================================

POSE_MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence to detect a pose
POSE_MIN_TRACKING_CONFIDENCE = 0.5   # Minimum confidence to track between frames

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
# RTSP STREAMING (Phase 2)
# ============================================================================

RTSP_BUFFER_SIZE = 30              # Frames to buffer for smooth playback
RTSP_RECONNECT_DELAY = 5.0         # Seconds between reconnection attempts
RTSP_TIMEOUT = 10.0                # Connection timeout in seconds
RTSP_DEFAULT_FPS = 30.0            # Assumed FPS when stream doesn't report

# ============================================================================
# POS SIMULATOR (Phase 2)
# ============================================================================

POS_WEBHOOK_PORT = 5001            # Port for webhook API demo
POS_WEBHOOK_HOST = "0.0.0.0"       # Host for webhook API

# Product catalog for POS simulator
POS_PRODUCT_CATALOG = [
    {"sku": "ITEM001", "name": "Chocolate Bar", "price": 2.99},
    {"sku": "ITEM002", "name": "Energy Drink", "price": 3.49},
    {"sku": "ITEM003", "name": "Snack Pack", "price": 4.99},
    {"sku": "ITEM004", "name": "Bottled Water", "price": 1.49},
    {"sku": "ITEM005", "name": "Chewing Gum", "price": 1.99},
    {"sku": "ITEM006", "name": "Protein Bar", "price": 3.99},
    {"sku": "ITEM007", "name": "Soda Can", "price": 1.79},
    {"sku": "ITEM008", "name": "Candy Pack", "price": 2.49},
    {"sku": "ITEM009", "name": "Coffee Cup", "price": 2.99},
    {"sku": "ITEM010", "name": "Sandwich", "price": 5.99},
]

# ============================================================================
# EMAIL NOTIFICATION (Phase 2)
# ============================================================================

EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_USE_TLS = True
# Read from environment variables for security
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")  # Gmail App Password
EMAIL_RECIPIENTS = os.environ.get("EMAIL_RECIPIENTS", "").split(",") if os.environ.get("EMAIL_RECIPIENTS") else []

# ============================================================================
# ALERT MANAGEMENT (Phase 2)
# ============================================================================

ALERTS_STORAGE_DIR = OUTPUTS_DIR / "alerts"
ALERT_AUTO_NOTIFY = True           # Auto-send email on new alerts
ALERT_NOTIFICATION_COOLDOWN = 60   # Seconds between notifications for same session
