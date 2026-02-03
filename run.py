"""
Digital Witness - Unified Entry Point
======================================

This is the main CLI entry point for the Digital Witness retail security system.
It routes commands to the appropriate subsystems based on command-line arguments.

Architecture Overview:
    The system uses a deep learning pipeline for behavior analysis:

    Video Input → YOLO (Detection) → CNN (Features) → LSTM (Classification) → Alert

    - YOLO v8: Detects persons and products in video frames
    - ResNet18 CNN: Extracts 512-dim spatial features per frame
    - Bidirectional LSTM: Classifies temporal behavior sequences
    - Bias-Aware Scorer: Adjusts scores for fairness

Available Commands:
    python run.py              # Demo mode with simulated data
    python run.py --ui         # Launch Streamlit web interface
    python run.py --train      # Train LSTM model on video dataset
    python run.py --evaluate   # Evaluate model with detailed metrics
    python run.py <video.mp4>  # Analyze specific video file

Key Design Principle:
    This system does NOT determine guilt. It provides intent risk assessments
    with explainable evidence for human operators to review.

Author: Digital Witness Team
Date: 2026
"""
import sys
import subprocess
from pathlib import Path

# Ensure src package is importable from project root
# This allows imports like "from src.config import ..." to work
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, MODELS_DIR


def print_banner():
    """Display ASCII banner for CLI output."""
    print()
    print("=" * 60)
    print("  DIGITAL WITNESS")
    print("  Deep Learning Retail Security Assistant")
    print("  YOLO → CNN → LSTM Pipeline")
    print("=" * 60)
    print()


def print_help():
    """Display CLI usage instructions."""
    print_banner()
    print("Usage:")
    print("  python run.py              Run demo mode (simulated data)")
    print("  python run.py --ui         Launch web interface")
    print("  python run.py --train      Train LSTM model on video dataset")
    print("  python run.py --evaluate   Evaluate model with detailed metrics")
    print("  python run.py <video.mp4>  Analyze specific video")
    print("  python run.py --help       Show this help message")
    print()
    print("Pipeline: YOLO (detection) → CNN (features) → LSTM (classification)")
    print()


def launch_ui():
    """
    Launch Streamlit web interface.

    Spawns the Streamlit server process with disabled telemetry.
    """
    print_banner()
    print("[MODE] Web Interface\n")

    lstm_model = MODELS_DIR / "lstm_classifier.pt"
    if not lstm_model.exists():
        print("[INFO] LSTM model not found. Will use untrained model.")
        print("[TIP]  Run 'python run.py --train' to train the model.\n")

    print("Starting Streamlit at http://localhost:8501")
    print("Press Ctrl+C to stop.\n" + "-" * 60 + "\n")

    # Resolve streamlit path: prefer venv, fallback to system PATH
    streamlit_path = (
        Path(".venv/Scripts/streamlit.exe") if sys.platform == "win32"
        else Path(".venv/bin/streamlit")
    )
    if not streamlit_path.exists():
        streamlit_path = "streamlit"

    try:
        subprocess.run([
            str(streamlit_path), "run", "src/ui/app.py",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.port=8501"
        ])
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")
    except FileNotFoundError:
        print("[ERROR] Streamlit not found. Run: pip install -r requirements.txt")
        sys.exit(1)


def run_training():
    """
    Execute the deep learning model training pipeline.

<<<<<<< HEAD
    Delegates to training module which processes videos
    from data/training/ and saves the trained model to models/.
=======
    Trains the LSTM classifier using CNN-extracted features from video data.
    Requires training videos in data/training/normal/ and data/training/shoplifting/
>>>>>>> MVP
    """
    print_banner()
    print("[MODE] Deep Learning Model Training\n")
    print("Training LSTM classifier on video dataset...")
    print("  - Feature extraction: CNN (ResNet18)")
    print("  - Sequence classification: Bidirectional LSTM")
    print()

<<<<<<< HEAD
    # Lazy import to avoid loading ML dependencies until needed
    from src.training import main as train_main
=======
    from src.models.train_deep_model import main as train_main
>>>>>>> MVP
    result = train_main()

    if result.get('success'):
        print("\n" + "=" * 60)
        print("  Training complete!")
        print(f"  Model saved to: {result.get('model_path')}")
        print("=" * 60)
    else:
        print(f"\n[ERROR] Training failed: {result.get('error', 'Unknown')}")
        sys.exit(1)


def run_evaluation():
    """
    Run model evaluation with detailed metrics.

    Computes accuracy, precision, recall, F1 score, and confusion matrix
    on the training dataset (or specified test dataset).
    """
    print_banner()
    print("[MODE] Model Evaluation\n")
    print("Evaluating LSTM classifier...")
    print("  - Computing: Accuracy, Precision, Recall, F1 Score")
    print("  - Generating: Confusion Matrix")
    print()

<<<<<<< HEAD
    # Model bootstrap: train model if none exists
    if not BEHAVIOR_MODEL_PATH.exists():
        print("[SETUP] No model found. Please train the model first.\n")
        print("Run: python run.py --train\n")
        return
=======
    from src.models.train_deep_model import evaluate_model
    result = evaluate_model()

    if result.get('success'):
        print("\n" + "=" * 60)
        print("  Evaluation complete!")
        print("  Metrics saved to models/lstm_classifier_info.json")
        print("=" * 60)
    else:
        print(f"\n[ERROR] Evaluation failed: {result.get('error', 'Unknown')}")
        sys.exit(1)


def run_analysis(video_path=None):
    """
    Run deep learning video analysis pipeline.

    Uses YOLO for detection, CNN for feature extraction,
    and LSTM for temporal behavior classification.
    """
    print_banner()
    print("[MODE] Video Analysis\n")
>>>>>>> MVP

    # Auto-discover video file if not specified
    if video_path is None:
        videos_dir = DATA_DIR / "videos"
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if video_files:
            video_path = video_files[0]
            print(f"[INFO] Using video: {video_path.name}\n")
        else:
            print("[INFO] No video found in data/videos/")
            print("[INFO] Running demo mode with simulated data.\n")

    # Delegate to main pipeline
    from src.main import run_pipeline, run_demo_mode

    if video_path and Path(video_path).exists():
        run_pipeline(video_path)
    else:
        run_demo_mode()


def main():
    """
    CLI argument router - Main entry point for command dispatch.

    This function parses command-line arguments and routes to the
    appropriate handler function. It supports the following modes:

    Modes:
        --ui        Launch the Streamlit web dashboard for visual analysis
        --train     Train the LSTM classifier using videos in data/training/
        --evaluate  Run evaluation metrics on the trained model
        --help/-h   Display usage instructions
        <path>      Analyze a specific video file
        (none)      Run demo mode with simulated data

    Example:
        python run.py --train           # Train model
        python run.py video.mp4         # Analyze video
        python run.py                   # Demo mode
    """
    args = sys.argv[1:]  # Get command-line arguments (excluding script name)

    # No arguments: run demo mode
    if not args:
        run_analysis()
        return

    arg = args[0]

    # Help command
    if arg in ("--help", "-h"):
        print_help()
        return

    # Web interface mode
    if arg == "--ui":
        launch_ui()
        return

    # Training mode - trains LSTM on video dataset
    if arg == "--train":
        run_training()
        return

    # Evaluation mode - computes precision, recall, F1, confusion matrix
    if arg == "--evaluate":
        run_evaluation()
        return

    # If argument doesn't match any command, treat it as a video file path
    video_path = Path(arg)
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}\n")
        print_help()
        sys.exit(1)

    run_analysis(video_path)


if __name__ == "__main__":
    main()
