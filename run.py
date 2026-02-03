"""
Digital Witness - Entry Point

Unified entry point for the Digital Witness retail security system.
Launches the Streamlit web interface or runs training.

Usage:
    python run.py              Launch web interface (default)
    python run.py --train      Train LSTM model on video dataset
    python run.py --evaluate   Evaluate model with detailed metrics
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR


def launch_ui():
    """Launch Streamlit web interface."""
    print("\n" + "=" * 50)
    print("  DIGITAL WITNESS")
    print("  Retail Security Assistant")
    print("=" * 50 + "\n")

    lstm_model = MODELS_DIR / "lstm_classifier.pt"
    if not lstm_model.exists():
        print("[INFO] LSTM model not found.")
        print("[TIP]  Run 'python run.py --train' first.\n")

    print("Starting web interface at http://localhost:8501")
    print("Press Ctrl+C to stop.\n")

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
        print("\nServer stopped.")
    except FileNotFoundError:
        print("[ERROR] Streamlit not found. Run: pip install -r requirements.txt")
        sys.exit(1)


def run_training():
    """Train the LSTM classifier on video dataset."""
    print("\n" + "=" * 50)
    print("  LSTM MODEL TRAINING")
    print("=" * 50 + "\n")

    from src.models.train_deep_model import train_lstm_classifier
    result = train_lstm_classifier()

    if result.get('success'):
        print(f"\nModel saved to: {result.get('model_path')}")
    else:
        print(f"\nTraining failed: {result.get('error', 'Unknown')}")
        sys.exit(1)


def run_evaluation():
    """Evaluate trained model with detailed metrics."""
    print("\n" + "=" * 50)
    print("  MODEL EVALUATION")
    print("=" * 50 + "\n")

    from src.models.train_deep_model import evaluate_model
    result = evaluate_model()

    if not result.get('success'):
        print(f"\nEvaluation failed: {result.get('error', 'Unknown')}")
        sys.exit(1)


def main():
    """Route to appropriate handler based on arguments."""
    args = sys.argv[1:]

    if not args:
        launch_ui()
        return

    arg = args[0]

    if arg in ("--help", "-h"):
        print(__doc__)
        return

    if arg == "--train":
        run_training()
        return

    if arg == "--evaluate":
        run_evaluation()
        return

    if arg == "--ui":
        launch_ui()
        return

    print(f"Unknown argument: {arg}")
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
