"""
Digital Witness - Unified Entry Point

Application router that delegates to the appropriate module based on CLI arguments.
Supports three execution modes: UI (Streamlit), training, and analysis (CLI).
"""
import sys
import subprocess
from pathlib import Path

# Ensure src package is importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import BEHAVIOR_MODEL_PATH, DEFAULT_VIDEO_PATH, DATA_DIR


def print_banner():
    """Display ASCII banner for CLI output."""
    print()
    print("=" * 60)
    print("  DIGITAL WITNESS")
    print("  Bias-Aware, Explainable Retail Security Assistant")
    print("=" * 60)
    print()


def print_help():
    """Display CLI usage instructions."""
    print_banner()
    print("Usage:")
    print("  python run.py              Run CLI demo mode")
    print("  python run.py --ui         Launch web interface")
    print("  python run.py --train      Train model on video dataset")
    print("  python run.py <video.mp4>  Analyze specific video (CLI)")
    print()


def launch_ui():
    """
    Launch Streamlit web interface.

    Resolves the streamlit executable path based on OS (Windows vs Unix),
    then spawns the Streamlit server process with disabled telemetry.
    """
    print_banner()
    print("[MODE] Web Interface\n")

    if not BEHAVIOR_MODEL_PATH.exists():
        print("[WARNING] Model not found. Run: python run.py --train\n")

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
            str(streamlit_path), "run", "app.py",
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
    Execute the model training pipeline.

    Delegates to train_video_classifier module which processes videos
    from data/training/ and saves the trained model to models/.
    """
    print_banner()
    print("[MODE] Video-Based Model Training\n")

    # Lazy import to avoid loading ML dependencies until needed
    from src.pose.train_video_classifier import main as train_main
    result = train_main()

    if result.get('success'):
        print("\n" + "=" * 60)
        print("  Training complete. Model saved.")
        print("=" * 60)
    else:
        print(f"\n[ERROR] Training failed: {result.get('error', 'Unknown')}")
        sys.exit(1)


def run_analysis(video_path=None):
    """
    Run video analysis pipeline (CLI mode).

    If no model exists, falls back to synthetic training for demo purposes.
    Auto-discovers video files from data/videos/ if no path is provided.
    """
    print_banner()
    print("[MODE] Video Analysis (CLI)\n")

    # Model bootstrap: train synthetic model if none exists
    if not BEHAVIOR_MODEL_PATH.exists():
        print("[SETUP] No model found. Training synthetic model for demo...\n")
        from src.pose.train_classifier import main as train_synthetic
        train_synthetic()
        print()

    # Auto-discover video file if not specified
    if video_path is None:
        videos_dir = DATA_DIR / "videos"
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if video_files:
            video_path = video_files[0]
            print(f"[INFO] Using video: {video_path.name}")
        else:
            print("[INFO] No video found. Running demo mode.\n")

    # Delegate to main pipeline
    from src.main import run_pipeline, run_demo_mode

    if video_path:
        run_pipeline(video_path)
    else:
        run_demo_mode()


def main():
    """
    CLI argument router.

    Routes execution based on first argument:
      --ui    -> Streamlit web interface
      --train -> Model training pipeline
      <path>  -> Video analysis on specified file
      (none)  -> Demo mode with auto-discovered video
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg in ("--help", "-h"):
            print_help()
            return

        if arg == "--ui":
            launch_ui()
            return

        if arg == "--train":
            run_training()
            return

        # Treat argument as video file path
        video_path = Path(arg)
        if not video_path.exists():
            print(f"[ERROR] File not found: {video_path}\n")
            print_help()
            sys.exit(1)

        run_analysis(video_path)
    else:
        run_analysis()


if __name__ == "__main__":
    main()
