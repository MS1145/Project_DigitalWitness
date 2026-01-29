"""
Digital Witness - Unified Entry Point

Usage:
    python run.py                    # Run demo mode (CLI)
    python run.py --ui               # Launch web interface (Streamlit)
    python run.py --train            # Train model on video dataset
    python run.py path/to/video.mp4  # Analyze a specific video (CLI)

Place videos in: data/videos/
Training videos in: data/training/normal/ and data/training/shoplifting/
"""
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import BEHAVIOR_MODEL_PATH, DEFAULT_VIDEO_PATH, DATA_DIR


def print_banner():
    """Print application banner."""
    print()
    print("=" * 60)
    print("  DIGITAL WITNESS")
    print("  Bias-Aware, Explainable Retail Security Assistant")
    print("=" * 60)
    print()


def print_help():
    """Print usage information."""
    print_banner()
    print("Usage:")
    print("  python run.py              Run CLI demo mode")
    print("  python run.py --ui         Launch web interface")
    print("  python run.py --train      Train model on video dataset")
    print("  python run.py <video.mp4>  Analyze specific video (CLI)")
    print()
    print("Examples:")
    print("  python run.py --ui")
    print("  python run.py --train")
    print("  python run.py data/videos/sample.mp4")
    print()


def launch_ui():
    """Launch Streamlit web interface."""
    print_banner()
    print("[MODE] Web Interface")
    print()

    # Check if model exists
    if not BEHAVIOR_MODEL_PATH.exists():
        print("[WARNING] Model not found!")
        print("The UI will show empty Model Performance tab.")
        print("To train the model, run: python run.py --train")
        print()

    print("Starting Streamlit server...")
    print("Opening http://localhost:8501 in your browser...")
    print()
    print("Press Ctrl+C to stop the server.")
    print("-" * 60)
    print()

    # Find streamlit executable
    if sys.platform == "win32":
        streamlit_path = Path(".venv/Scripts/streamlit.exe")
    else:
        streamlit_path = Path(".venv/bin/streamlit")

    if not streamlit_path.exists():
        streamlit_path = "streamlit"  # Fallback to system streamlit

    # Launch Streamlit
    try:
        subprocess.run([
            str(streamlit_path),
            "run",
            "app.py",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.port=8501"
        ])
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped.")
    except FileNotFoundError:
        print("[ERROR] Streamlit not found!")
        print("Install dependencies: pip install -r requirements.txt")
        sys.exit(1)


def run_training():
    """Run model training."""
    print_banner()
    print("[MODE] Video-Based Model Training")
    print()

    from src.pose.train_video_classifier import main as train_main
    result = train_main()

    if result.get('success'):
        print("\n" + "=" * 60)
        print("  SUCCESS! Model is ready for use.")
        print("=" * 60)
        print("\nNext steps:")
        print("  python run.py --ui         Launch web interface")
        print("  python run.py <video.mp4>  Analyze a video")
    else:
        print("\n" + "=" * 60)
        print("  TRAINING FAILED")
        print("=" * 60)
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def run_analysis(video_path=None):
    """Run video analysis (CLI mode)."""
    print_banner()
    print("[MODE] Video Analysis (CLI)")
    print()

    # Check if model exists, train if not
    if not BEHAVIOR_MODEL_PATH.exists():
        print("[SETUP] No trained model found!")
        print()
        print("Options:")
        print("  1. Train with real videos: python run.py --train")
        print("  2. Continue with synthetic model (demo only)")
        print()
        print("Falling back to demo mode with synthetic data...")
        print()
        from src.pose.train_classifier import main as train_synthetic
        train_synthetic()
        print()

    # Find video if not specified
    if video_path is None:
        videos_dir = DATA_DIR / "videos"
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if video_files:
            video_path = video_files[0]
            print(f"[INFO] Found video: {video_path.name}")
        else:
            print("[INFO] No video found in data/videos/")
            print("[INFO] Running in DEMO MODE with simulated data")
            print()

    # Run analysis
    from src.main import run_pipeline, run_demo_mode

    if video_path:
        run_pipeline(video_path)
    else:
        run_demo_mode()


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg in ["--help", "-h"]:
            print_help()
            return

        if arg == "--ui":
            launch_ui()
            return

        if arg == "--train":
            run_training()
            return

        # Assume it's a video path
        video_path = Path(arg)
        if not video_path.exists():
            print(f"[ERROR] File not found: {video_path}")
            print()
            print_help()
            sys.exit(1)

        run_analysis(video_path)
    else:
        # No arguments - run CLI demo
        run_analysis()


if __name__ == "__main__":
    main()
