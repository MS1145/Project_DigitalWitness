"""
Digital Witness - One-Click Runner

Usage:
    python run.py                    # Run demo mode (no video needed)
    python run.py path/to/video.mp4  # Analyze a specific video
    python run.py --train            # Train model on video dataset

Place videos in: data/videos/
Training videos in: data/training/normal/ and data/training/shoplifting/
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import BEHAVIOR_MODEL_PATH, DEFAULT_VIDEO_PATH, DATA_DIR


def main():
    print("=" * 60)
    print("  DIGITAL WITNESS - Retail Security Analysis")
    print("=" * 60)
    print()

    # Check for training mode
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("[MODE] Video-Based Model Training")
        print()
        from src.pose.train_video_classifier import main as train_main
        result = train_main()
        if result.get('success'):
            print("\nTraining completed successfully!")
        else:
            print(f"\nTraining failed: {result.get('error', 'Unknown error')}")
        return

    # Step 1: Check if model exists, train if not
    if not BEHAVIOR_MODEL_PATH.exists():
        print("[SETUP] No trained model found!")
        print()
        print("To train the model using your video dataset, run:")
        print("  python run.py --train")
        print()
        print("Or place training videos in:")
        print("  - data/training/normal/     (normal behavior videos)")
        print("  - data/training/shoplifting/ (shoplifting videos)")
        print()
        print("Falling back to demo mode with synthetic data...")
        print()
        from src.pose.train_classifier import main as train_synthetic
        train_synthetic()
        print()

    # Step 2: Check for video
    video_path = None
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            sys.exit(1)
    else:
        # Check default location
        videos_dir = DATA_DIR / "videos"
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

        if video_files:
            video_path = video_files[0]
            print(f"[INFO] Found video: {video_path.name}")
        else:
            print("[INFO] No video found in data/videos/")
            print("[INFO] Running in DEMO MODE with simulated data")
            print()
            print("To analyze a real video:")
            print("  1. Place your .mp4 file in: data/videos/")
            print("  2. Run: python run.py")
            print()
            print("Or specify a path directly:")
            print("  python run.py C:/path/to/your/video.mp4")
            print()

    # Step 3: Run analysis
    from src.main import run_pipeline, run_demo_mode

    if video_path:
        run_pipeline(video_path)
    else:
        run_demo_mode()


if __name__ == "__main__":
    main()
