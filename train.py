"""
Digital Witness - Model Training Script

This script trains the behavior classifier using real video data
from the Kaggle shoplifting detection dataset.

Usage:
    python train.py

Training data structure:
    data/training/
    ├── normal/           # Videos of normal shopping behavior
    │   ├── normal-1.mp4
    │   ├── normal-2.mp4
    │   └── ...
    └── shoplifting/      # Videos of shoplifting behavior
        ├── shoplifting-1.mp4
        ├── shoplifting-2.mp4
        └── ...

The script will:
1. Process all videos and extract pose-based features
2. Train a Random Forest classifier
3. Evaluate with cross-validation
4. Save the model to models/behavior_classifier.pkl
5. Generate visualizations (confusion matrix, feature importance, etc.)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pose.train_video_classifier import main

if __name__ == "__main__":
    result = main()

    if result.get('success'):
        print("\n" + "=" * 60)
        print("  SUCCESS! Model is ready for use.")
        print("=" * 60)
        print("\nTo analyze a video, run:")
        print("  python run.py path/to/video.mp4")
        print("\nOr place a video in data/videos/ and run:")
        print("  python run.py")
    else:
        print("\n" + "=" * 60)
        print("  TRAINING FAILED")
        print("=" * 60)
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
