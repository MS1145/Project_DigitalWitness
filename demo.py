"""
Digital Witness - Demo Launcher

This script launches the Streamlit web UI for demonstrating the system.

Usage:
    python demo.py
"""
import subprocess
import sys
import webbrowser
import time
import threading
from pathlib import Path

def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

def main():
    print()
    print("=" * 60)
    print("  DIGITAL WITNESS - MVP Demo UI")
    print("  Bias-Aware, Explainable Retail Security Assistant")
    print("=" * 60)
    print()

    # Check if model exists
    model_path = Path("models/behavior_classifier.pkl")
    if not model_path.exists():
        print("[WARNING] Model not found!")
        print()
        print("The UI will show the Model Performance tab as empty.")
        print("To train the model, run:")
        print("  python train.py")
        print()
        print("Continuing to launch UI anyway...")
        print()

    print("[INFO] Starting Streamlit server...")
    print()
    print("Features in this demo:")
    print("  - Model Performance Tab: View accuracy, precision, recall, F1")
    print("  - Video Analysis Tab: Upload video + configure POS data")
    print("  - Interactive visualizations and explainable results")
    print()
    print("Opening http://localhost:8501 in your browser...")
    print()
    print("Press Ctrl+C to stop the server.")
    print("-" * 60)
    print()

    # Get the path to streamlit in the virtual environment
    if sys.platform == "win32":
        streamlit_path = Path(".venv/Scripts/streamlit.exe")
    else:
        streamlit_path = Path(".venv/bin/streamlit")

    if not streamlit_path.exists():
        # Fallback to system streamlit
        streamlit_path = "streamlit"

    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

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
        print()
        print("Please install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
