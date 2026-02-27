"""
Digital Witness — Entry Point

Usage:
    python run.py        Launch the Streamlit web interface
"""
import sys
import subprocess


def main():
    print("\n" + "=" * 50)
    print("  DIGITAL WITNESS")
    print("  Retail Security Assistant")
    print("=" * 50)
    print("\nStarting web interface at http://localhost:8501")
    print("Press Ctrl+C to stop.\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.port=8501",
        ])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("[ERROR] Streamlit not found. Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
