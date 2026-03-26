import sys
import subprocess


def main():
    print("\n DIGITAL WITNESS")
    print("  Retail Security Assistant")

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
