"""
Model downloader for cloud deployment.
Downloads models from external storage if not present locally.
"""
import os
from pathlib import Path
import urllib.request


# Model URLs - UPDATE THESE with your hosted model URLs
MODEL_URLS = {
    "lstm_classifier.pt": None,  # Set your URL here (Google Drive, Hugging Face, etc.)
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
}


def download_file(url: str, destination: Path, show_progress: bool = True):
    """Download a file from URL."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {destination.name}...")

    if show_progress:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r  Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print()  # New line after progress
    else:
        urllib.request.urlretrieve(url, destination)

    print(f"  Saved to: {destination}")


def ensure_models_exist(models_dir: Path):
    """
    Ensure all required models exist, downloading if necessary.

    Args:
        models_dir: Path to models directory
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, url in MODEL_URLS.items():
        model_path = models_dir / model_name

        if model_path.exists():
            print(f"Model found: {model_name}")
            continue

        if url is None:
            print(f"WARNING: {model_name} not found and no URL configured!")
            print(f"  Please upload {model_name} to the models/ folder")
            print(f"  Or set MODEL_URLS['{model_name}'] in model_downloader.py")
            continue

        try:
            download_file(url, model_path)
        except Exception as e:
            print(f"ERROR downloading {model_name}: {e}")


def get_gdrive_download_url(file_id: str) -> str:
    """
    Convert Google Drive file ID to direct download URL.

    To get file_id:
    1. Upload file to Google Drive
    2. Right-click > Share > Anyone with link
    3. Copy link: https://drive.google.com/file/d/FILE_ID/view
    4. Extract FILE_ID from the URL
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"
