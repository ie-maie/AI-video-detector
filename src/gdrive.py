"""
Google Drive utility for downloading model files.

This module provides functions to download the best_model.pth file from Google Drive
to save storage space in the local repository.
"""

import os
from pathlib import Path


# Google Drive file ID for best_model.pth (can be overridden via env var)
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID", "1a_o94Fn5cw3mdtTcJhx2gbjUOlE_7Gor")

# Default model filename
MODEL_FILENAME = "best_model.pth"

# Default local directory to store the model
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Minimum plausible model size to reject placeholders/pointers.
MIN_MODEL_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def _is_git_lfs_pointer(path: Path) -> bool:
    """Return True if the file looks like a Git LFS pointer."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256)
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _is_valid_model_file(path: Path) -> bool:
    """Return True when file exists and appears to be a real model binary."""
    if not path.exists():
        return False
    if path.stat().st_size < MIN_MODEL_SIZE_BYTES:
        return False
    if _is_git_lfs_pointer(path):
        return False
    return True


def _build_download_url() -> str:
    """Construct Google Drive URL from env override or file ID."""
    direct_url = os.getenv("MODEL_URL", "").strip()
    if direct_url:
        return direct_url
    return f"https://drive.google.com/uc?id={MODEL_FILE_ID}"


def download_model_from_gdrive(output_dir: Path = None, force: bool = False) -> Path:
    """
    Download the best_model.pth file from Google Drive.

    Args:
        output_dir: Directory to save the model file. Defaults to {project_root}/models
        force: If True, always download even if file exists. Defaults to False

    Returns:
        Path to the downloaded model file
    """
    if output_dir is None:
        output_dir = DEFAULT_MODEL_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME

    # Reuse an existing valid model.
    if not force and _is_valid_model_file(model_path):
        print(f"Model already exists at {model_path}")
        return model_path

    # Remove invalid placeholders before downloading.
    if model_path.exists():
        print(f"Existing model file is invalid/placeholder. Re-downloading: {model_path}")
        model_path.unlink()

    url = _build_download_url()

    print("Downloading model from Google Drive...")
    print(f"URL: {url}")
    print(f"Destination: {model_path}")

    try:
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError("Missing dependency 'gdown'. Add it to requirements.") from e

        downloaded_path = gdown.download(url, str(model_path), quiet=False, fuzzy=True)

        if downloaded_path and _is_valid_model_file(model_path):
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"Model downloaded successfully. Size: {file_size_mb:.2f} MB")
            return model_path

        raise FileNotFoundError(
            "Model download did not produce a valid .pth file. "
            "Check Google Drive sharing/link."
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def get_model_path(force_download: bool = False) -> Path:
    """
    Get the local path to best_model.pth.
    Downloads from Google Drive if the file is missing or invalid locally.

    Returns:
        Path to the model file
    """
    model_path = DEFAULT_MODEL_DIR / MODEL_FILENAME

    if force_download or not _is_valid_model_file(model_path):
        print(f"Model missing or invalid at {model_path}")
        print("Downloading from Google Drive...")
        return download_model_from_gdrive(force=force_download)

    return model_path


if __name__ == "__main__":
    print("Testing Google Drive download...")
    path = download_model_from_gdrive()
    print(f"Model saved to: {path}")
