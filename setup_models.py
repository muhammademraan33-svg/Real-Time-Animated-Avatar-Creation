"""
setup_models.py – Download Wav2Lip model weights.

Weights are fetched from Hugging Face Hub (no login required).
Mirrors are attempted in order; the first that succeeds is used.

Usage:
    python setup_models.py              # download wav2lip_gan.pth (recommended)
    python setup_models.py --base       # download wav2lip.pth (faster, lower quality)
    python setup_models.py --all        # download both

The checkpoints land in ./checkpoints/
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

# ── Model registry ─────────────────────────────────────────────────────────────
#
# Each entry is a list of (url, expected_sha256_prefix) pairs tried in order.
# sha256_prefix = first 16 hex chars of the file SHA-256, used as a quick sanity
# check. Set to None to skip verification.
#
MODELS: dict[str, dict] = {
    "wav2lip_gan.pth": {
        "description": "Wav2Lip + GAN discriminator (higher visual quality)",
        "size_mb": 415,
        "sources": [
            # Nekochu/Wav2Lip (primary, publicly accessible)
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth",
            # tensorbanana mirror
            "https://huggingface.co/tensorbanana/wav2lip/resolve/main/wav2lip_gan.pth",
            # numz mirror (older path)
            "https://huggingface.co/numz/wav2lip_studio/resolve/97658f7668e804cd2f74c3b799a9f0ceecbfa8d0/Wav2lip/wav2lip_gan.pth",
        ],
        "sha256_prefix": "ca9ab7b7b812c0e8",   # first 16 chars of known SHA256
    },
    "wav2lip.pth": {
        "description": "Wav2Lip base (faster, slightly lower quality)",
        "size_mb": 415,
        "sources": [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip.pth",
            "https://huggingface.co/tensorbanana/wav2lip/resolve/main/wav2lip.pth",
        ],
        "sha256_prefix": None,
    },
}


# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("ERROR: Missing dependencies. Run:  pip install requests tqdm")
        sys.exit(1)

    print(f"  -> {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    try:
        with open(tmp, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            disable=not show_progress,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
        return True
    except Exception as exc:
        print(f"  ✗ Write error: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def _sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _verify(path: Path, expected: str | None) -> bool:
    if expected is None:
        return True  # skip check
    actual = _sha256_prefix(path)
    ok = actual == expected
    if not ok:
        print(f"  ✗ Checksum mismatch: expected {expected}, got {actual}")
    return ok


def download_model(name: str) -> bool:
    """Try each source URL in order.  Returns True if the file is ready."""
    info = MODELS[name]
    dest = CHECKPOINTS_DIR / name

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        if size_mb > 50:        # sanity: real checkpoint > 50 MB
            print(f"  ✓ {name} already present ({size_mb:.0f} MB)")
            return True
        else:
            print(f"  ⚠ {name} looks incomplete ({size_mb:.1f} MB), re-downloading …")
            dest.unlink()

    print(f"\n{'-'*60}")
    print(f"  Downloading: {name}")
    print(f"  Description: {info['description']}")
    print(f"  Size ~{info['size_mb']} MB")
    print(f"{'-'*60}")

    for url in info["sources"]:
        if _download(url, dest):
            if _verify(dest, info.get("sha256_prefix")):
                print(f"  ✓ {name} saved to {dest}")
                return True
            else:
                dest.unlink(missing_ok=True)

    print(f"\n  ✗ All sources failed for {name}.")
    print(f"  Manual download:")
    print(f"    1. Go to https://github.com/Rudrabha/Wav2Lip (see Readme for GDrive links)")
    print(f"    2. Download {name} and place it in:  {CHECKPOINTS_DIR}/")
    return False


# ── Face detection model for MediaPipe (auto-downloaded by mediapipe itself) ──

def check_mediapipe() -> None:
    """Trigger MediaPipe face detection model download by importing it."""
    try:
        import mediapipe as mp
        _ = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        print("  ✓ MediaPipe face detection model ready")
    except Exception as exc:
        print(f"  ⚠ MediaPipe check: {exc}")
        print("    Ensure mediapipe is installed:  pip install mediapipe")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Wav2Lip model weights")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base", action="store_true", help="Download wav2lip.pth (base model)")
    group.add_argument("--all",  action="store_true", help="Download both GAN and base models")
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINTS_DIR}")

    targets: list[str] = []
    if args.base:
        targets = ["wav2lip.pth"]
    elif args.all:
        targets = ["wav2lip_gan.pth", "wav2lip.pth"]
    else:
        targets = ["wav2lip_gan.pth"]   # default: GAN model

    success = True
    for name in targets:
        ok = download_model(name)
        success = success and ok

    print()
    check_mediapipe()

    print()
    if success:
        print("✓ Setup complete.  Start the server with:")
        print()
        print("    # Local:")
        print("    uvicorn server.main:app --host 0.0.0.0 --port 8000")
        print()
        print("    # Docker:")
        print("    docker compose up")
    else:
        print("⚠ Some models failed to download.  See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
