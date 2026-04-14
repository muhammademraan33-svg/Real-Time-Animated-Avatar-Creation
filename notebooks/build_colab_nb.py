"""Emit colab_gpu_smoke_test.ipynb — GPU latency benchmark; fully automated on Colab."""
import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [],
}


def md(s: str):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": f"c{len(nb['cells'])}"},
        "source": s.splitlines(keepends=True),
    })


def code(s: str):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": f"c{len(nb['cells'])}"},
        "outputs": [],
        "source": s.splitlines(keepends=True),
    })


md("""# Wav2Lip GPU smoke test

**Only manual steps in Colab:** **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

This notebook finds or clones the GitHub repo, installs CUDA PyTorch, downloads `wav2lip_gan.pth`, and prints **inference latency (ms)** on the Colab GPU.

**Expected:** `CUDA: True`, GPU name, mean **~10–40 ms** on T4.
""")

code("""# Auto-setup: find repo under /content or shallow-clone into /content/avatar
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/muhammademraan33-svg/Real-Time-Animated-Avatar-Creation.git"


def get_project() -> Path:
    # Locate server/wav2lip_model.py under /content, else shallow-clone.
    preferred = [
        Path("/content/avatar"),
        Path("/content/Real-Time-Animated-Avatar-Creation"),
    ]
    for p in preferred:
        if (p / "server" / "wav2lip_model.py").is_file():
            print("Using existing repo:", p)
            return p.resolve()
    for p in Path("/content").iterdir():
        if p.is_dir() and (p / "server" / "wav2lip_model.py").is_file():
            print("Using existing repo:", p)
            return p.resolve()
    dest = Path("/content/avatar")
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
        check=True,
    )
    print("Cloned to", dest)
    return dest.resolve()


PROJECT = get_project()
print("PROJECT =", PROJECT)
""")

code("""# PyTorch CUDA + helpers
%pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
%pip install -q "numpy>=1.24,<2.0" huggingface_hub tqdm
""")

code("""import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU — enable Runtime → Change runtime type → GPU, then Run all again.")
""")

code("""import statistics
import time
import sys
from pathlib import Path

import torch

if "get_project" not in globals():
    raise RuntimeError("Run all cells from the top (Runtime → Run all), or run the setup cell first.")

PROJECT = get_project()
sys.path.insert(0, str(PROJECT / "server"))

from huggingface_hub import hf_hub_download
from wav2lip_model import load_wav2lip

ckpt_dir = PROJECT / "checkpoints"
checkpoint_path = hf_hub_download(
    repo_id="Nekochu/Wav2Lip",
    filename="wav2lip_gan.pth",
    local_dir=str(ckpt_dir),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_wav2lip(checkpoint_path, device=device)

mel = torch.zeros(1, 1, 80, 16, device=device)
face = torch.zeros(1, 6, 96, 96, device=device)

with torch.no_grad():
    for _ in range(5):
        _ = model(mel, face)
if device == "cuda":
    torch.cuda.synchronize()

times_ms = []
with torch.no_grad():
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model(mel, face)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

mean_ms = statistics.mean(times_ms)
p50_ms = statistics.median(times_ms)

print("=" * 60)
print("Wav2Lip GAN — single forward pass (batch=1)")
print("Device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
print(f"Mean latency: {mean_ms:.1f} ms | median: {p50_ms:.1f} ms")
print("=" * 60)
""")

out = Path(__file__).resolve().parent / "colab_gpu_smoke_test.ipynb"
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", out)
