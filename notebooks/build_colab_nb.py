"""One-off helper to emit colab_gpu_smoke_test.ipynb — run if you edit cells in code."""
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
    nb["cells"].append({"cell_type": "markdown", "metadata": {"id": f"c{len(nb['cells'])}"}, "source": s.splitlines(keepends=True)})

def code(s: str):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {"id": f"c{len(nb['cells'])}"}, "outputs": [], "source": s.splitlines(keepends=True)})

md("""# Wav2Lip GPU smoke test (proof of concept)

This notebook **runs the same Wav2Lip checkpoint on a real NVIDIA GPU** (Colab T4/L4, etc.) and prints **measured inference latency in milliseconds**. Use it when reviewers need evidence beyond “it will work on GPU.”

## Before you run

1. **Runtime → Change runtime type → GPU** (required).
2. **Get the project code on Colab** — pick one:
   - **A)** Push this repo to GitHub, then run the `git clone` cell below (edit the URL).
   - **B)** Zip your project folder, **Upload** in Colab Files, unzip to `/content/avatar` so `server/wav2lip_model.py` exists.

3. The checkpoint `wav2lip_gan.pth` is downloaded automatically from Hugging Face (same source as `setup_models.py`).

---
**Expected:** `CUDA: True`, GPU name, mean inference **~10–40 ms** on T4 (batch size 1). CPU would be ~80–200 ms.
""")

code("""# Optional: clone your GitHub repo (replace URL). Skip if you uploaded a zip to /content/avatar
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/avatar
""")

code("""import os
from pathlib import Path

# Project root on Colab (must contain server/wav2lip_model.py)
PROJECT = Path("/content/avatar")
if not (PROJECT / "server" / "wav2lip_model.py").is_file():
    raise FileNotFoundError(
        "Missing server/wav2lip_model.py. Clone this repo to /content/avatar "
        "or unzip the project there, then re-run."
    )
print("Project OK:", PROJECT.resolve())
""")

code("""# PyTorch with CUDA 12.1 wheels (matches Dockerfile / consumer GPUs)
%pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
%pip install -q "numpy>=1.24,<2.0" huggingface_hub tqdm
""")

code("""import sys
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU. Enable Runtime → Change runtime type → GPU.")
""")

code("""import statistics
import time
import sys
from pathlib import Path

import torch

PROJECT = Path("/content/avatar")
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

# Dummy inputs matching real-time inference (1 mel frame + 96x96 face)
mel = torch.zeros(1, 1, 80, 16, device=device)
face = torch.zeros(1, 6, 96, 96, device=device)

# Warm-up (CUDA kernels compile on first runs)
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
if device == "cpu":
    print("For contest POC, re-run with a GPU runtime to show <300 ms end-to-end is realistic.")
""")

out = Path(__file__).resolve().parent / "colab_gpu_smoke_test.ipynb"
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", out)
