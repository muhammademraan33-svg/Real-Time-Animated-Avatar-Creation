"""Generate colab_full_demo.ipynb — full FastAPI + tunnel on Colab GPU."""
import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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


md("""# Full lipsync demo on Colab GPU (FastAPI + browser)

This notebook **starts the same web app** as on your PC: upload a photo, **Start**, speak into the mic. Inference runs on the **Colab GPU** (T4/L4).

## Steps

1. **Runtime → Change runtime type → GPU** (T4 is fine).
2. **Get this repo** onto Colab (clone cell below). If you used **Open in Colab** from GitHub, run the path cell — it often finds the repo under `/content/…`.
3. Run **all cells in order**. Wait until you see a **public https://…trycloudflare.com** URL.
4. **Open that URL in a normal browser tab** (not Colab’s output frame). Allow the **microphone** when asked.
5. Upload a portrait → **Set as Avatar** → **Start** → speak.

**Note:** Cloudflare quick tunnels are temporary and can be slow the first time. For a polished client deliverable, also record a **local GPU** screen capture.

The UI uses **wss://** when the page is HTTPS so WebSockets work behind the tunnel (included in `static/index.html`).
""")

code("""# If needed: clone your GitHub repo (edit URL). Skip if path cell already found the project.
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/avatar
""")

code("""from pathlib import Path

_candidates = [
    Path("/content/avatar"),
    Path("/content/Real-Time-Animated-Avatar-Creation-Contest"),
]
PROJECT = None
for c in _candidates:
    if (c / "server" / "main.py").is_file() and (c / "static" / "index.html").is_file():
        PROJECT = c.resolve()
        break

if PROJECT is None:
    raise FileNotFoundError(
        "Clone or upload the repo so server/main.py and static/index.html exist "
        "(e.g. under /content/avatar)."
    )
print("PROJECT =", PROJECT)
""")

code("""import subprocess
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(
    ["apt-get", "install", "-qq", "-y", "ffmpeg", "libsndfile1", "libgl1-mesa-glx"],
    check=False,
)
print("apt deps OK")
""")

code("""import subprocess
import sys

# GPU PyTorch first, then requirements from your repo
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24,<2.0", "huggingface_hub", "tqdm", "requests",
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt"),
])
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
""")

code("""import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ckpt_dir = PROJECT / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = hf_hub_download(
    repo_id="Nekochu/Wav2Lip",
    filename="wav2lip_gan.pth",
    local_dir=str(ckpt_dir),
)
os.environ["CHECKPOINT_PATH"] = str(Path(checkpoint_path).resolve())
print("CHECKPOINT_PATH =", os.environ["CHECKPOINT_PATH"])
""")

code("""import os
import subprocess
import time

import requests

# Stop previous runs if you re-execute the notebook
subprocess.run(["pkill", "-f", "uvicorn"], check=False)
subprocess.run(["pkill", "-f", "cloudflared"], check=False)
time.sleep(1)

log_path = "/tmp/uvicorn_avatar.log"
log = open(log_path, "w")
env = {**os.environ, "CHECKPOINT_PATH": os.environ.get("CHECKPOINT_PATH", "")}
proc = subprocess.Popen(
    [os.sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=str(PROJECT),
    stdout=log,
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)
print("uvicorn PID:", proc.pid, "| log:", log_path)

for i in range(180):
    time.sleep(2)
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code == 200:
            print("Server ready:", r.json())
            break
    except Exception as e:
        if i > 0 and i % 15 == 0:
            print("Still waiting for /health … (GPU model load can take a few minutes)", e)
else:
    log.seek(0)
    print(log.read()[-4000:])
    raise RuntimeError("Server did not become healthy in time — see log above")
""")

code("""import os
import re
import subprocess
import threading
import time
import urllib.request

# Cloudflare quick tunnel — no signup; URL changes each session
BIN = "/content/cloudflared"
if not os.path.isfile(BIN):
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, BIN)
    os.chmod(BIN, 0o755)

print("Starting tunnel (watch for https://…trycloudflare.com)…\\n")

p = subprocess.Popen(
    [BIN, "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

public = {"url": None}
# [.] matches a literal dot (no escape issues in generated notebook)
URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+[.]trycloudflare[.]com")


def reader():
    for line in p.stdout:
        print(line, end="")
        m = URL_RE.search(line)
        if m:
            public["url"] = m.group(0)

threading.Thread(target=reader, daemon=True).start()

for _ in range(90):
    time.sleep(1)
    if public["url"]:
        break

if public["url"]:
    u = public["url"]
    print("\\n" + "=" * 60)
    print("OPEN IN BROWSER:", u)
    print("=" * 60)
else:
    print("Could not parse tunnel URL — read Cloudflare lines above for https://…trycloudflare.com")
""")

out = Path(__file__).resolve().parent / "colab_full_demo.ipynb"
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", out)
