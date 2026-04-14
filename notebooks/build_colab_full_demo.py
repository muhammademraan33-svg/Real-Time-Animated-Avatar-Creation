"""Generate colab_full_demo.ipynb — FastAPI + Cloudflare tunnel; fully automated on Colab."""
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


md("""# Full lipsync demo on Colab GPU

**Only manual steps in Colab:** **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

When the last cell prints **OPEN IN BROWSER:** `https://…trycloudflare.com` — open that link in a **new tab**, allow the **microphone**, upload a photo → **Set as Avatar** → **Start** → speak.

The notebook clones your repo if needed, installs dependencies, starts FastAPI on GPU, and opens a public HTTPS tunnel (no signup).
""")

code("""# Auto-setup: find full repo (server + static) or shallow-clone to /content/avatar
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/muhammademraan33-svg/Real-Time-Animated-Avatar-Creation.git"


def _is_full_repo(p: Path) -> bool:
    return (
        (p / "server" / "main.py").is_file()
        and (p / "static" / "index.html").is_file()
    )


def get_project() -> Path:
    preferred = [
        Path("/content/avatar"),
        Path("/content/Real-Time-Animated-Avatar-Creation"),
    ]
    for p in preferred:
        if _is_full_repo(p):
            print("Using existing repo:", p)
            return p.resolve()
    for p in Path("/content").iterdir():
        if p.is_dir() and _is_full_repo(p):
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

code("""import subprocess

subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(
    ["apt-get", "install", "-qq", "-y", "git", "ffmpeg", "libsndfile1", "libgl1-mesa-glx"],
    check=False,
)
print("apt deps OK")
""")

code("""import subprocess
import sys

if "get_project" not in globals():
    raise RuntimeError("Run all cells from the top (Runtime → Run all), or run the setup cell first.")

PROJECT = get_project()

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

if "get_project" not in globals():
    raise RuntimeError("Run all cells from the top.")

PROJECT = get_project()

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

if "get_project" not in globals():
    raise RuntimeError("Run all cells from the top.")

PROJECT = get_project()

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

if "get_project" not in globals():
    raise RuntimeError("Run all cells from the top.")

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
    print("Could not parse tunnel URL — read Cloudflare output above for https://…trycloudflare.com")
""")

out = Path(__file__).resolve().parent / "colab_full_demo.ipynb"
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", out)
