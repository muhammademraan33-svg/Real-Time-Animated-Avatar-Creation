# ── Stage 1: base CUDA + Python image ─────────────────────────────────────────
# CUDA 12.1 + cuDNN 8, Ubuntu 22.04
# Choose a CUDA version matching your driver (nvidia-smi → check "CUDA Version:")
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    # OpenCV system libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Audio (libsndfile for librosa)
    libsndfile1 \
    ffmpeg \
    # Utilities
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python alias ──────────────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3     1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps (cached layer) ────────────────────────────────────────
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first (separate to use the correct index)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps, then force-pin numpy<2.0 so mediapipe doesn't crash
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0"

# ── Copy source code ───────────────────────────────────────────────────────────
COPY server/   ./server/
COPY static/   ./static/
COPY setup_models.py .

# ── Checkpoint volume mount point ─────────────────────────────────────────────
# Weights are NOT baked into the image; mount them at runtime:
#   docker run -v /path/to/checkpoints:/app/checkpoints ...
RUN mkdir -p checkpoints

# ── Environment variables ──────────────────────────────────────────────────────
ENV CHECKPOINT_PATH=/app/checkpoints/wav2lip_gan.pth
ENV PORT=8000

# ── Expose & run ──────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
