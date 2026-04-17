"""
AvatarEngine – real-time lipsync orchestrator.

Latency budget (RTX 3060 or equivalent):
  ┌─────────────────────────────────────┬──────────┐
  │ Stage                               │ Time     │
  ├─────────────────────────────────────┼──────────┤
  │ Audio buffer fill (200 ms window)   │ 200 ms   │ ← one-time startup delay
  │ Mel computation (librosa)           │ ~3 ms    │
  │ Wav2Lip GPU inference (batch=1)     │ ~15 ms   │
  │ JPEG encode + WebSocket send        │ ~3 ms    │
  ├─────────────────────────────────────┼──────────┤
  │ Steady-state end-to-end latency     │ ~220 ms  │ ✓ < 300 ms
  └─────────────────────────────────────┴──────────┘
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .audio_utils import AudioBuffer, pcm16_to_float, MEL_STEP_SIZE
from .face_utils import FacePreprocessed, FaceProcessor, frame_to_jpeg, resize_keep_aspect
from .wav2lip_model import Wav2Lip, load_wav2lip

logger = logging.getLogger(__name__)


class AvatarEngine:
    """
    Ties together face preprocessing, audio buffering, and Wav2Lip inference.

    Typical flow:
        engine = await AvatarEngine.create("checkpoints/wav2lip_gan.pth")
        engine.set_avatar(image_bgr)   # call once per photo
        frames = engine.process_audio(pcm_bytes)   # call as audio streams in
    """

    CHECKPOINT_DEFAULT = Path(__file__).parent.parent / "checkpoints" / "wav2lip_gan.pth"

    def __init__(self, model: Wav2Lip, device: str, face_processor: FaceProcessor):
        self._model = model
        self._device = device
        self._face_proc = face_processor
        self._audio_buf = AudioBuffer()
        self._avatar: FacePreprocessed | None = None

        # Pre-allocate reusable GPU tensors to avoid allocation overhead
        self._face_tensor: torch.Tensor | None = None
        self._mel_tensor: torch.Tensor | None = None
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        self._use_autocast = device == "cuda"
        self._jpeg_quality = int(os.environ.get("JPEG_QUALITY", "72"))

        # Latency tracking
        self._last_latency_ms: float = 0.0
        self._total_frames: int = 0

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> "AvatarEngine":
        """
        Load the Wav2Lip model and return a ready-to-use AvatarEngine.

        Args:
            checkpoint_path: Path to wav2lip_gan.pth (or wav2lip.pth).
                             Defaults to checkpoints/wav2lip_gan.pth.
            device: "cuda", "cpu", or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        path = Path(checkpoint_path) if checkpoint_path else cls.CHECKPOINT_DEFAULT
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {path}\n"
                "Run  python setup_models.py  to download the weights."
            )

        logger.info("Loading Wav2Lip from %s on %s …", path, device)
        model = load_wav2lip(str(path), device=device)
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            model = model.half()

        # Warm up model with a dummy forward pass so first real inference is fast
        warm_dtype = torch.float16 if device == "cuda" else torch.float32
        dummy_audio = torch.zeros(1, 1, 80, 16, device=device, dtype=warm_dtype)
        dummy_face = torch.zeros(1, 6, 96, 96, device=device, dtype=warm_dtype)
        with torch.no_grad():
            _ = model(dummy_audio, dummy_face)
        logger.info("Wav2Lip warmed up. Device: %s", device)

        return cls(model=model, device=device, face_processor=FaceProcessor())

    # ── Avatar setup ──────────────────────────────────────────────────────────

    def set_avatar(self, image_bgr: np.ndarray) -> None:
        """
        Pre-process a new avatar photo.  Called once per uploaded image.

        Args:
            image_bgr: OpenCV BGR image (H, W, 3).

        Raises:
            ValueError: if no face is found.
        """
        # Bound avatar frame size to keep compositing/JPEG encode fast.
        max_side = int(os.environ.get("MAX_AVATAR_SIDE", "384"))
        image_bgr = resize_keep_aspect(image_bgr, max_side, max_side)

        self._avatar = self._face_proc.prepare(image_bgr)
        self._audio_buf.reset()
        self._build_face_tensor()
        logger.info("Avatar set. Face bbox: %s", self._avatar.crop[:4])

    def has_avatar(self) -> bool:
        return self._avatar is not None

    # ── Real-time processing ──────────────────────────────────────────────────

    def push_audio(self, raw_bytes: bytes) -> None:
        """
        Push raw 16-bit PCM audio (16 kHz mono) into the rolling audio buffer.
        Thread-safe – call from any thread / the asyncio loop.
        """
        if self._avatar is None:
            return
        samples = pcm16_to_float(raw_bytes)
        self._audio_buf.push(samples)

    # ── Async streaming loop (used by WebSocket handler) ──────────────────────

    async def stream_frames(
        self,
        stop_event: asyncio.Event,
        send_frame: Callable[[bytes, float], Awaitable[None]],
    ) -> None:
        """
        Continuously infer frames at maximum CPU/GPU speed and call *send_frame*.

        No queue is used – audio is fed directly via push_audio() and this loop
        drains the rolling mel buffer as fast as the hardware allows.  This
        ensures lip-sync always tracks *current* speech instead of a backlogged
        queue that can be seconds behind.

        Args:
            stop_event : set this to stop the loop.
            send_frame : async callable(jpeg_bytes, latency_ms).
        """
        loop = asyncio.get_running_loop()
        frame_count = 0
        logger.info("stream_frames started (bufferless mode).")

        while not stop_event.is_set():
            if not self.has_avatar():
                await asyncio.sleep(0.02)
                continue

            try:
                jpeg, latency = await loop.run_in_executor(None, self._infer_latest)
            except Exception as exc:
                logger.exception("Inference error (frame skipped): %s", exc)
                await asyncio.sleep(0.02)
                continue

            if jpeg is None:
                # Not enough new audio yet — yield to the event loop briefly
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            if frame_count <= 5 or frame_count % 50 == 0:
                logger.info("Frame #%d sent, latency=%.0f ms", frame_count, latency)
            try:
                await send_frame(jpeg, latency)
            except Exception:
                pass  # WebSocket may have closed; stop_event will be set shortly

    def _infer_latest(self) -> tuple[bytes | None, float]:
        """
        Drain one mel chunk from the rolling buffer and run inference.
        Returns (None, 0) if no new audio has arrived since the last call.
        """
        mel_chunks = self._audio_buf.drain_mel_chunks()
        if not mel_chunks:
            return None, 0.0

        t0 = time.perf_counter()
        frame_bgr = self._infer_frame(mel_chunks[-1])   # use the most recent chunk
        jpeg = frame_to_jpeg(frame_bgr, quality=self._jpeg_quality)
        latency = (time.perf_counter() - t0) * 1000
        self._last_latency_ms = latency
        self._total_frames += 1
        return jpeg, latency

    # ── Internal inference ────────────────────────────────────────────────────

    def _build_face_tensor(self) -> None:
        """Pre-build the (1, 6, 96, 96) face tensor and keep it on the GPU."""
        assert self._avatar is not None
        face_np = self._avatar.face_input  # (96, 96, 6) float32 [0,1]
        # (6, 96, 96) → (1, 6, 96, 96)
        face_t = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0)
        self._face_tensor = face_t.to(self._device, dtype=self._dtype, non_blocking=True)
        self._mel_tensor = torch.zeros((1, 1, 80, MEL_STEP_SIZE), device=self._device, dtype=self._dtype)

    @torch.no_grad()
    def _infer_frame(self, mel_chunk: np.ndarray) -> np.ndarray:
        """
        Run one Wav2Lip forward pass and composite the result back into the
        full avatar frame.

        Args:
            mel_chunk: (80, 16) float32 mel spectrogram.

        Returns:
            Composited BGR frame (full avatar image with animated mouth).
        """
        assert self._avatar is not None
        assert self._face_tensor is not None
        assert self._mel_tensor is not None

        # Reuse preallocated mel tensor to avoid per-frame allocations/copies.
        mel_src = torch.from_numpy(mel_chunk).to(self._dtype)
        self._mel_tensor[0, 0].copy_(mel_src, non_blocking=True)

        # Forward pass
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_autocast
            else nullcontext()
        )
        with autocast_ctx:
            pred = self._model(self._mel_tensor, self._face_tensor)  # (1, 3, 96, 96)
        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (96, 96, 3) [0,1]

        # Composite back into the full avatar frame
        composited = self._face_proc.composite(
            base_frame=self._avatar.original_frame.copy(),
            generated_face_01=pred_np,
            crop=self._avatar.crop,
        )
        return composited

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def device(self) -> str:
        return self._device
