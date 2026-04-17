"""
Audio processing utilities for real-time Wav2Lip inference.

Mel-spectrogram parameters match the original Wav2Lip training setup:
  sample_rate  = 16 000 Hz
  n_fft        = 800  (50 ms window)
  hop_length   = 200  (12.5 ms hop → 80 mel frames / second)
  n_mels       = 80
  fmin         = 55 Hz
  fmax         = 7 600 Hz

Video FPS = 25  →  each video frame = 40 ms = 3.2 mel frames.
Each Wav2Lip mel chunk = 16 mel frames = 200 ms (≈ 5 video frames).
For single-frame real-time mode we use a 200 ms *sliding window* of mel
and advance by 3.2 frames per output frame.
"""
from __future__ import annotations

import threading
import numpy as np
import librosa
from scipy import signal

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000
N_FFT: int = 800
HOP_LENGTH: int = 200
WIN_LENGTH: int = 800
N_MELS: int = 80
FMIN: float = 55.0
FMAX: float = 7_600.0
MIN_LEVEL_DB: float = -100.0
REF_LEVEL_DB: float = 20.0
PREEMPHASIS: float = 0.97
MAX_ABS_VALUE: float = 4.0

FPS: int = 25
MEL_STEP_SIZE: int = 16        # mel frames per chunk fed to Wav2Lip
MEL_FRAMES_PER_SEC: float = SAMPLE_RATE / HOP_LENGTH          # = 80
MEL_FRAMES_PER_VIDEO_FRAME: float = MEL_FRAMES_PER_SEC / FPS  # = 3.2

# 200 ms of audio at 16 kHz (must cover MEL_STEP_SIZE mel frames fully)
AUDIO_BUFFER_SAMPLES: int = MEL_STEP_SIZE * HOP_LENGTH        # = 3200

# ── Mel filter bank (built once, reused) ──────────────────────────────────────
_mel_basis: np.ndarray | None = None
_mel_basis_lock = threading.Lock()


def _get_mel_basis() -> np.ndarray:
    global _mel_basis
    if _mel_basis is None:
        with _mel_basis_lock:
            if _mel_basis is None:
                _mel_basis = librosa.filters.mel(
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    n_mels=N_MELS,
                    fmin=FMIN,
                    fmax=FMAX,
                )
    return _mel_basis


# ── Core mel computation ───────────────────────────────────────────────────────

def _amp_to_db(x: np.ndarray) -> np.ndarray:
    min_level = np.exp(MIN_LEVEL_DB / 20.0 * np.log(10))
    return 20.0 * np.log10(np.maximum(min_level, x)) - REF_LEVEL_DB


def _normalize(S: np.ndarray) -> np.ndarray:
    """
    Match Wav2Lip training normalization (symmetric mels in [-4, 4]).
    """
    return np.clip(
        (2.0 * MAX_ABS_VALUE) * ((S - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE,
        -MAX_ABS_VALUE,
        MAX_ABS_VALUE,
    )


def wav_to_mel(wav: np.ndarray) -> np.ndarray:
    """
    Compute normalised mel spectrogram from a 1-D float32 waveform.

    Args:
        wav: 1-D float32 array, values in [-1, 1], sample rate = 16 000 Hz.

    Returns:
        mel: (N_MELS=80, T) float32 array, values in [-4, 4] (Wav2Lip scale).
    """
    wav = wav.astype(np.float32)
    # Match original Wav2Lip preprocessing: pre-emphasis before STFT.
    wav = signal.lfilter([1.0, -PREEMPHASIS], [1.0], wav).astype(np.float32)
    D = librosa.stft(
        y=wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
    )
    S = np.abs(D)
    mel_raw = np.dot(_get_mel_basis(), S)          # (80, T)
    mel_db = _amp_to_db(mel_raw)
    return _normalize(mel_db).astype(np.float32)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Convert raw 16-bit little-endian PCM bytes to float32 in [-1, 1]."""
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


# ── Rolling audio buffer ───────────────────────────────────────────────────────

class AudioBuffer:
    """
    Thread-safe rolling buffer that accumulates PCM audio and emits mel chunks
    aligned to video frames (25 fps → one chunk per 40 ms).

    Usage::

        buf = AudioBuffer()
        buf.push(pcm_float32_chunk)   # call as audio arrives
        chunks = buf.drain_mel_chunks()  # list of (80, 16) arrays
    """

    def __init__(self):
        # Keep 200 ms of audio (MEL_STEP_SIZE frames) as the mel window
        self._buffer = np.zeros(AUDIO_BUFFER_SAMPLES, dtype=np.float32)
        # Accumulate new samples here until we have enough for a video frame
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._lock = threading.Lock()

        # Fractional mel frame accumulator (handles non-integer steps)
        self._mel_frame_accum: float = 0.0
        self._primed = False

    # Samples needed per video frame at 16 kHz / 25 fps = 640 samples
    SAMPLES_PER_FRAME: int = SAMPLE_RATE // FPS  # 640

    def push(self, audio: np.ndarray) -> None:
        """Add float32 PCM samples to the buffer."""
        with self._lock:
            self._pending.append(audio.astype(np.float32))
            self._pending_samples += len(audio)

    def drain_mel_chunks(self) -> list[np.ndarray]:
        """
        Return a list of (80, 16) mel chunks—one per video frame worth of audio.
        Call this from the processing loop at ~25 fps or whenever audio arrives.
        """
        with self._lock:
            if self._pending_samples < self.SAMPLES_PER_FRAME:
                return []

            # Collect pending samples
            new_samples = np.concatenate(self._pending)
            self._pending.clear()
            self._pending_samples = 0

        chunks: list[np.ndarray] = []
        offset = 0

        while offset + self.SAMPLES_PER_FRAME <= len(new_samples):
            frame_pcm = new_samples[offset: offset + self.SAMPLES_PER_FRAME]
            offset += self.SAMPLES_PER_FRAME

            if not self._primed:
                # Avoid the first 200 ms being mostly silence.
                # Bootstrap with the earliest speech chunk repeated so the first
                # visible mouth movement appears quickly after the user starts talking.
                reps = AUDIO_BUFFER_SAMPLES // len(frame_pcm)
                self._buffer[:] = np.tile(frame_pcm, reps)
                self._primed = True
            else:
                # Slide the window: drop oldest, append newest
                self._buffer = np.roll(self._buffer, -self.SAMPLES_PER_FRAME)
                self._buffer[-self.SAMPLES_PER_FRAME:] = frame_pcm

            mel = wav_to_mel(self._buffer)  # (80, T_total)
            # Take the last MEL_STEP_SIZE columns (most recent 200 ms)
            if mel.shape[1] >= MEL_STEP_SIZE:
                chunk = mel[:, -MEL_STEP_SIZE:].copy()
            else:
                # Pad if needed (should not happen in steady state)
                pad = MEL_STEP_SIZE - mel.shape[1]
                chunk = np.pad(mel, ((0, 0), (pad, 0)), mode="edge")

            chunks.append(chunk)

        # Put leftover samples back
        if offset < len(new_samples):
            leftover = new_samples[offset:]
            with self._lock:
                self._pending.insert(0, leftover)
                self._pending_samples += len(leftover)

        return chunks

    def reset(self) -> None:
        """Clear all buffers (e.g. when avatar changes)."""
        with self._lock:
            self._buffer[:] = 0.0
            self._pending.clear()
            self._pending_samples = 0
            self._mel_frame_accum = 0.0
            self._primed = False
