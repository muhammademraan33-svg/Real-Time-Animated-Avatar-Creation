"""
Real-Time Lipsync Avatar Server
================================
FastAPI application exposing:

  POST /api/avatar          – upload avatar photo (JPEG/PNG)
  WS   /ws/lipsync          – bidirectional: send PCM audio → receive JPEG frames
  GET  /stream/mjpeg        – MJPEG endpoint (e.g. for OBS virtual camera)
  GET  /health              – health check + GPU info
  GET  /                    – serve static demo UI
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .avatar_engine import AvatarEngine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(BASE_DIR / "checkpoints" / "wav2lip_gan.pth"),
)

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup (runs in thread pool to avoid blocking event loop)."""
    global engine

    def _load():
        return AvatarEngine.create(checkpoint_path=CHECKPOINT_PATH)

    try:
        engine = await asyncio.get_running_loop().run_in_executor(None, _load)
        logger.info("AvatarEngine ready. Device: %s", engine.device)
    except FileNotFoundError as exc:
        logger.error("⚠  %s", exc)
        logger.error("⚠  Run  python setup_models.py  then restart the server.")
    yield


app = FastAPI(
    title="Real-Time Lipsync Avatar",
    version="1.0.0",
    description="Single photo + live audio → real-time animated avatar (< 300 ms latency)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global engine (loaded once at startup) ────────────────────────────────────
engine: AvatarEngine | None = None

# Latest frame for MJPEG stream (shared state)
_latest_frame: bytes | None = None
_latest_frame_event = asyncio.Event()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Real-Time Lipsync Avatar</h1><p>UI not found.</p>")


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 1),
        }

    return {
        "status": "ok" if engine else "model_not_loaded",
        "model_loaded": engine is not None,
        "has_avatar": engine.has_avatar() if engine else False,
        "total_frames_generated": engine.total_frames if engine else 0,
        "last_inference_ms": round(engine.last_latency_ms, 1) if engine else None,
        "device": engine.device if engine else ("cuda" if torch.cuda.is_available() else "cpu"),
        "gpu": gpu_info or None,
    }


@app.post("/api/avatar")
async def upload_avatar(file: UploadFile = File(...)):
    """Upload a portrait photo to set as the avatar."""
    if engine is None:
        raise HTTPException(503, "Model not loaded. Run setup_models.py first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG or PNG).")

    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(400, "Could not decode image. Send a valid JPEG or PNG.")

    try:
        await asyncio.get_running_loop().run_in_executor(None, engine.set_avatar, image_bgr)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return {"status": "ok", "message": "Avatar set successfully."}


# ── WebSocket: audio-in / frame-out ───────────────────────────────────────────

@app.websocket("/ws/lipsync")
async def ws_lipsync(websocket: WebSocket):
    """
    WebSocket protocol
    ------------------
    Client → Server:
        Binary message : raw signed 16-bit PCM audio, 16 kHz mono.
        Text message   : JSON control {"type": "set_avatar", "data": "<base64 PNG/JPG>"}

    Server → Client:
        Binary message : JPEG frame bytes.
        Text message   : JSON status/error {"type": "info"|"error", "message": "..."}
                         + latency ping:  {"type":"latency","ms":213,"frame":42}
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    if engine is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Model not loaded. Run setup_models.py first.",
        }))
        await websocket.close()
        return

    stop_event = asyncio.Event()

    async def frame_sender(jpeg: bytes, latency: float) -> None:
        global _latest_frame
        _latest_frame = jpeg
        _latest_frame_event.set()
        try:
            await websocket.send_bytes(jpeg)
            if engine.total_frames % 5 == 0:
                await websocket.send_text(json.dumps({
                    "type": "latency",
                    "ms": round(latency, 1),
                    "frame": engine.total_frames,
                }))
        except Exception:
            pass

    # Start the inference loop — bufferless: audio is pushed directly into the
    # rolling mel buffer so every frame uses the LATEST speech, never a stale queue.
    stream_task = asyncio.create_task(
        engine.stream_frames(stop_event, frame_sender)
    )

    try:
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes"):
                # Push PCM audio directly into the engine's rolling buffer.
                # No queue — inference always runs on the freshest audio.
                if engine.has_avatar():
                    engine.push_audio(message["bytes"])
                else:
                    await websocket.send_text(json.dumps({
                        "type": "info",
                        "message": "Upload an avatar photo first (/api/avatar).",
                    }))

            elif message.get("text"):
                await _handle_text_message(message["text"], websocket)

    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
    finally:
        logger.info("WebSocket client disconnected")
        stop_event.set()
        stream_task.cancel()


async def _handle_text_message(text: str, ws: WebSocket) -> None:
    """Handle JSON control messages from the client."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return

    if msg.get("type") == "set_avatar":
        b64 = msg.get("data", "")
        # Strip data-URL prefix if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            img_data = base64.b64decode(b64)
            nparr = np.frombuffer(img_data, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Cannot decode image")
            await asyncio.get_running_loop().run_in_executor(None, engine.set_avatar, image_bgr)
            await ws.send_text(json.dumps({"type": "info", "message": "Avatar set via WebSocket."}))
        except Exception as exc:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))

    elif msg.get("type") == "ping":
        await ws.send_text(json.dumps({"type": "pong", "ts": msg.get("ts", 0)}))


# ── MJPEG stream (for OBS / browser img tag) ──────────────────────────────────

@app.get("/stream/mjpeg")
async def mjpeg_stream():
    """
    MJPEG stream endpoint.  Open in OBS → Media Source → URL.
    Also works in <img src="/stream/mjpeg"> in most browsers.
    """
    async def generate() -> AsyncGenerator[bytes, None]:
        boundary = b"--frame\r\n"
        while True:
            # Wait for a new frame (max 100 ms timeout = 10 fps minimum)
            try:
                await asyncio.wait_for(_latest_frame_event.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            _latest_frame_event.clear()

            frame = _latest_frame
            if frame is None:
                await asyncio.sleep(0.04)
                continue

            header = (
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            )
            yield boundary + header + frame + b"\r\n"

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
