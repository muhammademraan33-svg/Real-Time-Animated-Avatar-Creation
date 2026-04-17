"""
Face detection, cropping, and compositing utilities for real-time Wav2Lip.

Pipeline:
  1. Detect face – tries MediaPipe (direct module import, works in 0.10.x),
     then OpenCV Haar cascade, then a sensible centre-crop fallback.
  2. Crop face ROI with a small margin → resize to 96×96 for Wav2Lip.
  3. Build the 6-channel input tensor: [masked (lower half 0) | original].
  4. After inference, composite only the lower 55 % (mouth/chin) of the
     generated face back using Gaussian-blurred alpha blending for seamless
     results with no hard seam.
"""
from __future__ import annotations

import cv2
import logging
import numpy as np
from typing import NamedTuple

logger = logging.getLogger(__name__)

FACE_SIZE: int = 96      # Wav2Lip input/output face size
MARGIN_RATIO: float = 0.15  # padding around detected face bbox

# ── Try to import MediaPipe solutions directly ────────────────────────────────
_mp_detector = None
try:
    from mediapipe.python.solutions import face_detection as _mp_fd
    _mp_detector = _mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    logger.info("Face detector: MediaPipe (python.solutions)")
except Exception:
    logger.info("Face detector: falling back to OpenCV Haar cascade")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class FaceCrop(NamedTuple):
    """Bounding box of the detected face in the original image."""
    x1: int
    y1: int
    x2: int
    y2: int


class FacePreprocessed(NamedTuple):
    """All cached data for a single avatar photo."""
    face_input: np.ndarray      # (96, 96, 6) float32 [0,1]: masked | original
    crop: FaceCrop
    original_frame: np.ndarray  # full-size BGR for compositing


# ─────────────────────────────────────────────────────────────────────────────
# FaceProcessor
# ─────────────────────────────────────────────────────────────────────────────

class FaceProcessor:
    """Detects and preprocesses the face from a single avatar photo."""

    def __init__(self):
        # Haar cascade is the fallback when MediaPipe is unavailable
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self, image_bgr: np.ndarray) -> FacePreprocessed:
        """Detect face, build model input, cache everything for real-time use."""
        x1, y1, x2, y2 = self._detect_face(image_bgr)

        face_roi = image_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (FACE_SIZE, FACE_SIZE),
                                  interpolation=cv2.INTER_AREA)

        face_input = self._make_face_input(face_resized)
        crop = FaceCrop(x1=x1, y1=y1, x2=x2, y2=y2)

        logger.info("Face crop: x1=%d y1=%d x2=%d y2=%d (face %dx%d)",
                    x1, y1, x2, y2, x2-x1, y2-y1)

        return FacePreprocessed(
            face_input=face_input,
            crop=crop,
            original_frame=image_bgr.copy(),
        )

    def composite(
        self,
        base_frame: np.ndarray,
        generated_face_01: np.ndarray,
        crop: FaceCrop,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Blend only the mouth-centric region from Wav2Lip back into base_frame.

        This keeps the original eyes/forehead sharp while making lip motion
        clearly visible.  The mask focuses on the lower-centre face region
        where viseme changes happen.
        """
        base_frame, crop = self._apply_head_motion(base_frame, crop, frame_index)
        x1, y1, x2, y2 = crop.x1, crop.y1, crop.x2, crop.y2
        face_h = y2 - y1
        face_w = x2 - x1
        if face_h <= 0 or face_w <= 0:
            return base_frame.copy()

        # Model output: (96,96,3) float32 RGB [0,1] -> BGR uint8
        gen_rgb = (generated_face_01 * 255.0).clip(0, 255).astype(np.uint8)
        gen_bgr = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)
        # INTER_LINEAR is much faster than LANCZOS4 and visually sufficient here.
        gen_up = cv2.resize(gen_bgr, (face_w, face_h), interpolation=cv2.INTER_LINEAR)

        # Build a mouth-focused soft mask:
        # - ellipse centered around mouth
        # - extra support for lower lip/chin strip
        yy, xx = np.mgrid[0:face_h, 0:face_w].astype(np.float32)
        cx = face_w * 0.50
        cy = face_h * 0.68
        rx = max(1.0, face_w * 0.33)
        ry = max(1.0, face_h * 0.24)
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        alpha_ellipse = np.clip(1.0 - ellipse, 0.0, 1.0)

        alpha = alpha_ellipse
        strip_y = int(face_h * 0.62)
        if strip_y < face_h:
            alpha[strip_y:, :] = np.maximum(alpha[strip_y:, :], 0.55)

        # Soften edges, but keep center strong for visible lip motion.
        sigma = max(2.0, face_h * 0.04)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha = np.clip(alpha * 1.25, 0.0, 1.0)[:, :, None]

        out = base_frame.copy()
        roi = out[y1:y2, x1:x2].astype(np.float32)
        gen_f = gen_up.astype(np.float32)
        out[y1:y2, x1:x2] = (roi * (1.0 - alpha) + gen_f * alpha).clip(0, 255).astype(np.uint8)
        self._apply_eye_blink(out, crop, frame_index)
        return out

    @staticmethod
    def _apply_head_motion(base_frame: np.ndarray, crop: FaceCrop, frame_index: int) -> tuple[np.ndarray, FaceCrop]:
        """Apply a tiny global sway so the portrait feels less static."""
        h, w = base_frame.shape[:2]
        dx = int(round(2.0 * np.sin(frame_index * 0.11)))
        dy = int(round(1.0 * np.sin(frame_index * 0.07)))
        if dx == 0 and dy == 0:
            return base_frame, crop

        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(
            base_frame,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        moved_crop = FaceCrop(
            x1=max(0, min(w, crop.x1 + dx)),
            y1=max(0, min(h, crop.y1 + dy)),
            x2=max(0, min(w, crop.x2 + dx)),
            y2=max(0, min(h, crop.y2 + dy)),
        )
        return moved, moved_crop

    @staticmethod
    def _apply_eye_blink(frame_bgr: np.ndarray, crop: FaceCrop, frame_index: int) -> None:
        """Overlay a lightweight deterministic blink over the eye region."""
        blink_cycle = frame_index % 120
        blink_strength_map = {0: 0.35, 1: 0.8, 2: 1.0, 3: 0.55}
        strength = blink_strength_map.get(blink_cycle, 0.0)
        if strength <= 0.0:
            return

        x1, y1, x2, y2 = crop.x1, crop.y1, crop.x2, crop.y2
        face_h = y2 - y1
        face_w = x2 - x1
        if face_h <= 0 or face_w <= 0:
            return

        eye_boxes = [
            (
                x1 + int(face_w * 0.16),
                y1 + int(face_h * 0.26),
                int(face_w * 0.26),
                int(face_h * 0.10),
            ),
            (
                x1 + int(face_w * 0.58),
                y1 + int(face_h * 0.26),
                int(face_w * 0.26),
                int(face_h * 0.10),
            ),
        ]

        for ex, ey, ew, eh in eye_boxes:
            if ew <= 0 or eh <= 0:
                continue
            ex2 = min(frame_bgr.shape[1], ex + ew)
            ey2 = min(frame_bgr.shape[0], ey + eh)
            ex = max(0, ex)
            ey = max(0, ey)
            if ex2 <= ex or ey2 <= ey:
                continue

            roi = frame_bgr[ey:ey2, ex:ex2]
            if roi.size == 0:
                continue

            lid_h = max(1, int((ey2 - ey) * 0.5 * strength))
            eyelid_color = roi.mean(axis=(0, 1), dtype=np.float32)

            overlay = roi.astype(np.float32)
            overlay[:lid_h, :, :] = eyelid_color
            overlay[-lid_h:, :, :] = eyelid_color
            frame_bgr[ey:ey2, ex:ex2] = overlay.clip(0, 255).astype(np.uint8)

    # ── Face detection ────────────────────────────────────────────────────────

    def _detect_face(self, image_bgr: np.ndarray) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) with margin. Never raises – uses fallback."""
        h, w = image_bgr.shape[:2]

        bbox = None

        # 1. MediaPipe (most accurate)
        if _mp_detector is not None:
            bbox = _detect_mediapipe(image_bgr, _mp_detector)

        # 2. Haar cascade
        if bbox is None:
            bbox = _detect_haar(image_bgr, self._haar)

        # 3. Fallback: centre-upper portrait crop
        if bbox is None:
            logger.warning("No face detected – using centre-upper fallback.")
            fw = int(w * 0.50)
            fh = int(h * 0.48)
            fx = (w - fw) // 2
            fy = int(h * 0.02)
            bbox = (fx, fy, fw, fh)

        fx, fy, fw, fh = bbox
        mx = int(fw * MARGIN_RATIO)
        my = int(fh * MARGIN_RATIO)
        x1 = max(fx - mx, 0)
        y1 = max(fy - my, 0)
        x2 = min(fx + fw + mx, w)
        y2 = min(fy + fh + my, h)
        return x1, y1, x2, y2

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_face_input(face_rgb_96: np.ndarray) -> np.ndarray:
        """
        Build (96, 96, 6) float32 tensor:
            ch 0-2: masked face (lower half zeroed = hide mouth)
            ch 3-5: original face
        """
        face_f = face_rgb_96.astype(np.float32) / 255.0
        masked = face_f.copy()
        masked[FACE_SIZE // 2:, :, :] = 0.0   # zero the lower half
        return np.concatenate([masked, face_f], axis=2)  # (96, 96, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_mediapipe(image_bgr: np.ndarray, detector) -> tuple | None:
    """Return (x, y, w, h) from MediaPipe or None."""
    try:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if not results.detections:
            return None
        best = max(results.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        fx = int(bb.xmin * w)
        fy = int(bb.ymin * h)
        fw = int(bb.width * w)
        fh = int(bb.height * h)
        return (fx, fy, fw, fh)
    except Exception as exc:
        logger.debug("MediaPipe detection error: %s", exc)
        return None


def _detect_haar(image_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple | None:
    """Return (x, y, w, h) from Haar cascade or None."""
    try:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(30, w // 12), max(30, h // 12)),
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return None
        return tuple(map(int, max(faces, key=lambda f: f[2] * f[3])))
    except Exception as exc:
        logger.debug("Haar detection error: %s", exc)
        return None


def frame_to_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def resize_keep_aspect(
    image: np.ndarray,
    max_width: int = 512,
    max_height: int = 512,
) -> np.ndarray:
    """Resize so neither dimension exceeds the max, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image
