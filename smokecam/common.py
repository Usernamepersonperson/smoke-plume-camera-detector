"""Shared constants and small helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H = 160, 96  # network / pipeline resolution (width, height)
STRIDE = 8  # heat-map cell size in pixels -> 20 x 12 grid
GW, GH = W // STRIDE, H // STRIDE
TESTSET_DIR = ROOT / "data" / "testset"
MODEL_PATH = ROOT / "models" / "smoke_cnn.onnx"
FRAME_INTERVAL_S = 60  # assumed capture interval for time-based numbers


def camera_in_test(camera: str) -> bool:
    """Deterministic camera-level split: about a quarter of cameras are held out."""
    return int(hashlib.md5(camera.encode()).hexdigest(), 16) % 4 == 0


def load_image(path: Path) -> np.ndarray:
    """Return an RGB uint8 (H, W, 3) array of the stored pipeline-size frame."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path: Path, rgb: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def parse_boxes(annotations: str) -> list[tuple[float, float, float, float]]:
    """YOLO lines 'cls cx cy w h' -> list of (x0, y0, x1, y1) in normalised units."""
    out = []
    for line in annotations.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        out.append((max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)))
    return out


def boxes_to_grid(boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Cell (gy, gx) is positive when at least 25 % of the cell is inside a box."""
    grid = np.zeros((GH, GW), np.float32)
    for x0, y0, x1, y1 in boxes:
        for gy in range(GH):
            for gx in range(GW):
                cx0, cx1 = gx / GW, (gx + 1) / GW
                cy0, cy1 = gy / GH, (gy + 1) / GH
                ix = max(0.0, min(x1, cx1) - max(x0, cx0))
                iy = max(0.0, min(y1, cy1) - max(y0, cy0))
                if ix * iy >= 0.25 * (1 / GW) * (1 / GH):
                    grid[gy, gx] = 1.0
    return grid
