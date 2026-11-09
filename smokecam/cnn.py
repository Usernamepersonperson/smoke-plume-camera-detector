"""ONNX heat-map smoke detector (CPU, onnxruntime). Training lives in scripts/train.py."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .common import GH, GW, MODEL_PATH, STRIDE
from .temporal import Detection


def heat_to_detections(heat: np.ndarray, thr: float) -> list[Detection]:
    """Connected cells above thr -> one box per blob (pixel units), score = max cell prob."""
    if heat.shape != (GH, GW):
        raise ValueError(f"heat map must be {(GH, GW)}, got {heat.shape}")
    mask = (heat >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        score = float(heat[labels == i].max())
        out.append(Detection((x * STRIDE, y * STRIDE, (x + w) * STRIDE, (y + h) * STRIDE), score))
    return out


class OnnxSmokeDetector:
    def __init__(self, path: Path | str = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.path = Path(path)

    @staticmethod
    def preprocess(frames: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 RGB -> (N, 3, H, W) float32 in [0, 1]."""
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must be (N, H, W, 3) or (H, W, 3)")
        return np.ascontiguousarray(frames.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0

    def heatmaps(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        """Per-cell smoke probability, shape (N, GH, GW)."""
        x = self.preprocess(frames)
        outs = []
        for i in range(0, len(x), batch):
            logits = self.sess.run(None, {self.input_name: x[i : i + batch]})[0]
            outs.append(1.0 / (1.0 + np.exp(-logits[:, 0])))
        return np.concatenate(outs) if outs else np.zeros((0, GH, GW), np.float32)

    def detect(self, frame: np.ndarray, thr: float = 0.5) -> list[Detection]:
        return heat_to_detections(self.heatmaps(frame)[0], thr)
