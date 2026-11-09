"""Classical baseline: temporal background subtraction + smoke colour/texture gate.

No training. Keeps a slow running background, removes global exposure changes,
and keeps foreground blobs that look like smoke: low colour saturation (grey/white),
soft edges (low texture). Upward drift and growth are judged later by the
temporal confirmer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .temporal import Detection


@dataclass
class ClassicalConfig:
    alpha: float = 0.02  # background learning rate (slow, so a plume is not absorbed quickly)
    warmup: int = 5  # frames used to initialise the background
    diff_thr: float = 14.0  # min brightness change (0-255) to count as foreground
    max_chroma: float = 45.0  # smoke is grey/white: max(rgb) - min(rgb) below this
    max_edge: float = 28.0  # mean gradient magnitude in the blob; smoke edges are soft
    min_area_px: int = 10
    max_blob_frac: float = 0.5  # a blob this large is a global change (fog, exposure), not a plume
    score_diff: float = 40.0  # mean change that maps to score 1.0


class ClassicalSmokeDetector:
    def __init__(self, cfg: ClassicalConfig | None = None):
        self.cfg = cfg or ClassicalConfig()
        self.reset()

    def reset(self) -> None:
        self.bg: np.ndarray | None = None
        self.n = 0

    def update(self, frame: np.ndarray) -> list[Detection]:
        """Process the next frame of a fixed camera; return candidate blobs."""
        cfg = self.cfg
        f = frame.astype(np.float32)
        gray = f.mean(axis=2)
        if self.bg is None:
            self.bg = f.copy()
            self.n = 1
            return []
        bg_gray = self.bg.mean(axis=2)
        # remove global exposure drift: scale the frame so its median matches the background
        valid = bg_gray > 20
        gain = float(np.median(bg_gray[valid] / np.maximum(gray[valid], 1.0))) if valid.any() else 1.0
        gray_n = gray * gain
        diff = gray_n - bg_gray
        warm = self.n < cfg.warmup
        dets: list[Detection] = []
        if not warm:
            fg = (np.abs(diff) > cfg.diff_thr).astype(np.uint8)
            chroma = f.max(axis=2) - f.min(axis=2)
            fg &= (chroma < cfg.max_chroma).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drops thin edge-jitter lines
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            gx = cv2.Sobel(gray_n, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_n, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.hypot(gx, gy) / 8.0
            n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            total = fg.shape[0] * fg.shape[1]
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if min(w, h) < 3 or area < cfg.min_area_px or area > cfg.max_blob_frac * total:
                    continue
                m = labels == i
                if float(grad[m].mean()) > cfg.max_edge:
                    continue
                strength = float(np.abs(diff[m]).mean()) / cfg.score_diff
                size = min(1.0, area / 60.0)
                score = float(np.clip(0.35 + 0.45 * strength + 0.2 * size, 0.0, 1.0))
                dets.append(Detection((float(x), float(y), float(x + w), float(y + h)), score))
        # update the background only where nothing is detected, so a lingering plume stays foreground
        rate = np.full(gray.shape, cfg.alpha, np.float32) if warm else np.where(
            np.abs(diff) > cfg.diff_thr, cfg.alpha * 0.1, cfg.alpha
        ).astype(np.float32)
        self.bg += (f * gain - self.bg) * rate[..., None]
        self.n += 1
        return dets
