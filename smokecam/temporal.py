"""Temporal confirmation: a plume must persist, stay put and grow before it alarms.

Rejects the usual look-alikes: a cloud drifts sideways, fog covers most of the frame,
glare flickers. Works on per-frame detections from any detector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .firedanger import AlarmPolicy


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    hits: deque
    frames: deque = field(default_factory=deque)  # (frame_idx, Detection) of hits
    misses: int = 0


@dataclass
class ConfirmConfig:
    max_area_frac: float = 0.30  # fog / large cloud banks cover more than this
    max_drift_px_per_frame: float = 1.2  # clouds drift sideways, a plume base is fixed
    require_growth: bool = True  # area grows or the top rises
    match_dist_px: float = 24.0
    frame_area: float = 160 * 96


class TemporalConfirmer:
    def __init__(self, policy: AlarmPolicy, cfg: ConfirmConfig | None = None):
        self.policy = policy
        self.cfg = cfg or ConfirmConfig()
        self.tracks: list[Track] = []
        self._next = 0
        self.idx = -1

    def reset(self) -> None:
        self.tracks, self._next, self.idx = [], 0, -1

    def _match(self, dets: list[Detection]) -> dict[int, Detection]:
        """Greedy nearest-centre association; returns track id -> detection."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            tc = t.frames[-1][1].center
            for di, d in enumerate(dets):
                dist = float(np.hypot(tc[0] - d.center[0], tc[1] - d.center[1]))
                if dist <= self.cfg.match_dist_px or iou(t.frames[-1][1].box, d.box) > 0.1:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d, out = set(), set(), {}
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out[self.tracks[ti].tid] = dets[di]
        return out

    def _plausible(self, t: Track) -> bool:
        frames = list(t.frames)[-self.policy.n :]
        if len(frames) < 2:
            return True
        last = frames[-1][1]
        if last.area / self.cfg.frame_area > self.cfg.max_area_frac:
            return False
        idx = np.array([f[0] for f in frames], float)
        cx = np.array([f[1].center[0] for f in frames])
        if idx[-1] > idx[0] and abs(np.polyfit(idx, cx, 1)[0]) > self.cfg.max_drift_px_per_frame:
            return False
        if self.cfg.require_growth:
            first = frames[0][1]
            grows = last.area >= 0.9 * first.area
            rises = last.box[1] <= first.box[1] - 0.5
            if not (grows or rises):
                return False
        return True

    def update(self, dets: list[Detection]) -> bool:
        """Feed one frame's detections; True while a confirmed alarm is active."""
        self.idx += 1
        dets = [d for d in dets if d.score >= self.policy.score_thr]
        matched = self._match(dets)
        taken = {id(d) for d in matched.values()}
        for t in self.tracks:
            d = matched.get(t.tid)
            t.hits.append(d is not None)
            if d is not None:
                t.frames.append((self.idx, d))
                t.misses = 0
            else:
                t.misses += 1
        for d in dets:
            if id(d) not in taken:
                t = Track(self._next, deque(maxlen=self.policy.n))
                self._next += 1
                t.hits.append(True)
                t.frames.append((self.idx, d))
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.misses < self.policy.n]
        for t in self.tracks:
            t.frames = deque(list(t.frames)[-self.policy.n * 2 :])
        return any(sum(t.hits) >= self.policy.k and self._plausible(t) for t in self.tracks)
