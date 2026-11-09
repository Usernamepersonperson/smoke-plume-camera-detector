"""Synthetic sequences: procedural smoke plumes and look-alike weather over real frames.

Everything here is SIMULATED. Backgrounds are real smoke-free frames from held-out
cameras; plumes, clouds, fog, glare, dust and exposure changes are procedural.
Real wildfire smoke looks different, so results on these sequences measure the
temporal logic and the detectors' behaviour, not field performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import H, W


def fractal_noise(rng: np.random.Generator, h: int, w: int, octaves: int = 4) -> np.ndarray:
    """Smooth value noise in [0, 1] (sum of upsampled random grids)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = 3 * 2**o, 4 * 2**o
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.55
    out /= total
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


@dataclass
class PlumeParams:
    origin: tuple[float, float]  # base of the plume (x, y) in pixels
    lean_deg: float  # wind lean from vertical, + = to the right
    rise_speed: float  # px per frame of plume-head growth
    strength: float  # peak opacity at maturity, 0.25 - 0.8
    width0: float = 1.2
    spread: float = 0.16  # width growth per px of length
    seed: int = 0
    noise: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.noise = fractal_noise(rng, H * 2, W * 2, octaves=5)


def plume_alpha(p: PlumeParams, t: float) -> np.ndarray:
    """Opacity map (H, W) in [0, 1] of the plume t frames after ignition (t < 0 -> empty)."""
    alpha = np.zeros((H, W), np.float32)
    if t <= 0:
        return alpha
    length = p.rise_speed * t**0.92
    ang = np.radians(p.lean_deg)
    d = np.array([np.sin(ang), -np.cos(ang)], np.float32)  # image y points down
    nrm = np.array([np.cos(ang), np.sin(ang)], np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    rx, ry = xs - p.origin[0], ys - p.origin[1]
    u = rx * d[0] + ry * d[1]  # along the plume
    c = rx * nrm[0] + ry * nrm[1]  # across the plume
    inside = (u >= 0) & (u <= length)
    width = p.width0 + p.spread * np.clip(u, 0, None)
    profile = np.exp(-((c / width) ** 2))
    ramp = min(1.0, t / 18.0)  # the fire builds up over ~18 frames
    fade = np.exp(-np.clip(u, 0, None) / max(0.8 * length, 1.0)) * 0.6 + 0.4  # thins towards the head
    # turbulence: noise advected along the plume
    sx = int(p.noise.shape[1] // 4 + (t * 0.7 * d[0])) % (p.noise.shape[1] - W)
    sy = int(p.noise.shape[0] // 4 + (t * 0.7 * d[1])) % (p.noise.shape[0] - H)
    tex = 0.55 + 0.75 * p.noise[sy : sy + H, sx : sx + W]
    alpha = np.where(inside, profile * fade * tex * p.strength * ramp, 0.0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.8), 0, 1)


def plume_box(alpha: np.ndarray, thr: float = 0.10) -> tuple[int, int, int, int] | None:
    """Tight box (x0, y0, x1, y1) around visible smoke, or None if fewer than 8 px are visible."""
    m = alpha > thr
    if m.sum() < 8:
        return None
    ys, xs = np.where(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def blend(frame: np.ndarray, alpha: np.ndarray, colour: tuple[float, float, float] = (205, 207, 210)) -> np.ndarray:
    """Alpha-composite a flat smoke colour, brightened with the scene's own brightness."""
    f = frame.astype(np.float32)
    scene = float(f.mean())
    col = np.array(colour, np.float32) * (0.75 + 0.35 * min(1.0, scene / 140.0))
    return f * (1 - alpha[..., None]) + col * alpha[..., None]


def cloud_alpha(rng: np.random.Generator, t: float, x0: float, speed: float, size: float, y: float, life: float,
                noise: np.ndarray) -> np.ndarray:
    """A drifting cloud in the upper sky: soft ellipse x texture, grows and dissolves over `life` frames."""
    if not 0 <= t <= life:
        return np.zeros((H, W), np.float32)
    env = np.sin(np.pi * t / life) ** 0.7
    cx = x0 + speed * t
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    e = ((xs - cx) / (size * 1.8)) ** 2 + ((ys - y) / (size * 0.65)) ** 2
    tex = 0.5 + 0.8 * noise[:H, :W]
    return np.clip(np.exp(-e) * tex * 0.85 * env, 0, 1).astype(np.float32)


def fog_alpha(t: float, start: float, ramp: float, hold: float, depth: float) -> np.ndarray:
    """Fog veil: ramps in, holds, lifts. Denser near the ground. Covers most of the frame."""
    if t < start or t > start + ramp * 2 + hold:
        return np.zeros((H, W), np.float32)
    if t < start + ramp:
        k = (t - start) / ramp
    elif t < start + ramp + hold:
        k = 1.0
    else:
        k = 1 - (t - start - ramp - hold) / ramp
    grad = np.linspace(0.45, 1.0, H, dtype=np.float32)[:, None]
    return np.clip(np.ones((H, W), np.float32) * grad * depth * k, 0, 1)


def glare_alpha(t: float, start: float, dur: float, x: float, y: float, radius: float, peak: float) -> np.ndarray:
    """Sun glare: a bright radial blob that flares and fades."""
    if not start <= t <= start + dur:
        return np.zeros((H, W), np.float32)
    env = np.sin(np.pi * (t - start) / dur)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xs - x) ** 2 + (ys - y) ** 2) / radius**2
    return np.clip(np.exp(-r2) * peak * env, 0, 1).astype(np.float32)


def dust_alpha(rng: np.random.Generator, t: float, start: float, dur: float, speed: float, y: float,
               noise: np.ndarray) -> np.ndarray:
    """Dust cloud kicked up near the ground: low, wide, moves fast sideways."""
    if not start <= t <= start + dur:
        return np.zeros((H, W), np.float32)
    env = np.sin(np.pi * (t - start) / dur)
    cx = -20 + speed * (t - start)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    e = ((xs - cx) / 26.0) ** 2 + ((ys - y) / 8.0) ** 2
    return np.clip(np.exp(-e) * (0.4 + 0.7 * noise[:H, :W]) * 0.5 * env, 0, 1).astype(np.float32)


def daylight_gain(frame_idx: int, n_frames: int) -> float:
    """Smooth day curve 0.7 .. 1.1 over the simulated daylight window."""
    return 0.7 + 0.4 * float(np.sin(np.pi * frame_idx / max(n_frames - 1, 1)))


@dataclass
class Sequence:
    frames: np.ndarray  # (T, H, W, 3) uint8
    kind: str
    ignition: int | None = None  # frame index of ignition
    first_visible: int | None = None
    boxes: list = field(default_factory=list)  # per-frame GT box or None


def _noisy(img: np.ndarray, rng: np.random.Generator, gain: float, jitter: tuple[float, float]) -> np.ndarray:
    M = np.float32([[1, 0, jitter[0]], [0, 1, jitter[1]]])
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = img * gain + rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


def weather_layers(rng: np.random.Generator, n: int, scenario: str):
    """Build the list of look-alike events for a scenario: clouds, fog, glare_dust, mixed."""
    noise = fractal_noise(rng, H, W * 2, 4)
    events = []
    if scenario in ("clouds", "mixed"):
        for _ in range(rng.integers(3, 6)):
            start = float(rng.uniform(0, max(n - 60, 1)))
            life = float(rng.uniform(60, 240))
            speed = float(rng.choice([-1, 1]) * rng.uniform(0.1, 2.2))
            x0 = (0 if speed > 0 else W) + float(rng.uniform(-30, 30))
            size = float(rng.uniform(14, 34))
            y = float(rng.uniform(10, 40))
            off = int(rng.integers(0, W))
            events.append(("cloud", start, dict(x0=x0, speed=speed, size=size, y=y, life=life,
                                                noise=noise[:, off : off + W])))
    if scenario in ("fog", "mixed"):
        for _ in range(rng.integers(1, 3)):
            events.append(("fog", 0.0, dict(start=float(rng.uniform(0, max(n - 150, 1))), ramp=float(rng.uniform(20, 60)),
                                            hold=float(rng.uniform(30, 120)), depth=float(rng.uniform(0.35, 0.8)))))
    if scenario in ("glare_dust", "mixed"):
        for _ in range(rng.integers(2, 4)):
            events.append(("glare", 0.0, dict(start=float(rng.uniform(0, max(n - 40, 1))), dur=float(rng.uniform(10, 40)),
                                              x=float(rng.uniform(20, W - 20)), y=float(rng.uniform(5, 45)),
                                              radius=float(rng.uniform(10, 26)), peak=float(rng.uniform(0.5, 0.95)))))
        for _ in range(rng.integers(2, 4)):
            events.append(("dust", 0.0, dict(start=float(rng.uniform(0, max(n - 60, 1))), dur=float(rng.uniform(15, 45)),
                                             speed=float(rng.uniform(2.0, 5.0)), y=float(rng.uniform(55, 85)),
                                             noise=noise)))
        if rng.random() < 0.5:  # auto-exposure step
            events.append(("exposure", float(rng.uniform(0.2, 0.8) * n), dict(step=float(rng.uniform(0.82, 1.18)))))
    return events


def render_sequence(
    background: np.ndarray, n: int, scenario: str, seed: int, plume: PlumeParams | None = None,
    ignition: int = 20, distractors: bool = True,
) -> Sequence:
    """Render n frames (1 frame/min assumed) of a fixed camera over a real background frame.

    scenario: 'clear' | 'clouds' | 'fog' | 'glare_dust' | 'mixed' (look-alike weather).
    plume: if given, smoke ignites at frame `ignition`.
    """
    rng = np.random.default_rng(seed)
    bg = background.astype(np.float32)
    events = weather_layers(rng, n, scenario) if distractors and scenario != "clear" else []
    cloud_rng = np.random.default_rng(seed + 1)
    frames = np.empty((n, H, W, 3), np.uint8)
    boxes: list = []
    first_visible = None
    exp_gain = 1.0
    for i in range(n):
        img = bg.copy()
        for kind, start, kw in events:
            if kind == "cloud":
                a = cloud_alpha(cloud_rng, i - start, **kw)
                if a.any():
                    img = blend(img, a, (225, 228, 232))
            elif kind == "fog":
                a = fog_alpha(i, **kw)
                if a.any():
                    img = blend(img, a, (190, 195, 200))
            elif kind == "glare":
                a = glare_alpha(i, **kw)
                if a.any():
                    img = blend(img, a, (255, 252, 240))
            elif kind == "dust":
                a = dust_alpha(cloud_rng, i, **kw)
                if a.any():
                    img = blend(img, a, (190, 165, 130))
            elif kind == "exposure" and i >= start:
                exp_gain = kw["step"]
        box = None
        if plume is not None:
            a = plume_alpha(plume, i - ignition)
            if a.any():
                img = blend(img, a)
            box = plume_box(a)
            if box is not None and first_visible is None:
                first_visible = i
        boxes.append(box)
        jitter = (float(rng.normal(0, 0.25)), float(rng.normal(0, 0.25)))
        frames[i] = _noisy(img, rng, daylight_gain(i, n) * exp_gain, jitter).astype(np.uint8)
    return Sequence(frames, scenario, ignition if plume is not None else None, first_visible, boxes)


def random_plume(rng: np.random.Generator, seed: int) -> PlumeParams:
    return PlumeParams(
        origin=(float(rng.uniform(25, W - 25)), float(rng.uniform(H * 0.62, H * 0.9))),
        lean_deg=float(rng.uniform(-35, 35)),
        rise_speed=float(rng.uniform(0.6, 1.4)),
        strength=float(rng.uniform(0.3, 0.8)),
        spread=float(rng.uniform(0.10, 0.22)),
        seed=seed,
    )
