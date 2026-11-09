"""Measurements: real held-out frames, real short sequences, simulated sequences, latency."""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from .classical import ClassicalSmokeDetector
from .cnn import OnnxSmokeDetector, heat_to_detections
from .common import FRAME_INTERVAL_S, GH, GW, MODEL_PATH, ROOT, STRIDE, TESTSET_DIR, H, W, load_image
from .firedanger import policy_for_level
from .synth import random_plume, render_sequence
from .temporal import TemporalConfirmer

CAMERA_DAY_FRAMES = 840  # 14 h of daylight at one frame per minute
EPISODE_GAP = 30  # alarm runs closer than this many frames are one false-alarm episode


def load_testset() -> tuple[np.ndarray, list[dict]]:
    meta = json.loads((TESTSET_DIR / "meta.json").read_text())
    frames = np.stack([load_image(TESTSET_DIR / m["name"]) for m in meta])
    return frames, meta


def box_px(b) -> tuple[float, float, float, float]:
    return b[0] * W, b[1] * H, b[2] * W, b[3] * H


# ---------------------------------------------------------------- real frames


def frame_metrics(heat: np.ndarray, meta: list[dict], thr: float) -> dict:
    """Per-frame precision/recall/F1 and localisation on REAL held-out frames."""
    tp = fp = fn = tn = 0
    loc_hits = loc_total = 0
    for h, m in zip(heat, meta, strict=True):
        dets = heat_to_detections(h, thr)
        positive = bool(m["boxes"])
        flagged = bool(dets)
        if positive and flagged:
            tp += 1
            loc_total += 1
            gts = [box_px(b) for b in m["boxes"]]
            ok = any(overlaps(d.box, g, 0.0) for d in dets for g in gts)
            loc_hits += ok
        elif positive:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "n": len(meta), "positives": tp + fn, "negatives": fp + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
        "false_positive_rate": fp / max(fp + tn, 1),
        "localisation_hit_rate": loc_hits / max(loc_total, 1),
    }


def roc_summary(heat: np.ndarray, meta: list[dict]) -> dict:
    """Threshold-free view of the per-frame score (max cell probability) on the real held-out frames."""
    score = heat.reshape(len(heat), -1).max(axis=1)
    pos = np.array([bool(m["boxes"]) for m in meta])
    sp, sn = score[pos], score[~pos]
    auc = float(((sp[:, None] > sn[None, :]).sum() + 0.5 * (sp[:, None] == sn[None, :]).sum()) / (len(sp) * len(sn)))
    out = {"auc": auc}
    for fpr in (0.1, 0.2):
        thr = float(np.quantile(sn, 1 - fpr))
        out[f"recall_at_fpr_{int(fpr * 100)}"] = float((sp > thr).mean())
        out[f"thr_at_fpr_{int(fpr * 100)}"] = thr
    return out


def center_x_error(heat: np.ndarray, meta: list[dict], thr: float) -> list[float]:
    """Signed error (pixels, 160 px wide frame) of the best detection's centre column vs the GT box centre."""
    errs = []
    for h, m in zip(heat, meta, strict=True):
        if not m["boxes"]:
            continue
        dets = heat_to_detections(h, thr)
        if not dets:
            continue
        best = max(dets, key=lambda d: d.score)
        gts = [box_px(b) for b in m["boxes"]]
        g = min(gts, key=lambda g: abs((g[0] + g[2]) / 2 - best.center[0]))
        if overlaps(best.box, g, 0.0):
            errs.append(best.center[0] - (g[0] + g[2]) / 2)
    return errs


def real_sequences(meta: list[dict], max_gap_min: float = 15.0) -> list[list[int]]:
    """Index lists of consecutive frames from the same camera, < max_gap_min apart (>= 3 frames)."""
    def ts(m):
        return datetime.strptime(m["date"], "%Y-%m-%dT%H-%M-%S")

    order = sorted(range(len(meta)), key=lambda i: (meta[i]["camera"], meta[i]["date"]))
    seqs, cur = [], []
    for i in order:
        if cur and (meta[i]["camera"] != meta[cur[-1]]["camera"]
                    or (ts(meta[i]) - ts(meta[cur[-1]])).total_seconds() > max_gap_min * 60):
            seqs.append(cur)
            cur = []
        cur.append(i)
    seqs.append(cur)
    return [s for s in seqs if len(s) >= 3]


def eval_real_sequences(heat: np.ndarray, meta: list[dict], level: int) -> dict:
    """Run the confirmer over REAL consecutive frames (irregular spacing, not 1/min)."""
    pol = policy_for_level(level, "cnn")
    out = {"smoke": [0, 0], "no_smoke": [0, 0], "frames_to_confirm": []}
    for s in real_sequences(meta):
        pos = sum(bool(meta[i]["boxes"]) for i in s) >= len(s) / 2
        conf = TemporalConfirmer(pol)
        alarm_at = None
        for k, i in enumerate(s):
            if conf.update(heat_to_detections(heat[i], pol.score_thr)) and alarm_at is None:
                alarm_at = k
        key = "smoke" if pos else "no_smoke"
        out[key][1] += 1
        out[key][0] += alarm_at is not None
        if pos and alarm_at is not None:
            out["frames_to_confirm"].append(alarm_at + 1)
    return {
        "smoke_sequences": out["smoke"][1], "smoke_confirmed": out["smoke"][0],
        "non_smoke_sequences": out["no_smoke"][1], "non_smoke_alarms": out["no_smoke"][0],
        "median_frames_to_confirm": float(np.median(out["frames_to_confirm"])) if out["frames_to_confirm"] else None,
    }


# ---------------------------------------------------------------- simulated sequences


def episodes(alarm: np.ndarray, gap: int = EPISODE_GAP) -> int:
    """Number of separate alarm episodes (runs closer than `gap` frames merge)."""
    idx = np.where(alarm)[0]
    if len(idx) == 0:
        return 0
    return 1 + int((np.diff(idx) > gap).sum())


def detector_outputs(kind: str, frames: np.ndarray, cnn: OnnxSmokeDetector | None):
    """Run the detector once over a sequence; thresholds are applied later per alarm policy.

    kinds: 'cnn' (single-frame ONNX model), 'classical' (background subtraction),
    'hybrid' (CNN detections kept only where the classical detector sees change).
    """
    if kind == "cnn":
        return cnn.heatmaps(frames)
    det = ClassicalSmokeDetector()
    cls = [det.update(f) for f in frames]
    if kind == "classical":
        return cls
    return cnn.heatmaps(frames), cls


def overlaps(a, b, pad: float = 8.0) -> bool:
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def alarms_from_outputs(kind: str, outputs, level: int, confirmed: bool) -> np.ndarray:
    pol = policy_for_level(level, "classical" if kind == "classical" else "cnn")
    if kind == "classical":
        dets = outputs
    elif kind == "cnn":
        dets = [heat_to_detections(h, pol.score_thr) for h in outputs]
    else:
        heat, cls = outputs
        dets = [[d for d in heat_to_detections(h, pol.score_thr) if any(overlaps(d.box, c.box) for c in cs)]
                for h, cs in zip(heat, cls, strict=True)]
    if not confirmed:
        return np.array([any(d.score >= pol.score_thr for d in ds) for ds in dets])
    conf = TemporalConfirmer(pol)
    return np.array([conf.update(ds) for ds in dets])


def smoke_sequence_outputs(backgrounds: np.ndarray, kinds, cnn, n_seq: int = 40, n: int = 120, seed: int = 0):
    """Render simulated smoke sequences plus a smoke-free control of each (same weather and noise).

    Returns [(first_visible, {kind: outputs}, {kind: control_outputs})].
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_seq):
        bg = backgrounds[s % len(backgrounds)]
        plume = random_plume(rng, seed * 1000 + s)
        scen = "mixed" if s % 2 else "clear"
        ign = int(rng.integers(15, 30))
        seq = render_sequence(bg, n, scen, seed * 1000 + s, plume=plume, ignition=ign)
        ctl = render_sequence(bg, n, scen, seed * 1000 + s)
        out.append((seq.first_visible, {k: detector_outputs(k, seq.frames, cnn) for k in kinds},
                    {k: detector_outputs(k, ctl.frames, cnn) for k in kinds}))
    return out


def eval_smoke(outputs, kind: str, level: int, confirmed: bool) -> dict:
    """Detection counts only if the alarm fires on the plume run and not on its smoke-free control."""
    detected, ttd, total = 0, [], 0
    for first_visible, outs, ctl in outputs:
        if first_visible is None:
            continue
        total += 1
        a = alarms_from_outputs(kind, outs[kind], level, confirmed)
        c = alarms_from_outputs(kind, ctl[kind], level, confirmed)
        hit = np.where((a & ~c)[first_visible:])[0]
        if len(hit):
            detected += 1
            ttd.append(int(hit[0]))
    return {
        "sequences": total, "detected": detected, "recall": detected / max(total, 1),
        "median_ttd_frames": float(np.median(ttd)) if ttd else None,
        "p90_ttd_frames": float(np.percentile(ttd, 90)) if ttd else None,
        "median_ttd_minutes": float(np.median(ttd)) * FRAME_INTERVAL_S / 60 if ttd else None,
    }


SCENARIOS = ("clear", "clouds", "fog", "glare_dust", "mixed")


def camera_day_outputs(backgrounds: np.ndarray, kinds, cnn, per_scenario: int = 6, seed: int = 100):
    """Render smoke-free camera-days once; return [(scenario, {kind: outputs})]."""
    out = []
    for scen in SCENARIOS:
        for k in range(per_scenario):
            seq = render_sequence(backgrounds[(k * 3 + len(scen)) % len(backgrounds)], CAMERA_DAY_FRAMES, scen,
                                  seed + k * 17 + len(scen))
            out.append((scen, {kk: detector_outputs(kk, seq.frames, cnn) for kk in kinds}))
    return out


def eval_false_alarms(days, kind: str, level: int, confirmed: bool) -> dict:
    """False-alarm episodes per camera-day (14 h daylight, 1 frame/min) on smoke-free sequences."""
    per: dict[str, list[int]] = {}
    for scen, outs in days:
        per.setdefault(scen, []).append(episodes(alarms_from_outputs(kind, outs[kind], level, confirmed)))
    res = {s: {"camera_days": len(v), "episodes": int(sum(v)), "per_camera_day": sum(v) / len(v)} for s, v in per.items()}
    allv = [x for v in per.values() for x in v]
    res["all"] = {"camera_days": len(allv), "episodes": int(sum(allv)), "per_camera_day": sum(allv) / len(allv)}
    return res


# ---------------------------------------------------------------- latency / size


def latency_ms(fn, repeats: int = 200, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def edge_costs(cnn: OnnxSmokeDetector, frame: np.ndarray) -> dict:
    x = cnn.preprocess(frame)
    onnx_lat = latency_ms(lambda: cnn.sess.run(None, {cnn.input_name: x}))
    det = ClassicalSmokeDetector()
    det.update(frame)
    cl_lat = latency_ms(lambda: det.update(frame))
    return {"onnx_bytes": MODEL_PATH.stat().st_size, "cnn_1thread": onnx_lat, "classical": cl_lat,
            "grid": [GH, GW], "stride": STRIDE}


def save_json(obj: dict, name: str) -> None:
    p = ROOT / "docs" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float))
