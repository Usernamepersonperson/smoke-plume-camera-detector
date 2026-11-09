"""Figures: docs/figure.png (1600x900) and docs/post-figure.png (960x700)."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from . import evaluate as ev  # noqa: E402
from .cnn import heat_to_detections  # noqa: E402
from .common import ROOT, H, W  # noqa: E402
from .firedanger import policy_for_level  # noqa: E402
from .geo import distance_m, to_local  # noqa: E402
from .synth import random_plume, render_sequence  # noqa: E402
from .triangsim import example  # noqa: E402

BG, GRID, TXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACC, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
NAMES = {"cnn": "CNN", "classical": "classic", "hybrid": "hybrid"}
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def _style(ax, grid=True):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    if grid:
        ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def best_example(heat, meta) -> int:
    best, bi = -1.0, 0
    for i, (h, m) in enumerate(zip(heat, meta, strict=True)):
        if not m["boxes"]:
            continue
        dets = heat_to_detections(h, policy_for_level(3, "cnn").score_thr)
        gts = [ev.box_px(b) for b in m["boxes"]]
        for d in dets:
            hit = any(ev.overlaps(d.box, g, 0.0) for g in gts)
            area = (gts[0][2] - gts[0][0]) * (gts[0][3] - gts[0][1])
            score = d.score + (1 if hit else 0) + min(area, 900) / 3000
            if hit and score > best:
                best, bi = score, i
    return bi


def draw_frame(ax, frame, dets, gts, fs=11):
    ax.imshow(frame, interpolation="bicubic", extent=(0, W, H, 0))
    for g in gts:
        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=GREEN, lw=1.6, ls="--"))
    for d in dets:
        b = d.box
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=RED, lw=2.4))
        ax.text(b[0], max(b[1] - 2, 6), f"smoke {d.score:.2f}", color=RED, fontsize=fs, fontweight="bold",
                va="bottom", bbox=dict(fc=BG, ec="none", alpha=0.7, pad=1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def draw_map(ax, fs=11, label=True):
    ex = example()
    ca = ex["cams"][0][0]
    pts = {}
    for cam, _, brg in ex["cams"]:
        e, n = to_local(cam.lat, cam.lon, ca.lat, ca.lon)
        pts[cam.name] = (e / 1000, n / 1000, brg, cam)
    fe, fn = (v / 1000 for v in to_local(*ex["fire"], ca.lat, ca.lon))
    ee, en = (v / 1000 for v in to_local(*ex["estimate"], ca.lat, ca.lon))
    cols = [ACC, VIOLET]
    for (name, (x, y, brg, cam)), col in zip(pts.items(), cols, strict=True):
        L = 1.35 * math.hypot(fe - x, fn - y)
        lo, hi = math.radians(cam.heading_deg - cam.hfov_deg / 2), math.radians(cam.heading_deg + cam.hfov_deg / 2)
        ax.add_patch(Polygon([(x, y), (x + L * math.sin(lo), y + L * math.cos(lo)),
                              (x + L * math.sin(hi), y + L * math.cos(hi))], closed=True, fc=col, alpha=0.10, ec="none"))
        ax.plot([x, x + L * math.sin(math.radians(brg))], [y, y + L * math.cos(math.radians(brg))], color=col, lw=2.2)
        ax.scatter([x], [y], s=90, color=col, zorder=5, marker="^")
        ax.text(x + 0.6, y + 0.6, name.split(" ")[0] + " camera", color=col, fontsize=fs, fontweight="bold")
    ax.scatter([fe], [fn], s=130, color=RED, zorder=6, marker="x", lw=3)
    ax.scatter([ee], [en], s=70, facecolors="none", edgecolors=AMBER, zorder=7, lw=2)
    err = distance_m(*ex["fire"], *ex["estimate"])
    ax.annotate(f"fix error {err:.0f} m", (ee, en), (ee + 1.4, en - 1.0), color=AMBER, fontsize=fs,
                fontweight="bold")
    ax.annotate("true fire", (fe, fn), (fe + 1.4, fn + 1.2), color=RED, fontsize=fs, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)", color=MUTED, fontsize=fs)
    ax.set_ylabel("north (km)", color=MUTED, fontsize=fs)
    _style(ax)
    xs = [p[0] for p in pts.values()] + [fe]
    ys = [p[1] for p in pts.values()] + [fn]
    ax.set_xlim(min(xs) - 3, max(xs) + 6)
    ax.set_ylim(min(ys) - 3, max(ys) + 4)


def title(fig, text, sub, y=0.965, fs=13.7):
    fig.text(0.03, y, text, color=TXT, fontsize=fs, fontweight="bold", va="top")
    fig.text(0.03, y - 0.055 if fs > 12 else y - 0.04, sub, color=MUTED, fontsize=11, va="top")


def make_all(res, cnn, frames, meta, heat) -> None:
    i = best_example(heat, meta)
    pol = policy_for_level(3, "cnn")
    dets = heat_to_detections(heat[i], pol.score_thr)
    gts = [ev.box_px(b) for b in meta[i]["boxes"]]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # post figure: frame + sidebar stats + map
    make_post_full(res, frames[i], dets, gts, docs / "post-figure.png")

    # main figure
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    title(fig, "Smoke-plume camera detector: CPU edge model + temporal confirmation + two-camera fix",
          "Real: 708 held-out frames from unseen cameras (Pyro-SDIS, Apache-2.0). Simulated: weather look-alikes, plumes, map geometry.",
          y=0.975)
    ax = fig.add_axes([0.03, 0.55, 0.30, 0.30])
    draw_frame(ax, frames[i], dets, gts, fs=11)
    ax.set_title("Real held-out frame", color=TXT, fontsize=12, loc="left")

    # timeline of one simulated sequence
    rng = np.random.default_rng(4)
    bgs = np.stack([frames[k] for k, m in enumerate(meta) if not m["boxes"]])
    seq = render_sequence(bgs[3], 120, "clear", 77, plume=random_plume(rng, 77), ignition=20)
    cn = ev.detector_outputs("cnn", seq.frames, cnn)
    cl = ev.detector_outputs("classical", seq.frames, cnn)
    ax2 = fig.add_axes([0.40, 0.55, 0.27, 0.30])
    _style(ax2)
    ax2.plot([max([d.score for d in ds], default=0) for ds in [heat_to_detections(h, 0.01) for h in cn]],
             color=ACC, lw=2, label="CNN score")
    ax2.plot([max([d.score for d in ds], default=0) for ds in cl], color=VIOLET, lw=2, label="classical score")
    a_cnn = ev.alarms_from_outputs("cnn", cn, 3, True)
    a_cl = ev.alarms_from_outputs("classical", cl, 3, True)
    a_hy = ev.alarms_from_outputs("hybrid", (cn, cl), 3, True)
    ax2.axvline(20, color=GREY, ls=":")
    if seq.first_visible is not None:
        ax2.axvline(seq.first_visible, color=GREEN, ls="--", lw=1.6, label="first visible smoke")
    for arr, col, yy in ((a_cnn, ACC, 1.03), (a_cl, VIOLET, 1.03), (a_hy, AMBER, 1.03)):
        idx = np.where(arr)[0]
        if len(idx):
            ax2.scatter([idx[0]], [yy], color=col, marker="v", s=70, clip_on=False)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("frame (1 per minute, simulated)", color=MUTED)
    ax2.set_title("Simulated sequence, v = first confirmed alarm", color=TXT, fontsize=12, loc="left", pad=16)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TXT, fontsize=9, loc="center right")

    # false alarms raw vs confirmed
    ax3 = fig.add_axes([0.74, 0.55, 0.24, 0.30])
    _style(ax3)
    labels, raw, conf = [], [], []
    for kind in ("cnn", "classical", "hybrid"):
        for lvl in (1, 3, 5):
            labels.append(f"{NAMES[kind]}\nL{lvl}")
            raw.append(res["simulated"][f"{kind}|level{lvl}|raw"]["false_alarms"]["all"]["per_camera_day"])
            conf.append(res["simulated"][f"{kind}|level{lvl}|confirmed"]["false_alarms"]["all"]["per_camera_day"])
    x = np.arange(len(labels))
    ax3.bar(x - 0.2, raw, 0.4, color=RED, label="per-frame alarm")
    ax3.bar(x + 0.2, conf, 0.4, color=GREEN, label="temporal confirmation")
    ax3.set_xticks(x, labels, fontsize=8)
    ax3.set_ylim(0, max(raw + conf) * 1.45)
    ax3.set_title("False alarms per camera-day (simulated look-alikes)", color=TXT, fontsize=11, loc="left")
    ax3.legend(facecolor=BG, edgecolor=GRID, labelcolor=TXT, fontsize=8, loc="upper center", ncol=2)

    ax4 = fig.add_axes([0.05, 0.08, 0.40, 0.36])
    draw_map(ax4, fs=10)
    ax4.set_title("Two-camera bearing triangulation (simulated geometry, no base map)", color=TXT, fontsize=12, loc="left")

    # table
    ax5 = fig.add_axes([0.52, 0.06, 0.46, 0.38])
    ax5.axis("off")
    r3 = res["real_frames"]["3"]
    c3 = res["simulated"]["cnn|level3|confirmed"]
    k3 = res["simulated"]["classical|level3|confirmed"]
    h3 = res["simulated"]["hybrid|level3|confirmed"]
    e = res["edge"]
    tri = res["triangulation"]["sweep"][2]  # assumed 2 px column error, 0.5 deg heading error
    rows = [
        (f"REAL held-out frames (level-3, cell thr {policy_for_level(3, 'cnn').score_thr})", f"P {r3['precision']:.2f}  R {r3['recall']:.2f}  F1 {r3['f1']:.2f}", ACC),
        ("REAL smoke-free frames flagged / ROC AUC", f"{r3['false_positive_rate'] * 100:.0f} % of {r3['negatives']} / {res['real_roc']['auc']:.2f}", RED),
        ("SIM recall, CNN / classic / hybrid (confirmed)", " / ".join(f"{x['smoke']['recall'] * 100:.0f} %" for x in (c3, k3, h3)), GREEN),
        ("SIM median min to alarm after first visible", " / ".join(_m(x) for x in (c3, k3, h3)), GREEN),
        ("SIM false alarms per camera-day", " / ".join(f"{x['false_alarms']['all']['per_camera_day']:.1f}" for x in (c3, k3, h3)), AMBER),
        ("ONNX model size / CPU latency (1 thread)", f"{e['onnx_bytes'] / 1024:.0f} KB / {e['cnn_1thread']['median_ms']:.2f} ms", VIOLET),
        ("SIM triangulation error, assumed 2 px (median / p90)", f"{tri['median_m']:.0f} m / {tri['p90_m']:.0f} m", ACC),
    ]
    for k, (a, b, col) in enumerate(rows):
        y = 0.93 - k * 0.135
        ax5.text(0.0, y, a, color=MUTED, fontsize=11, va="center")
        ax5.text(1.0, y, b, color=col, fontsize=12, fontweight="bold", va="center", ha="right")
        ax5.plot([0, 1], [y - 0.06, y - 0.06], color=GRID, lw=0.8)
    fig.savefig(docs / "figure.png", dpi=100, facecolor=BG)
    plt.close(fig)


def _m(r) -> str:
    v = r["smoke"]["median_ttd_minutes"]
    return "n/a" if v is None else f"{v:.0f}"


def make_post_full(res, frame, dets, gts, path):
    """960x700 phone image: detection frame (top) and triangulation map (bottom)."""
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    title(fig, "Smoke flagged on a camera frame, then located from two cameras",
          "Top: real held-out frame, Pyro-SDIS (Apache-2.0)  |  map: simulated geometry", y=0.975)
    ax1 = fig.add_axes([0.03, 0.50, 0.50, 0.34])
    draw_frame(ax1, frame, dets, gts, fs=11)
    fig.text(0.03, 0.485, "green dashed: human label   red: detector", color=MUTED, fontsize=11, va="top")
    e = res["edge"]
    tri = res["triangulation"]["sweep"][2]
    h5 = res["simulated"]["hybrid|level5|confirmed"]
    side = [(f"{h5['smoke']['recall'] * 100:.0f} %", "simulated plumes alarmed, high danger", ACC),
            (f"{e['cnn_1thread']['median_ms']:.1f} ms", f"CPU per frame, {e['onnx_bytes'] / 1024:.0f} KB model", GREEN),
            (f"{tri['median_m']:.0f} m", "median fix error (simulated, 2 px)", AMBER)]
    for k, (big, small, col) in enumerate(side):
        y = 0.80 - k * 0.12
        fig.text(0.56, y, big, color=col, fontsize=22, fontweight="bold", va="center")
        fig.text(0.56, y - 0.045, small, color=MUTED, fontsize=11, va="center")
    ax3 = fig.add_axes([0.09, 0.07, 0.88, 0.34])
    draw_map(ax3, fs=11)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
