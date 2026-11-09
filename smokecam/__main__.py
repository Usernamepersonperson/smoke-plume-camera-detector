"""CLI: python -m smokecam demo | danger | detect IMAGE | triangulate"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import evaluate as ev
from .cnn import OnnxSmokeDetector
from .common import load_image
from .firedanger import fetch_level, policy_for_level
from .triangsim import CAM_A, CAM_B, column_sigma_deg, simulate

LEVEL_THR = {l: policy_for_level(l, "cnn").score_thr for l in (1, 3, 5)}
SITES = {"Ticino (Lugano)": (46.00, 8.95), "Valais (Sion)": (46.23, 7.36), "Graubuenden (Chur)": (46.85, 9.53)}


def run_demo(quick: bool = False, figures: bool = True) -> dict:
    cnn = OnnxSmokeDetector()
    frames, meta = ev.load_testset()
    heat = cnn.heatmaps(frames)
    res: dict = {"testset": {"frames": len(meta), "cameras": len({m["camera"] for m in meta})}}
    res["real_frames"] = {str(l): ev.frame_metrics(heat, meta, LEVEL_THR[l]) for l in LEVEL_THR}
    res["real_roc"] = ev.roc_summary(heat, meta)
    res["real_sequences_level3"] = ev.eval_real_sequences(heat, meta, 3)
    errs = ev.center_x_error(heat, meta, LEVEL_THR[3])
    sigma_px = float(np.std(errs)) if errs else 0.0
    res["column_error"] = {"n": len(errs), "std_px_of_160": sigma_px, "mean_px": float(np.mean(errs)) if errs else 0.0}

    bgs = np.stack([frames[i] for i, m in enumerate(meta) if not m["boxes"]])
    kinds = ("cnn", "classical", "hybrid")
    n_seq, per = (8, 1) if quick else (40, 6)
    smoke = ev.smoke_sequence_outputs(bgs, kinds, cnn, n_seq=n_seq)
    days = ev.camera_day_outputs(bgs, kinds, cnn, per_scenario=per)
    res["simulated"] = {}
    for kind in kinds:
        for level in LEVEL_THR:
            for confirmed in (False, True):
                key = f"{kind}|level{level}|{'confirmed' if confirmed else 'raw'}"
                res["simulated"][key] = {
                    "smoke": ev.eval_smoke(smoke, kind, level, confirmed),
                    "false_alarms": ev.eval_false_alarms(days, kind, level, confirmed),
                }
    res["edge"] = ev.edge_costs(cnn, frames[0])
    res["triangulation"] = {
        "measured_sigma_px": simulate(sigma_px, 0.5, n=3000),
        "measured_sigma_deg": column_sigma_deg(sigma_px),
        "sweep": [simulate(s, h, n=2000) for s, h in ((0.5, 0.1), (1.0, 0.25), (2.0, 0.5), (4.0, 1.0))],
    }
    snap = {}
    if not quick:
        for name, (lat, lon) in SITES.items():
            info = fetch_level(lat, lon)
            snap[name] = None if info is None else {"level": info.level, "title": info.title, "region": info.region,
                                                    "valid_from": info.valid_from}
    res["danger_snapshot"] = snap
    ev.save_json(res, "results.json")
    if figures:
        from . import figures

        figures.make_all(res, cnn, frames, meta, heat)
    return res


def print_summary(res: dict) -> None:
    print(f"held-out frames: {res['testset']['frames']} from {res['testset']['cameras']} unseen cameras (REAL)")
    for lvl, m in res["real_frames"].items():
        print(f"  level {lvl} thr: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"FPR {m['false_positive_rate']:.3f} loc {m['localisation_hit_rate']:.3f}")
    print("real roc:", res["real_roc"])
    print("real sequences:", res["real_sequences_level3"])
    print("simulated (smoke recall / median ttd min / false alarms per camera-day):")
    for k, v in res["simulated"].items():
        print(f"  {k:30s} {v['smoke']['recall']:.2f} {v['smoke']['median_ttd_minutes']} "
              f"{v['false_alarms']['all']['per_camera_day']:.2f}")
    print("edge:", res["edge"])
    print("triangulation:", res["triangulation"]["measured_sigma_px"])
    print("danger:", res["danger_snapshot"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="smokecam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="reproduce every README number and figure")
    d.add_argument("--quick", action="store_true", help="small run for CI")
    d.add_argument("--no-figures", action="store_true")
    sub.add_parser("figures", help="redraw docs/*.png from docs/results.json")
    sub.add_parser("danger", help="print the current federal forest-fire danger level for sample sites")
    det = sub.add_parser("detect", help="run the ONNX detector on one image")
    det.add_argument("image")
    det.add_argument("--level", type=int, default=3)
    sub.add_parser("triangulate", help="print the two-camera triangulation simulation")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        print_summary(run_demo(a.quick, not a.no_figures))
    elif a.cmd == "figures":
        from . import figures

        cnn = OnnxSmokeDetector()
        frames, meta = ev.load_testset()
        res = json.loads((ev.ROOT / "docs" / "results.json").read_text())
        figures.make_all(res, cnn, frames, meta, cnn.heatmaps(frames))
    elif a.cmd == "danger":
        for name, (lat, lon) in SITES.items():
            info = fetch_level(lat, lon)
            if info is None:
                print(f"{name}: unavailable (offline or service error)")
            else:
                p = policy_for_level(info.level)
                print(f"{name}: level {info.level} ({info.title}), valid from {info.valid_from} -> "
                      f"thr {p.score_thr}, {p.k} of {p.n} frames")
    elif a.cmd == "detect":
        cnn = OnnxSmokeDetector()
        pol = policy_for_level(a.level, "cnn")
        dets = cnn.detect(load_image(__import__("pathlib").Path(a.image)), pol.score_thr)
        print(json.dumps([{"box": d.box, "score": round(d.score, 3)} for d in dets]))
    elif a.cmd == "triangulate":
        print(json.dumps({"cameras": [CAM_A.name, CAM_B.name], **simulate(2.0, 0.5, n=3000)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
