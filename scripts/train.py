"""Train the tiny heat-map CNN on the prepared Pyro-SDIS subset and export ONNX.

Dev-only (needs torch). CI and the demo only need the committed models/smoke_cnn.onnx.
    python scripts/prepare_data.py && python scripts/train.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smokecam.common import GH, GW, MODEL_PATH, ROOT, boxes_to_grid, load_image  # noqa: E402


def build_model() -> nn.Module:
    def block(i, o, s=1, d=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, padding=d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    return nn.Sequential(
        block(3, 16, 2),  # 80x48
        block(16, 32, 2),  # 40x24
        block(32, 48, 2),  # 20x12 (stride 8)
        block(48, 48, 1, 1),
        block(48, 64, 1, 2),
        block(64, 64, 1, 4),
        nn.Conv2d(64, 1, 1),
    )


def val_camera(camera: str) -> bool:
    return int(hashlib.md5(("val" + camera).encode()).hexdigest(), 16) % 6 == 0


def augment(x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flip = torch.rand(len(x)) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    g = torch.where(flip[:, None, None], g.flip(2), g)
    b = len(x)
    contrast = 0.7 + 0.6 * torch.rand(b, 1, 1, 1)
    bright = 0.15 * (torch.rand(b, 1, 1, 1) - 0.5)
    gamma = 0.75 + 0.5 * torch.rand(b, 1, 1, 1)
    tint = 1 + 0.12 * (torch.rand(b, 3, 1, 1) - 0.5)
    x = ((x.clamp(1e-4, 1) ** gamma - 0.5) * contrast + 0.5 + bright) * tint
    return x.clamp(0, 1), g


def frame_prf(score: np.ndarray, label: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = score >= thr
    tp = float((pred & (label > 0)).sum())
    fp = float((pred & (label == 0)).sum())
    fn = float((~pred & (label > 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    meta = json.loads((ROOT / "cache" / "train.json").read_text())
    imgs = np.stack([load_image(ROOT / "cache" / "train" / m["name"]) for m in meta])
    grids = np.stack([boxes_to_grid([tuple(b) for b in m["boxes"]]) for m in meta])
    labels = np.array([1 if m["boxes"] else 0 for m in meta])
    is_val = np.array([val_camera(m["camera"]) for m in meta])
    print(f"train {int((~is_val).sum())} (pos {int(labels[~is_val].sum())}), val {int(is_val.sum())} (pos {int(labels[is_val].sum())})")
    X = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    G = torch.from_numpy(grids)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    model = build_model()
    n_par = sum(p.numel() for p in model.parameters())
    pos_w = torch.tensor(float((1 - G[tr].mean()) / max(G[tr].mean(), 1e-6)) ** 0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    epochs = 30
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(tr) + 63) // 64))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i : i + 64]
            xb, gb = augment(X[idx], G[idx])
            loss = loss_fn(model(xb)[:, 0], gb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                sc = torch.sigmoid(model(X[va])[:, 0]).amax(dim=(1, 2)).numpy()
            p, r, f = frame_prf(sc, labels[va], 0.5)
            print(f"ep {ep + 1} loss {tot / len(tr):.4f} val P {p:.3f} R {r:.3f} F1 {f:.3f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        sc = torch.sigmoid(model(X[va])[:, 0]).amax(dim=(1, 2)).numpy()
    thrs = np.linspace(0.2, 0.9, 36)
    best = max(thrs, key=lambda t: frame_prf(sc, labels[va], t)[2])
    p, r, f = frame_prf(sc, labels[va], best)
    print(f"val-chosen frame threshold {best:.2f}: P {p:.3f} R {r:.3f} F1 {f:.3f}")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    (ROOT / "cache").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "cache" / "model.pt")
    dummy = torch.zeros(1, 3, 96, 160)
    torch.onnx.export(
        model, dummy, str(MODEL_PATH), input_names=["frame"], output_names=["logits"],
        dynamic_axes={"frame": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False,
    )
    report = {"params": n_par, "grid": [GH, GW], "val_frame_threshold": round(float(best), 3),
              "val_precision": round(p, 4), "val_recall": round(r, 4), "val_f1": round(f, 4),
              "train_images": int(len(tr)), "val_images": int(len(va)), "epochs": epochs}
    (MODEL_PATH.parent / "train_report.json").write_text(json.dumps(report, indent=2))
    print("saved", MODEL_PATH, MODEL_PATH.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
