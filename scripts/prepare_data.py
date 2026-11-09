"""Build the small working set from the Pyro-SDIS parquet files (Apache-2.0).

Streams the four parquet files over HTTP range requests (nothing big is stored),
downsizes every kept frame to 160x96 and writes
  cache/train/*.jpg + cache/train.json   (git-ignored, training only)
  data/testset/*.jpg      + data/testset/meta.json  (committed, held-out cameras)
Cameras are split by hash so no camera appears in both sets.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smokecam.common import ROOT, H, W, camera_in_test, parse_boxes, save_image  # noqa: E402
from smokecam.remote import open_parquet  # noqa: E402

BASE = "https://huggingface.co/datasets/pyronear/pyro-sdis/resolve/main/data/"
FILES = [
    "val-00000-of-00001.parquet",
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
]
TRAIN_POS_PER_CAMERA = 120
TEST_TARGET = 700


def main() -> None:
    rng = random.Random(7)
    cache = ROOT / "cache" / "train"
    test_dir = ROOT / "data" / "testset"
    train_meta: list[dict] = []
    test_rows: list[tuple[dict, np.ndarray]] = []
    pos_count: dict[str, int] = defaultdict(int)
    stats = defaultdict(int)
    for fname in FILES:
        pf = open_parquet(BASE + fname)
        for rg in range(pf.metadata.num_row_groups):
            rows = pf.read_row_group(rg).to_pylist()
            for r in rows:
                cam = r["camera"]
                boxes = parse_boxes(r["annotations"])
                is_test = camera_in_test(cam)
                if not is_test and boxes and pos_count[cam] >= TRAIN_POS_PER_CAMERA:
                    continue
                arr = np.frombuffer(r["image"]["bytes"], np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    stats["undecodable"] += 1
                    continue
                rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                meta = {
                    "name": Path(r["image_name"]).stem + ".jpg",
                    "camera": cam,
                    "partner": r["partner"],
                    "date": r["date"],
                    "boxes": [[round(v, 4) for v in b] for b in boxes],
                }
                if is_test:
                    test_rows.append((meta, rgb))
                else:
                    pos_count[cam] += 1 if boxes else 0
                    save_image(cache / meta["name"], rgb)
                    train_meta.append(meta)
            print(fname, rg, len(train_meta), len(test_rows), flush=True)

    # test set: keep whole camera-days together (sequences stay intact), until the target size
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for meta, rgb in test_rows:
        groups[(meta["camera"], meta["date"][:10])].append((meta, rgb))
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen, total = [], 0
    for k in keys:
        if total >= TEST_TARGET:
            break
        chosen.extend(groups[k])
        total += len(groups[k])
    test_meta = []
    for meta, rgb in chosen:
        save_image(test_dir / meta["name"], rgb)
        test_meta.append(meta)
    test_meta.sort(key=lambda m: (m["camera"], m["date"]))
    (test_dir / "meta.json").write_text(json.dumps(test_meta, separators=(",", ":")))
    (ROOT / "cache" / "train.json").write_text(json.dumps(train_meta, separators=(",", ":")))
    print("train", len(train_meta), "pos", sum(1 for m in train_meta if m["boxes"]))
    print("test", len(test_meta), "pos", sum(1 for m in test_meta if m["boxes"]))
    print("train cameras", len({m["camera"] for m in train_meta}), "test cameras", len({m["camera"] for m in test_meta}))


if __name__ == "__main__":
    main()
