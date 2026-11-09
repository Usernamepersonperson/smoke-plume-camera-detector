import json

import numpy as np
import pytest

from smokecam import evaluate as ev
from smokecam.cnn import OnnxSmokeDetector, heat_to_detections
from smokecam.common import GH, GW, MODEL_PATH, TESTSET_DIR, H, W


@pytest.fixture(scope="module")
def cnn():
    return OnnxSmokeDetector()


@pytest.fixture(scope="module")
def testset():
    return ev.load_testset()


def test_model_is_small_and_present():
    assert MODEL_PATH.exists() and MODEL_PATH.stat().st_size < 1_000_000


def test_heatmap_shape_range_and_batching(cnn):
    x = np.random.default_rng(0).integers(0, 255, (5, H, W, 3), dtype=np.uint8)
    h = cnn.heatmaps(x, batch=2)
    assert h.shape == (5, GH, GW) and h.min() >= 0 and h.max() <= 1
    assert np.allclose(h[:1], cnn.heatmaps(x[:1]), atol=1e-5)


def test_single_frame_input_is_accepted(cnn):
    assert cnn.heatmaps(np.zeros((H, W, 3), np.uint8)).shape == (1, GH, GW)


@pytest.mark.parametrize("bad", [np.zeros((H, W), np.uint8), np.zeros((2, H, W, 4), np.uint8)])
def test_bad_input_shapes_raise(cnn, bad):
    with pytest.raises(ValueError):
        cnn.heatmaps(bad)


def test_empty_batch(cnn):
    assert cnn.heatmaps(np.zeros((0, H, W, 3), np.uint8)).shape[0] == 0


def test_heat_to_detections_blobs():
    heat = np.zeros((GH, GW), np.float32)
    heat[2:4, 3:6] = 0.9
    heat[8, 15] = 0.7
    dets = heat_to_detections(heat, 0.5)
    assert len(dets) == 2
    big = max(dets, key=lambda d: d.area)
    assert big.box == (24, 16, 48, 32) and big.score == pytest.approx(0.9)
    assert heat_to_detections(heat, 0.95) == []


def test_heat_to_detections_rejects_wrong_shape():
    with pytest.raises(ValueError):
        heat_to_detections(np.zeros((3, 3)), 0.5)


def test_testset_is_consistent_and_camera_disjoint_from_training_split(testset):
    frames, meta = testset
    assert frames.shape == (len(meta), H, W, 3)
    from smokecam.common import camera_in_test

    assert all(camera_in_test(m["camera"]) for m in meta)
    assert any(m["boxes"] for m in meta) and any(not m["boxes"] for m in meta)
    assert len(list(TESTSET_DIR.glob("*.jpg"))) == len(meta)


def test_frame_metrics_counts_add_up(cnn, testset):
    frames, meta = testset
    m = ev.frame_metrics(cnn.heatmaps(frames[:120]), meta[:120], 0.5)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 120
    assert 0 <= m["precision"] <= 1 and 0 <= m["recall"] <= 1


def test_frame_metrics_perfect_and_empty_cases():
    meta = [{"boxes": [[0.1, 0.1, 0.4, 0.4]]}, {"boxes": []}]
    heat = np.zeros((2, GH, GW), np.float32)
    heat[0, 2:5, 2:6] = 0.9
    m = ev.frame_metrics(heat, meta, 0.5)
    assert (m["precision"], m["recall"], m["false_positive_rate"]) == (1.0, 1.0, 0.0)
    assert m["localisation_hit_rate"] == 1.0
    none = ev.frame_metrics(np.zeros((2, GH, GW), np.float32), meta, 0.5)
    assert none["recall"] == 0.0 and none["precision"] == 0.0


def test_real_sequences_group_by_camera_and_gap():
    meta = [{"camera": "a", "date": f"2025-05-01T10-0{i}-00", "boxes": []} for i in range(4)]
    meta += [{"camera": "a", "date": "2025-05-01T15-00-00", "boxes": []}]
    meta += [{"camera": "b", "date": f"2025-05-01T10-0{i}-00", "boxes": []} for i in range(2)]
    seqs = ev.real_sequences(meta)
    assert [len(s) for s in seqs] == [4]  # lone frame and 2-frame sequence are dropped


def test_episodes_merge_close_alarms():
    a = np.zeros(300, bool)
    a[[10, 11, 12, 20]] = True  # one episode (gap < 30)
    a[[100, 101]] = True
    assert ev.episodes(a) == 2
    assert ev.episodes(np.zeros(10, bool)) == 0


def test_center_x_error_signs():
    meta = [{"boxes": [[0.25, 0.25, 0.5, 0.5]]}]
    heat = np.zeros((1, GH, GW), np.float32)
    heat[0, 3:6, 5:9] = 0.9  # box x 40..72 vs GT 40..80: centre 56 vs 60
    assert ev.center_x_error(heat, meta, 0.5) == [-4.0]


def test_end_to_end_alarm_on_simulated_plume(cnn, testset):
    frames, meta = testset
    bgs = np.stack([f for f, m in zip(frames, meta, strict=True) if not m["boxes"]])
    out = ev.smoke_sequence_outputs(bgs, ("classical",), cnn, n_seq=4, n=100)
    r = ev.eval_smoke(out, "classical", 5, True)
    assert r["sequences"] == 4 and 0 <= r["recall"] <= 1


def test_quick_demo_writes_results(tmp_path, monkeypatch):
    from smokecam import __main__ as cli

    monkeypatch.setattr(ev, "ROOT", tmp_path)
    res = cli.run_demo(quick=True, figures=False)
    assert (tmp_path / "docs" / "results.json").exists()
    saved = json.loads((tmp_path / "docs" / "results.json").read_text())
    assert saved["testset"]["frames"] == res["testset"]["frames"]
    assert "cnn|level3|confirmed" in res["simulated"]
