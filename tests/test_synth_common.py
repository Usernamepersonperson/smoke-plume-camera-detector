import numpy as np
import pytest

from smokecam.classical import ClassicalSmokeDetector
from smokecam.common import GH, GW, H, W, boxes_to_grid, camera_in_test, parse_boxes
from smokecam.synth import PlumeParams, plume_alpha, plume_box, random_plume, render_sequence

BG = np.full((H, W, 3), 90, np.uint8)
BG[: H // 2] = (150, 170, 200)


def test_parse_boxes_yolo():
    boxes = parse_boxes("0 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.2 0.2")
    assert boxes[0] == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert boxes[1][0] == 0.0  # clipped at the frame edge


@pytest.mark.parametrize("text", ["", "   ", "0 0.5", "0 a b c d", "garbage line"])
def test_parse_boxes_malformed(text):
    assert parse_boxes(text) == []


def test_boxes_to_grid_marks_overlapping_cells():
    g = boxes_to_grid([(0.0, 0.0, 0.5, 0.5)])
    assert g.shape == (GH, GW)
    assert g[: GH // 2, : GW // 2].all() and not g[GH // 2 :].any() and not g[:, GW // 2 :].any()
    assert boxes_to_grid([]).sum() == 0


def test_camera_split_is_deterministic_and_roughly_a_quarter():
    cams = [f"cam-{i}" for i in range(400)]
    flags = [camera_in_test(c) for c in cams]
    assert flags == [camera_in_test(c) for c in cams]
    assert 0.18 < sum(flags) / len(flags) < 0.32


def test_plume_empty_before_ignition_and_grows_afterwards():
    p = PlumeParams(origin=(80, 80), lean_deg=0, rise_speed=1.0, strength=0.6, seed=1)
    assert plume_alpha(p, -3).sum() == 0 and plume_alpha(p, 0).sum() == 0
    areas = [(plume_alpha(p, t) > 0.1).sum() for t in (5, 15, 40, 80)]
    assert areas == sorted(areas) and areas[-1] > areas[0]


def test_plume_rises_from_origin_and_leans_with_wind():
    right = PlumeParams(origin=(80, 85), lean_deg=30, rise_speed=1.0, strength=0.7, seed=2)
    box = plume_box(plume_alpha(right, 50))
    assert box is not None and box[1] < 70 and box[2] > 80  # top is above, drifts right


def test_plume_box_requires_minimum_visible_pixels():
    assert plume_box(np.zeros((H, W), np.float32)) is None
    a = np.zeros((H, W), np.float32)
    a[10:12, 10:12] = 0.5  # 4 px only
    assert plume_box(a) is None


def test_plume_is_deterministic_per_seed():
    a = PlumeParams(origin=(70, 80), lean_deg=5, rise_speed=1, strength=0.5, seed=3)
    b = PlumeParams(origin=(70, 80), lean_deg=5, rise_speed=1, strength=0.5, seed=3)
    assert np.array_equal(plume_alpha(a, 30), plume_alpha(b, 30))


def test_render_sequence_shapes_ignition_and_visibility():
    rng = np.random.default_rng(0)
    seq = render_sequence(BG, 60, "clear", 5, plume=random_plume(rng, 5), ignition=10)
    assert seq.frames.shape == (60, H, W, 3) and seq.frames.dtype == np.uint8
    assert seq.first_visible is not None and seq.first_visible > 10
    assert seq.boxes[0] is None


@pytest.mark.parametrize("scenario", ["clear", "clouds", "fog", "glare_dust", "mixed"])
def test_weather_sequences_contain_no_smoke_ground_truth(scenario):
    seq = render_sequence(BG, 120, scenario, 11)
    assert seq.first_visible is None and all(b is None for b in seq.boxes)
    assert seq.frames.shape[0] == 120


def test_classical_detector_finds_a_plume_but_not_static_scene():
    det = ClassicalSmokeDetector()
    still = render_sequence(BG, 40, "clear", 21)
    assert all(not det.update(f) for f in still.frames[:30])
    rng = np.random.default_rng(1)
    seq = render_sequence(BG, 70, "clear", 22, plume=random_plume(rng, 22), ignition=10)
    det.reset()
    found = [bool(det.update(f)) for f in seq.frames]
    assert any(found[seq.first_visible :])
    assert not any(found[:10])


def test_classical_ignores_global_exposure_step():
    det = ClassicalSmokeDetector()
    frames = [BG.copy() for _ in range(15)] + [(BG * 0.8).astype(np.uint8) for _ in range(10)]
    assert not any(det.update(f) for f in frames)
