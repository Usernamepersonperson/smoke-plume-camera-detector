from smokecam.firedanger import AlarmPolicy
from smokecam.temporal import ConfirmConfig, Detection, TemporalConfirmer, iou

POL = AlarmPolicy(3, 0.5, 3, 5)


def det(x, y, w, h, s=0.9):
    return Detection((x, y, x + w, y + h), s)


def run(confirmer, seq):
    return [confirmer.update(d) for d in seq]


def test_iou_known_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_growing_static_plume_alarms_on_kth_frame():
    c = TemporalConfirmer(POL)
    out = run(c, [[det(60, 60 - i, 8 + i, 6 + i)] for i in range(6)])
    assert out == [False, False, True, True, True, True]


def test_single_frame_flash_never_alarms():
    c = TemporalConfirmer(POL)
    out = run(c, [[], [det(50, 50, 10, 10)], [], [], [], []])
    assert not any(out)


def test_low_scores_are_ignored():
    c = TemporalConfirmer(POL)
    assert not any(run(c, [[det(50, 50, 10, 10, s=0.3)] for _ in range(8)]))


def test_sideways_drifting_cloud_is_rejected():
    c = TemporalConfirmer(POL)
    out = run(c, [[det(10 + 3 * i, 20, 20, 10)] for i in range(8)])
    assert not any(out)


def test_drift_rule_can_be_relaxed():
    cfg = ConfirmConfig(max_drift_px_per_frame=10.0)
    c = TemporalConfirmer(POL, cfg)
    assert any(run(c, [[det(10 + 3 * i, 20, 20 + i, 10)] for i in range(8)]))


def test_fog_sized_blob_is_rejected():
    c = TemporalConfirmer(POL)
    assert not any(run(c, [[det(0, 0, 150, 80)] for _ in range(8)]))


def test_shrinking_static_blob_is_rejected_but_allowed_without_growth_rule():
    seq = [[det(60 + i, 40 + i, 30 - 2 * i, 20 - 2 * i)] for i in range(6)]
    assert not any(run(TemporalConfirmer(POL), seq))
    assert any(run(TemporalConfirmer(POL, ConfirmConfig(require_growth=False)), seq))


def test_track_survives_one_missed_frame():
    c = TemporalConfirmer(POL)
    seq = [[det(60, 60, 10, 10)], [det(60, 59, 10, 11)], [], [det(60, 58, 11, 12)]]
    assert run(c, seq)[-1] is True


def test_track_dies_after_n_misses_and_reset_clears_state():
    c = TemporalConfirmer(POL)
    run(c, [[det(60, 60, 10, 10)]] * 3 + [[]] * 6)
    assert c.tracks == []
    run(c, [[det(60, 60, 10, 10)]])
    c.reset()
    assert c.tracks == [] and c.idx == -1


def test_two_separate_objects_do_not_merge():
    c = TemporalConfirmer(POL)
    run(c, [[det(10, 60, 10, 10), det(120, 60, 10, 10)]] * 2)
    assert len(c.tracks) == 2


def test_stricter_policy_needs_more_frames():
    strict = TemporalConfirmer(AlarmPolicy(1, 0.6, 4, 6))
    out = run(strict, [[det(60, 60 - i, 8 + i, 6 + i)] for i in range(6)])
    assert out.index(True) == 3
