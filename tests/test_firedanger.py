import pytest

from smokecam import firedanger as fd

SAMPLE = {
    "results": [
        {"attributes": {"name_en": "Northern Ticino (TI)", "title_en": "Low danger",
                        "title_de": "Keine oder geringe Gefahr", "valid_from": "03.11.2026"}},
        {"attributes": {"name_en": "Central Ticino (TI)", "title_en": "Considerable danger",
                        "title_de": "Erhebliche Gefahr", "valid_from": "29.10.2026"}},
    ]
}


def test_parse_takes_most_severe_region():
    info = fd.parse_identify(SAMPLE)
    assert info.level == 3
    assert info.region.startswith("Central Ticino")
    assert info.valid_from == "29.10.2026"


@pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": [{"attributes": {}}]},
                                     {"results": [{"attributes": {"title_en": "Unknown"}}]}])
def test_parse_handles_empty_or_malformed(payload):
    assert fd.parse_identify(payload) is None


@pytest.mark.parametrize("en,de,lvl", [
    ("Low danger", None, 1), ("Moderate danger", None, 2), ("Considerable danger", None, 3),
    ("High danger", None, 4), ("Very high danger", None, 5),
    (None, "Keine oder geringe Gefahr", 1), (None, "Mässige Gefahr", 2),
    (None, "Erhebliche Gefahr", 3), (None, "Grosse Gefahr", 4), (None, "Sehr grosse Gefahr", 5),
])
def test_level_titles(en, de, lvl):
    assert fd.level_from_title(en, de) == lvl


def test_level_unknown_is_none():
    assert fd.level_from_title(None, None) is None
    assert fd.level_from_title("Weird", "Seltsam") is None


def test_policy_gets_more_sensitive_with_danger():
    lo, mid, hi = (fd.policy_for_level(i) for i in (1, 3, 5))
    assert lo.score_thr > mid.score_thr > hi.score_thr
    assert lo.k > mid.k > hi.k
    assert fd.policy_for_level(2) == fd.policy_for_level(1).__class__(2, lo.score_thr, lo.k, lo.n)
    assert fd.policy_for_level(4).score_thr == hi.score_thr


def test_cnn_thresholds_follow_same_ordering_and_bad_kind_raises():
    t = [fd.policy_for_level(i, "cnn").score_thr for i in (1, 3, 5)]
    assert t[0] > t[1] > t[2]
    with pytest.raises(ValueError):
        fd.policy_for_level(3, "lidar")


def test_fetch_returns_none_when_offline(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(fd.urllib.request, "urlopen", boom)
    assert fd.fetch_level(46.2, 8.8) is None
