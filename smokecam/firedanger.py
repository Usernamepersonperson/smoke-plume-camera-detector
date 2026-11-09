"""Federal forest-fire danger level (FOEN/BAFU) and the alarm policy it selects.

Source: FOEN/BAFU forest-fire danger warning layer served by the federal geo API
(layer ch.bafu.gefahren-waldbrand_warnung, listed on opendata.swiss). The live call
is optional: offline, the caller supplies the level.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

API = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
LAYER = "ch.bafu.gefahren-waldbrand_warnung"

LEVELS = {"low": 1, "moderate": 2, "considerable": 3, "high": 4, "very high": 5}
# German titles as served by the layer ("Keine oder geringe Gefahr" = level 1)
LEVELS_DE = {"geringe": 1, "mässige": 2, "massige": 2, "erhebliche": 3, "grosse": 4, "sehr grosse": 5}


@dataclass(frozen=True)
class DangerInfo:
    level: int
    title: str
    region: str
    valid_from: str


@dataclass(frozen=True)
class AlarmPolicy:
    """How eager the alarm is: score threshold and k-of-n frame confirmation."""

    level: int
    score_thr: float
    k: int
    n: int


# score thresholds per danger band: (classical blob score, CNN cell probability).
# The CNN is trained with a down-weighted positive class, so its probabilities are
# low; its thresholds were tuned on validation cameras (see models/train_report.json).
THRESHOLDS = {"low": (0.60, 0.30), "mid": (0.50, 0.20), "high": (0.40, 0.12)}


def policy_for_level(level: int, kind: str = "classical") -> AlarmPolicy:
    """Higher danger -> lower threshold and faster confirmation (more sensitive)."""
    if kind not in ("classical", "cnn"):
        raise ValueError("kind must be 'classical' or 'cnn'")
    i = 1 if kind == "cnn" else 0
    if level <= 2:
        return AlarmPolicy(level, THRESHOLDS["low"][i], 4, 6)
    if level == 3:
        return AlarmPolicy(level, THRESHOLDS["mid"][i], 3, 5)
    return AlarmPolicy(level, THRESHOLDS["high"][i], 2, 4)


def level_from_title(title_en: str | None, title_de: str | None = None) -> int | None:
    if title_en:
        key = title_en.lower().replace(" danger", "").strip()
        if key in LEVELS:
            return LEVELS[key]
    if title_de:
        low = title_de.lower()
        for k in sorted(LEVELS_DE, key=len, reverse=True):
            if k in low:
                return LEVELS_DE[k]
    return None


def parse_identify(payload: dict) -> DangerInfo | None:
    """Pick the most severe region among those returned for a point (conservative)."""
    best: DangerInfo | None = None
    for res in payload.get("results", []):
        a = res.get("attributes", {})
        lvl = level_from_title(a.get("title_en"), a.get("title_de"))
        if lvl is None:
            continue
        info = DangerInfo(
            lvl, a.get("title_en") or a.get("title_de") or "", a.get("name_en") or "", a.get("valid_from") or ""
        )
        if best is None or info.level > best.level:
            best = info
    return best


def fetch_level(lat: float, lon: float, timeout: float = 10.0) -> DangerInfo | None:
    """Current danger level at a WGS84 point, or None if the service is unreachable."""
    q = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "layers": f"all:{LAYER}",
        "mapExtent": f"{lon - 0.5},{lat - 0.5},{lon + 0.5},{lat + 0.5}",
        "imageDisplay": "100,100,96",
        "tolerance": "0",
        "sr": "4326",
        "returnGeometry": "false",
    }
    try:
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=timeout) as r:
            return parse_identify(json.load(r))
    except (OSError, ValueError):
        return None
