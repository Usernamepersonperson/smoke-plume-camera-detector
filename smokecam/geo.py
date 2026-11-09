"""Bearings from pixel columns and two-camera triangulation (flat-earth local frame)."""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_R = 6_371_008.8  # mean Earth radius in metres


@dataclass(frozen=True)
class Camera:
    name: str
    lat: float
    lon: float
    heading_deg: float  # compass direction of the optical axis (0 = north, clockwise)
    hfov_deg: float  # horizontal field of view of the frame


def bearing_from_pixel(x_px: float, width_px: int, heading_deg: float, hfov_deg: float) -> float:
    """Compass bearing of image column x_px for a pinhole camera (not linear in x)."""
    if width_px <= 0 or not 0 < hfov_deg < 180:
        raise ValueError("width must be > 0 and 0 < hfov < 180")
    f = (width_px / 2) / math.tan(math.radians(hfov_deg) / 2)
    off = math.degrees(math.atan((x_px - width_px / 2) / f))
    return (heading_deg + off) % 360.0


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """(east, north) metres of a point relative to the origin (equirectangular, fine below ~50 km)."""
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * EARTH_R
    return east, north


def from_local(east: float, north: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat = lat0 + math.degrees(north / EARTH_R)
    lon = lon0 + math.degrees(east / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (haversine) distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def intersect_rays(
    p1: tuple[float, float], b1_deg: float, p2: tuple[float, float], b2_deg: float, min_angle_deg: float = 5.0
) -> tuple[float, float] | None:
    """Intersection of two bearing rays in the local frame, or None.

    None when the rays are nearly parallel (< min_angle_deg apart, ill-conditioned)
    or when the crossing lies behind either camera.
    """
    d1 = (math.sin(math.radians(b1_deg)), math.cos(math.radians(b1_deg)))
    d2 = (math.sin(math.radians(b2_deg)), math.cos(math.radians(b2_deg)))
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < math.sin(math.radians(min_angle_deg)):
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t1 = (dx * d2[1] - dy * d2[0]) / cross
    t2 = (dx * d1[1] - dy * d1[0]) / cross
    if t1 < 0 or t2 < 0:
        return None
    return p1[0] + t1 * d1[0], p1[1] + t1 * d1[1]


def triangulate(
    cam_a: Camera, x_a: float, cam_b: Camera, x_b: float, width_px: int
) -> tuple[float, float] | None:
    """Fire position (lat, lon) from the smoke column seen by two cameras, or None."""
    lat0, lon0 = cam_a.lat, cam_a.lon
    pa = (0.0, 0.0)
    pb = to_local(cam_b.lat, cam_b.lon, lat0, lon0)
    ba = bearing_from_pixel(x_a, width_px, cam_a.heading_deg, cam_a.hfov_deg)
    bb = bearing_from_pixel(x_b, width_px, cam_b.heading_deg, cam_b.hfov_deg)
    hit = intersect_rays(pa, ba, pb, bb)
    if hit is None:
        return None
    return from_local(hit[0], hit[1], lat0, lon0)
