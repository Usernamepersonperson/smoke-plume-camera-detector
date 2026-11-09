"""Monte-Carlo triangulation error of a two-camera pair (SIMULATED geometry).

Camera sites are illustrative hill positions in southern Ticino, not real installations.
Bearing noise = detector column error (measured on held-out frames) + heading calibration error.
"""

from __future__ import annotations

import math

import numpy as np

from .common import W
from .geo import Camera, bearing_from_pixel, distance_m, from_local, to_local, triangulate

# approximate summit coordinates (illustrative; no claim that cameras exist there)
CAM_A = Camera("A (Monte Generoso area)", 45.9314, 9.0207, 0.0, 90.0)
CAM_B = Camera("B (Monte Tamaro area)", 46.1056, 8.8056, 0.0, 90.0)


def simulate(sigma_col_px: float, sigma_heading_deg: float = 0.5, n: int = 2000, seed: int = 0,
             cam_a: Camera = CAM_A, cam_b: Camera = CAM_B, width_px: int = W,
             min_km: float = 3.0, max_km: float = 20.0) -> dict:
    """Fires placed 3-20 km from the pair centroid; cameras pan so the smoke is in frame."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = (cam_a.lat + cam_b.lat) / 2, (cam_a.lon + cam_b.lon) / 2
    errs, failed = [], 0
    for _ in range(n):
        r = rng.uniform(min_km, max_km) * 1000
        th = rng.uniform(0, 2 * math.pi)
        lat_f, lon_f = from_local(r * math.sin(th), r * math.cos(th), lat0, lon0)
        # true compass bearing from each camera to the fire
        cams = []
        for cam in (cam_a, cam_b):
            e, nn = to_local(lat_f, lon_f, cam.lat, cam.lon)
            true_b = math.degrees(math.atan2(e, nn)) % 360
            x_true = float(rng.uniform(0.2, 0.8) * width_px)  # where the smoke sits in the frame
            # heading that puts the true bearing at x_true, then perturbed by calibration error
            off = bearing_from_pixel(x_true, width_px, 0.0, cam.hfov_deg)
            off = (off + 180) % 360 - 180
            heading = (true_b - off) % 360
            x_meas = x_true + rng.normal(0, sigma_col_px)
            noisy = Camera(cam.name, cam.lat, cam.lon, heading + rng.normal(0, sigma_heading_deg), cam.hfov_deg)
            cams.append((noisy, x_meas))
        (ca, xa), (cb, xb) = cams
        est = triangulate(ca, xa, cb, xb, width_px)
        if est is None:
            failed += 1
            continue
        errs.append(distance_m(est[0], est[1], lat_f, lon_f))
    errs_a = np.array(errs)
    return {
        "sigma_col_px": sigma_col_px, "sigma_heading_deg": sigma_heading_deg, "trials": n, "no_fix": failed,
        "median_m": float(np.median(errs_a)) if len(errs_a) else None,
        "p90_m": float(np.percentile(errs_a, 90)) if len(errs_a) else None,
        "baseline_km": distance_m(cam_a.lat, cam_a.lon, cam_b.lat, cam_b.lon) / 1000,
    }


def column_sigma_deg(sigma_col_px: float, width_px: int = W, hfov_deg: float = 90.0) -> float:
    """Approximate bearing noise (degrees) of a column error near the image centre."""
    return math.degrees(math.atan(2 * sigma_col_px / width_px * math.tan(math.radians(hfov_deg) / 2)))


def example(width_px: int = W, sigma_col_px: float = 2.0, seed: int = 3) -> dict:
    """One illustrative fix for the map figure."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = (CAM_A.lat + CAM_B.lat) / 2, (CAM_A.lon + CAM_B.lon) / 2
    lat_f, lon_f = from_local(2500.0, 8000.0, lat0, lon0)
    out = {"fire": (lat_f, lon_f), "cams": []}
    for cam in (CAM_A, CAM_B):
        e, nn = to_local(lat_f, lon_f, cam.lat, cam.lon)
        true_b = math.degrees(math.atan2(e, nn)) % 360
        x_true = width_px * (0.40 if cam is CAM_A else 0.62)
        off = (bearing_from_pixel(x_true, width_px, 0.0, cam.hfov_deg) + 180) % 360 - 180
        heading = (true_b - off) % 360
        x_meas = x_true + rng.normal(0, sigma_col_px)
        c = Camera(cam.name, cam.lat, cam.lon, heading, cam.hfov_deg)
        out["cams"].append((c, x_meas, bearing_from_pixel(x_meas, width_px, heading, cam.hfov_deg)))
    (ca, xa, _), (cb, xb, _) = out["cams"]
    out["estimate"] = triangulate(ca, xa, cb, xb, width_px)
    return out
