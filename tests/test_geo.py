import math

import pytest

from smokecam.geo import (
    Camera,
    bearing_from_pixel,
    distance_m,
    from_local,
    intersect_rays,
    to_local,
    triangulate,
)
from smokecam.triangsim import CAM_A, CAM_B, column_sigma_deg, simulate


def test_centre_pixel_is_heading():
    assert bearing_from_pixel(80, 160, 123.0, 90.0) == pytest.approx(123.0)


def test_edge_pixel_is_half_fov():
    assert bearing_from_pixel(160, 160, 0.0, 90.0) == pytest.approx(45.0)
    assert bearing_from_pixel(0, 160, 0.0, 90.0) == pytest.approx(315.0)


def test_bearing_wraps_and_is_pinhole_not_linear():
    b = bearing_from_pixel(120, 160, 350.0, 60.0)
    assert 0 <= b < 360
    # quarter-width offset is less than a quarter of the FOV for a pinhole
    off = (bearing_from_pixel(120, 160, 0.0, 90.0) + 180) % 360 - 180
    assert off == pytest.approx(math.degrees(math.atan(0.5)), abs=1e-6)
    assert off > 22.5  # pinhole stretches the edges relative to a linear mapping


@pytest.mark.parametrize("w,fov", [(0, 90), (160, 0), (160, 180), (-5, 90)])
def test_bearing_rejects_bad_inputs(w, fov):
    with pytest.raises(ValueError):
        bearing_from_pixel(10, w, 0, fov)


def test_local_roundtrip():
    e, n = to_local(46.2, 9.1, 46.0, 9.0)
    lat, lon = from_local(e, n, 46.0, 9.0)
    assert (lat, lon) == pytest.approx((46.2, 9.1), abs=1e-9)


def test_distance_known_value():
    # one degree of latitude is about 111.2 km
    assert distance_m(46.0, 9.0, 47.0, 9.0) == pytest.approx(111_195, rel=1e-3)
    assert distance_m(46.0, 9.0, 46.0, 9.0) == 0.0


def test_intersect_perpendicular_rays():
    # camera 1 at origin looking north, camera 2 at (1000, 1000) looking west -> cross at (0, 1000)
    hit = intersect_rays((0, 0), 0.0, (1000, 1000), 270.0)
    assert hit == pytest.approx((0.0, 1000.0), abs=1e-6)


def test_intersect_parallel_and_behind_return_none():
    assert intersect_rays((0, 0), 10.0, (500, 0), 10.0) is None
    assert intersect_rays((0, 0), 0.0, (1000, -1000), 270.0) is None  # crossing is behind camera 1
    assert intersect_rays((0, 0), 0.0, (1000, 1000), 90.0) is None  # behind camera 2


def test_intersect_small_angle_is_rejected_by_default():
    assert intersect_rays((0, 0), 0.0, (1000, 0), 358.0) is None
    assert intersect_rays((0, 0), 0.0, (1000, 0), 358.0, min_angle_deg=0.5) is not None


def test_triangulate_recovers_exact_fire_without_noise():
    fire = from_local(2500.0, 8000.0, 46.02, 8.91)
    cams = []
    for cam in (CAM_A, CAM_B):
        e, n = to_local(*fire, cam.lat, cam.lon)
        true_b = math.degrees(math.atan2(e, n)) % 360
        off = (bearing_from_pixel(50, 160, 0.0, cam.hfov_deg) + 180) % 360 - 180
        cams.append(Camera(cam.name, cam.lat, cam.lon, (true_b - off) % 360, cam.hfov_deg))
    est = triangulate(cams[0], 50, cams[1], 50, 160)
    assert est is not None
    assert distance_m(*est, *fire) < 20.0  # flat-earth approximation, ~5 m at 8 km range


def test_triangulate_degenerate_returns_none():
    cam = Camera("c", 46.0, 9.0, 0.0, 90.0)
    assert triangulate(cam, 80, cam, 80, 160) is None


def test_simulation_error_grows_with_noise_and_is_reproducible():
    a = simulate(1.0, 0.1, n=300, seed=1)
    b = simulate(8.0, 1.0, n=300, seed=1)
    assert a["median_m"] < b["median_m"]
    assert a == simulate(1.0, 0.1, n=300, seed=1)
    assert a["baseline_km"] == pytest.approx(distance_m(CAM_A.lat, CAM_A.lon, CAM_B.lat, CAM_B.lon) / 1000)


def test_column_sigma_deg_is_monotonic():
    assert column_sigma_deg(0.0) == 0.0
    assert column_sigma_deg(2.0) < column_sigma_deg(4.0)
