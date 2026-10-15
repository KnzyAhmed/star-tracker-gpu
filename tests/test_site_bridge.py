"""Day 7: the Python side of the website (runs in Pyodide in the browser; tested here in CPython).
Run:  pytest tests/test_site_bridge.py -q"""
import json

import numpy as np
import pytest

from web import bridge


@pytest.fixture(scope="module", autouse=True)
def ready():
    bridge.setup()


def test_orion_frame_is_solved_and_described():
    out = bridge.run_frame(83.8, -1.0, 20.0, read_noise=5.0, hot_pixels=0, seed=1)
    assert out["outcome"] == "correct"
    assert out["n_in_view"] >= 40 and out["n_detected"] >= 40 and out["n_verified"] >= 30
    assert out["error_arcsec"]["pointing"] < 10
    assert abs(out["estimate"]["ra"] - 83.8) < 0.01 and abs(out["estimate"]["dec"] + 1.0) < 0.01
    names = {d["name"] for d in out["dots"] if d["verified"]}
    assert {"Betelgeuse", "Rigel"} <= names
    assert set(out["timing_ms"]) == {"prepare", "render", "detect", "identify"}


def test_image_is_a_byte_per_pixel():
    out = bridge.run_frame(83.8, -1.0, 0.0, read_noise=5.0, hot_pixels=0, seed=1)
    img = out["image"]
    assert isinstance(img, (bytes, bytearray)) and len(img) == 1024 * 1024
    assert max(img) > 200 and min(img) == 0


def test_dots_carry_everything_the_page_draws():
    out = bridge.run_frame(101.3, -16.7, 30.0, read_noise=5.0, hot_pixels=5, seed=2)
    assert out["n_detected"] == len(out["dots"])
    for d in out["dots"]:
        assert set(d) >= {"u", "v", "flux", "verified", "votes", "hr", "name", "mag"}
    assert len(out["stars"]) == out["n_in_view"]                  # truth, for the "show catalog" overlay
    json.dumps({k: v for k, v in out.items() if k != "image"})     # everything else is JSON-safe


def test_deterministic_for_a_seed():
    a = bridge.run_frame(10.0, 20.0, 5.0, read_noise=8.0, hot_pixels=3, seed=7)
    b = bridge.run_frame(10.0, 20.0, 5.0, read_noise=8.0, hot_pixels=3, seed=7)
    assert a["image"] == b["image"] and a["dots"] == b["dots"]


def test_no_solution_is_reported_honestly():
    """Read noise 2000 puts the 5-sigma threshold (10,020 counts) above even Rigel's peak (~5,900):
    nothing is detected, so the answer must be 'no solution', not a guess. (At noise 200, Orion's
    brightest stars still get through and it solves correctly.)"""
    out = bridge.run_frame(83.8, -1.0, 0.0, read_noise=2000.0, hot_pixels=0, seed=3)
    assert out["outcome"] == "no solution" and out["estimate"] is None and out["error_arcsec"] is None


def test_random_pointing_and_sky():
    p = bridge.random_pointing(123)
    assert 0 <= p["ra"] < 360 and -90 <= p["dec"] <= 90 and -180 < p["roll"] <= 180
    sky = bridge.sky_stars()
    assert len(sky["ra"]) == len(sky["dec"]) == len(sky["mag"]) == 2847
