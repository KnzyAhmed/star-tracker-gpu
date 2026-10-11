"""Catalog loader. Run:  pytest tests/test_catalog.py -q"""
import numpy as np

from tracker.catalog import load_catalog


def test_bright_star_count():
    cat = load_catalog(5.5)
    assert 2700 < len(cat) < 3000          # 2,847 with the catalog fetched on Day 0
    assert (cat.mag < 5.5).all()


def test_vectors_are_unit_length():
    cat = load_catalog(5.5)
    assert np.allclose(np.linalg.norm(cat.r, axis=1), 1.0)


def test_sirius_is_brightest():
    cat = load_catalog(5.5)
    assert cat.hr[np.argmin(cat.mag)] == 2491
