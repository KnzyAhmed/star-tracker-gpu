"""Error bars on rates. Run:  pytest tests/test_stats.py -q"""
import numpy as np

from tracker.stats import wilson_interval


def test_known_value():
    """Textbook check: 81 successes out of 263 -> Wilson 95 % interval [0.2553, 0.3662]."""
    lo, hi = wilson_interval(81, 263)
    assert abs(lo - 0.2553) < 5e-4 and abs(hi - 0.3662) < 5e-4


def test_all_successes_is_not_a_zero_width_interval():
    """1000/1000 does not prove 100 %. The naive p +- 1.96 sqrt(p(1-p)/n) gives [1, 1]."""
    lo, hi = wilson_interval(1000, 1000)
    assert hi == 1.0 and 0.995 < lo < 0.9965          # ~ 1 - 3.8/n ("rule of three" ~ 3/n)


def test_zero_successes():
    lo, hi = wilson_interval(0, 1000)
    assert lo == 0.0 and 0.0035 < hi < 0.005


def test_interval_contains_the_estimate_and_shrinks_with_n():
    for k, n in [(5, 10), (50, 100), (500, 1000)]:
        lo, hi = wilson_interval(k, n)
        assert lo < k / n < hi
    assert np.diff([np.subtract(*wilson_interval(n // 2, n)[::-1]) for n in (10, 100, 1000)]).max() < 0


def test_coverage_is_about_95_percent():
    """Simulate: the true rate 0.97 must fall inside the interval ~95 % of the time."""
    rng = np.random.default_rng(0)
    k = rng.binomial(200, 0.97, size=20000)
    lo, hi = wilson_interval(k, 200)
    assert 0.93 < np.mean((lo <= 0.97) & (0.97 <= hi)) < 0.97
