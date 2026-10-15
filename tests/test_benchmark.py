"""Day 6: benchmark harness smoke tests (tiny sizes; the real run is experiments/benchmark.py).
Run:  pytest tests/test_benchmark.py -q"""
from experiments.benchmark import STAGES, cpu_stage_times, cpu_throughput, host_info


def test_host_info_names_the_cpu():
    info = host_info()
    assert info["cpu"] and info["cores"] >= 1


def test_cpu_stage_times_are_positive_per_stage():
    t = cpu_stage_times(n_frames=3)
    assert set(t) == set(STAGES)
    assert all(len(v) == 3 and min(v) > 0 for v in t.values())


def test_cpu_throughput_with_two_workers():
    for full in (False, True):
        r = cpu_throughput(n_frames=8, workers=2, full=full)
        assert r["frames"] == 8 and r["ms_per_frame"] > 0 and r["workers"] == 2
