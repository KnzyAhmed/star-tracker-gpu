"""Day 7: the static-site bundle. Run:  pytest tests/test_build_site.py -q"""
import json
import subprocess
import sys

from tools.build_site import build


def test_bundle_has_pages_code_data_and_results(tmp_path):
    out = build(tmp_path / "_site")
    man = json.loads((out / "manifest.json").read_text())
    assert (out / "index.html").exists()
    assert "tracker/identify.py" in man["python"] and "web/bridge.py" in man["python"]
    assert "data/bsc5.csv" in man["python"]
    assert "tracker/gpu.py" not in man["python"]            # CuPy cannot run in a browser
    assert "tracker/fetch_catalog.py" not in man["python"]  # needs astroquery + network
    for f in man["python"]:
        assert (out / "py" / f).exists(), f
    assert (out / "results" / "day5" / "summary.json").exists()
    assert (out / "results" / "day6" / "benchmark.png").exists()


def test_bundle_runs_on_its_own(tmp_path):
    """Only the files listed in the manifest, nothing from the repo: exactly what the browser gets."""
    out = build(tmp_path / "_site")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from web import bridge; "
            "o = bridge.run_frame(83.8, -1.0, 0.0, seed=1); print(o['outcome'])")
    r = subprocess.run([sys.executable, "-I", "-c", code, str(out / "py")], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "correct"
