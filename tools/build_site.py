"""Assemble the static website into _site/ (what GitHub Pages serves).

    python -m tools.build_site            # -> _site/
    python -m http.server -d _site 8000   # try it locally at http://localhost:8000

_site/
  index.html, *.js, *.css     the pages (from web/)
  py/tracker/*.py             the REAL tracker code, run in the browser by Pyodide (no gpu.py)
  py/web/bridge.py            the page's entry points
  py/data/bsc5.csv            the star catalog
  results/                    stored Day 5-6 results (figures, JSON, report)
  manifest.json               which py/ files the browser must load
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"gpu.py", "fetch_catalog.py"}           # CuPy / astroquery: not available in the browser
PAGES = ("index.html", "style.css", "app.js", "worker.js", "results.js")


def build(out=ROOT / "_site"):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "py").mkdir(parents=True)
    for page in PAGES:
        src = ROOT / "web" / page
        if src.exists():
            shutil.copy2(src, out / page)
    files = [f"tracker/{p.name}" for p in sorted((ROOT / "tracker").glob("*.py")) if p.name not in SKIP]
    files += ["web/__init__.py", "web/bridge.py", "data/bsc5.csv"]
    for f in files:
        (out / "py" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / "py" / f)
    shutil.copytree(ROOT / "docs" / "results", out / "results")
    (out / "manifest.json").write_text(json.dumps({"python": files}, indent=1))
    return out


if __name__ == "__main__":
    print(f"built {build()}")
