// Runs the REAL Python tracker in a background thread, so the page stays responsive.
// Pyodide = CPython compiled to WebAssembly; NumPy and SciPy come with it.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";   // pinned version
importScripts(PYODIDE + "pyodide.js");

let py = null;
let bridge = null;

const toObj = (p) => p.toJs({ dict_converter: Object.fromEntries });

async function init() {
  const t0 = performance.now();
  postMessage({ type: "status", text: "Loading Python (Pyodide 0.27.7)…" });
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", text: "Loading NumPy and SciPy…" });
  await py.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", text: "Loading the tracker code and the star catalog…" });
  const man = await (await fetch("manifest.json")).json();
  for (const f of man.python) {
    const buf = new Uint8Array(await (await fetch("py/" + f)).arrayBuffer());
    const dir = "/home/pyodide/" + f.split("/").slice(0, -1).join("/");
    py.FS.mkdirTree(dir);
    py.FS.writeFile("/home/pyodide/" + f, buf);
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  bridge = py.pyimport("web.bridge");
  postMessage({ type: "status", text: "Building the 266,153-pair catalog table…" });
  const info = toObj(bridge.setup());
  const sky = toObj(bridge.sky_stars());
  postMessage({ type: "ready", info, sky, files: man.python.length, loadMs: performance.now() - t0 });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === "solve") {
      const t0 = performance.now();
      const res = bridge.run_frame(m.ra, m.dec, m.roll, m.noise, m.hot, m.seed);
      const out = toObj(res);
      res.destroy();
      out.wallMs = performance.now() - t0;
      postMessage({ type: "result", id: m.id, out }, [out.image.buffer]);
    } else if (m.cmd === "random") {
      postMessage({ type: "random", id: m.id, out: toObj(bridge.random_pointing(m.seed)) });
    }
  } catch (err) {
    postMessage({ type: "error", text: String(err) });
  }
};

init().catch((err) => postMessage({ type: "error", text: "Failed to start: " + err }));
