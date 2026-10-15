// Live tracker page: sky globe, camera image with overlays, step-by-step results.
// All the star-tracker computation happens in worker.js (the real Python code via Pyodide).
"use strict";

const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAN10 = Math.tan(10 * D2R);
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- vector maths (mirrors tracker/camera.py)
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(a[0], a[1], a[2]); return [a[0] / n, a[1] / n, a[2] / n]; };
const unit = (ra, dec) => [Math.cos(dec * D2R) * Math.cos(ra * D2R), Math.cos(dec * D2R) * Math.sin(ra * D2R), Math.sin(dec * D2R)];

function cameraAxes(ra, dec, roll) {          // same as attitude_from_pointing: rows x, y, z
  const z = unit(ra, dec);
  let east = cross([0, 0, 1], z);
  if (Math.hypot(...east) < 1e-8) east = [1, 0, 0];
  east = norm(east);
  const north = cross(z, east);
  const x0 = east.map((c) => -c), y0 = north.map((c) => -c);
  const c = Math.cos(roll * D2R), s = Math.sin(roll * D2R);
  return [x0.map((v, i) => c * v + s * y0[i]), x0.map((v, i) => -s * v + c * y0[i]), z];
}

// ---------------------------------------------------------------- state
const S = { ra: 83.8, dec: -1.0, roll: 20, noise: 5, hot: 0, seed: 1, sky: null, skyR: null, last: null,
            busy: false, pending: false, pendingRandom: false, reqId: 0 };
const worker = new Worker("worker.js");

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") $("status").textContent = m.text;
  else if (m.type === "ready") {
    S.sky = m.sky;
    S.skyR = m.sky.ra.map((ra, i) => unit(ra, m.sky.dec[i]));
    S.info = m.info;
    $("loading").hidden = true;
    $("live").hidden = false;
    syncInputs();
    drawGlobe();
    solve();
  } else if (m.type === "result") {
    S.busy = false;
    setBusy(false);
    S.last = m.out;
    drawImage(m.out);
    drawOverlay();
    showSteps(m.out);
    if (S.pendingRandom) { S.pendingRandom = false; S.pending = false; lostInSpace(); }
    else if (S.pending) { S.pending = false; solve(); }
  } else if (m.type === "random") {
    Object.assign(S, { ra: m.out.ra, dec: m.out.dec, roll: m.out.roll });
    syncInputs();
    drawGlobe();
    S.busy = false;
    solve();
  } else if (m.type === "error") {
    S.busy = false;
    setBusy(false);
    $("status").textContent = m.text;
    $("answer").innerHTML = `<p class="badge wrong">Error</p><p class="small muted">${escapeHtml(m.text)}</p>`;
  }
};

function setBusy(b) { $("solve").disabled = b; $("lost").textContent = b ? "Solving…" : "Lost in space (random)"; }
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function solve() {
  if (!S.sky) return;
  if (S.busy) { S.pending = true; return; }
  S.busy = true;
  setBusy(true);
  S.seed += 1;                                   // fresh noise every solve
  worker.postMessage({ cmd: "solve", id: ++S.reqId, ra: S.ra, dec: S.dec, roll: S.roll, noise: S.noise, hot: S.hot, seed: S.seed });
}

// ---------------------------------------------------------------- inputs
const FIELDS = [["ra", 1], ["dec", 1], ["roll", 0], ["noise", 1], ["hot", 0]];
function syncInputs() {
  for (const [k, dp] of FIELDS) { $(k).value = S[k]; $(k + "Out").textContent = Number(S[k]).toFixed(dp); }
}
for (const [k, dp] of FIELDS) {
  $(k).addEventListener("input", () => { S[k] = Number($(k).value); $(k + "Out").textContent = S[k].toFixed(dp); drawGlobe(); });
  $(k).addEventListener("change", solve);
}
$("solve").addEventListener("click", solve);
function lostInSpace() {
  if (S.busy) { S.pendingRandom = true; return; }      // queue it: never drop a click silently
  S.busy = true;
  setBusy(true);
  worker.postMessage({ cmd: "random", id: ++S.reqId, seed: Math.floor(Math.random() * 1e9) });
}
$("lost").addEventListener("click", lostInSpace);
for (const id of ["showDots", "showNames", "showTruth"]) $(id).addEventListener("change", drawOverlay);

// ---------------------------------------------------------------- sky globe
const globe = $("globe"), g = globe.getContext("2d");
const ZOOM = 1.9;                                // orthographic zoom: ~±32 deg of sky across the disc

function project(axes, r, R, cx, cy) {
  const bz = dot(r, axes[2]);
  return [cx + R * ZOOM * dot(r, axes[0]), cy + R * ZOOM * dot(r, axes[1]), bz];
}

function drawGlobe() {
  if (!S.sky) return;
  const W = globe.width, cx = W / 2, cy = W / 2, R = W / 2 - 4;
  const axes = cameraAxes(S.ra, S.dec, S.roll);
  g.clearRect(0, 0, W, W);
  g.save();
  g.beginPath(); g.arc(cx, cy, R, 0, 2 * Math.PI); g.fillStyle = "#05070c"; g.fill(); g.clip();

  // RA / Dec grid
  g.strokeStyle = "rgba(120,140,180,0.22)"; g.lineWidth = 1;
  const line = (pts) => { g.beginPath(); let pen = false;
    for (const r of pts) { const [x, y, z] = project(axes, r, R, cx, cy);
      if (z > 0) { pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true; } else pen = false; }
    g.stroke(); };
  for (let d = -75; d <= 75; d += 15) line(Array.from({ length: 181 }, (_, i) => unit(i * 2, d)));
  for (let a = 0; a < 360; a += 15) line(Array.from({ length: 91 }, (_, i) => unit(a, -90 + i * 2)));

  // stars
  const sky = S.sky;
  for (let i = 0; i < S.skyR.length; i++) {
    const [x, y, z] = project(axes, S.skyR[i], R, cx, cy);
    if (z <= 0 || Math.hypot(x - cx, y - cy) > R) continue;
    const rad = Math.max(0.5, 3.4 - 0.55 * sky.mag[i]);
    g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI); g.fillStyle = "#e9eefc"; g.fill();
    if (sky.name[i]) { g.fillStyle = css("--gold"); g.font = "11px system-ui"; g.fillText(sky.name[i], x + 5, y - 4); }
  }

  // camera field of view (the pinhole's square, seen on the sphere)
  g.strokeStyle = css("--accent"); g.lineWidth = 2;
  g.beginPath();
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]];
  for (let k = 0; k < 4; k++) for (let t = 0; t <= 1.0001; t += 0.1) {
    const a = corners[k], b = corners[k + 1];
    const bx = TAN10 * (a[0] + (b[0] - a[0]) * t), by = TAN10 * (a[1] + (b[1] - a[1]) * t);
    const n = Math.hypot(bx, by, 1);
    const x = cx + R * ZOOM * bx / n, y = cy + R * ZOOM * by / n;
    (k === 0 && t === 0) ? g.moveTo(x, y) : g.lineTo(x, y);
  }
  g.stroke();
  g.restore();
  g.fillStyle = css("--ink-2"); g.font = "12px system-ui";
  g.fillText(`north ↑ at roll 0 · east ←`, 10, W - 10);
}

// drag: move the boresight opposite to the drag, in the camera's own axes (works at any roll, at the poles)
let drag = null;
globe.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; globe.setPointerCapture(e.pointerId); globe.classList.add("dragging"); });
globe.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const scale = globe.width / globe.getBoundingClientRect().width;
  const R = globe.width / 2 - 4;
  const dx = (e.clientX - drag.x) * scale / (R * ZOOM), dy = (e.clientY - drag.y) * scale / (R * ZOOM);
  drag = { x: e.clientX, y: e.clientY };
  const [x, y, z] = cameraAxes(S.ra, S.dec, S.roll);
  const z2 = norm([0, 1, 2].map((i) => z[i] - dx * x[i] - dy * y[i]));
  S.dec = Math.asin(Math.max(-1, Math.min(1, z2[2]))) * R2D;
  S.ra = ((Math.atan2(z2[1], z2[0]) * R2D) % 360 + 360) % 360;
  syncInputs();
  drawGlobe();
});
const endDrag = () => { if (drag) { drag = null; globe.classList.remove("dragging"); solve(); } };
globe.addEventListener("pointerup", endDrag);
globe.addEventListener("pointercancel", endDrag);

// ---------------------------------------------------------------- camera image + overlays
const imgC = $("image").getContext("2d"), ov = $("overlay").getContext("2d");
function drawImage(out) {
  const px = out.image, data = imgC.createImageData(1024, 1024);
  for (let i = 0, j = 0; i < px.length; i++, j += 4) { const v = px[i]; data.data[j] = v; data.data[j + 1] = v; data.data[j + 2] = v; data.data[j + 3] = 255; }
  imgC.putImageData(data, 0, 0);
}
function drawOverlay() {
  ov.clearRect(0, 0, 1024, 1024);
  const out = S.last;
  if (!out) return;
  if ($("showTruth").checked) {
    ov.strokeStyle = css("--truth"); ov.lineWidth = 2;
    for (const s of out.stars) { ov.beginPath(); ov.moveTo(s.u, s.v - 9); ov.lineTo(s.u + 9, s.v); ov.lineTo(s.u, s.v + 9); ov.lineTo(s.u - 9, s.v); ov.closePath(); ov.stroke(); }
  }
  if ($("showDots").checked) {
    ov.lineWidth = 3;
    for (const d of out.dots) {
      ov.strokeStyle = d.verified ? css("--accent") : css("--warn");
      ov.beginPath(); ov.arc(d.u, d.v, 15, 0, 2 * Math.PI); ov.stroke();
    }
  }
  if ($("showNames").checked) {
    ov.font = "600 22px system-ui"; ov.fillStyle = css("--gold");
    ov.shadowColor = "rgba(0,0,0,0.9)"; ov.shadowBlur = 4;
    for (const d of out.dots) if (d.verified && d.name && d.mag !== null && d.mag < 3.2) ov.fillText(d.name, d.u + 18, d.v - 14);
    ov.shadowBlur = 0;
  }
}

// ---------------------------------------------------------------- steps + answer
function fmt(x, dp = 2) { return x === null || x === undefined ? "–" : Number(x).toFixed(dp); }
function angDiff(a, b) { return ((a - b + 540) % 360) - 180; }

function showSteps(out) {
  const t = out.timing_ms, total = t.prepare + t.render + t.detect + t.identify;
  const steps = [
    ["Render", t.render, `${out.n_in_view} catalog stars drawn as blurred dots on 1,048,576 pixels, plus noise${S.hot ? ` and ${S.hot} fake stars` : ""}.`],
    ["Detect", t.detect, `${out.n_detected} dots found: 5σ above the background and brightest in their 3×3 neighbourhood; centres to ~0.1 px.`],
    ["Identify", t.identify, `${out.n_pairs.toLocaleString()} dot pairs voted against ${S.info.pairs.toLocaleString()} catalog pairs (±${S.info.eps_arcsec}″); ` +
      `names cross-checked, attitude solved by SVD, ${out.n_verified} dots verified.`],
  ];
  $("steps").innerHTML = steps.map(([n, ms, what], i) =>
    `<li><span class="name">${i + 1}. ${n}</span><span class="ms">${fmt(ms, 1)} ms</span><span class="what">${what}</span></li>`).join("") +
    `<li><span class="name">Total in your browser</span><span class="ms">${fmt(total, 0)} ms</span>` +
    `<span class="what">The project's Python, compiled to WebAssembly, on your CPU. Native Python on one Apple M1 Pro core takes ~14 ms per frame; ` +
    `the Tesla T4 renders + detects one in 0.32 ms.</span></li>`;

  if (out.outcome === "no solution") {
    $("answer").innerHTML = `<p class="badge none">No solution</p>
      <p class="small">Fewer than 4 dots could be verified, so the tracker says it doesn't know instead of guessing.
      A spacecraft would keep using its gyros and try the next frame. Try less noise or fewer fake stars.</p>`;
    return;
  }
  const e = out.estimate, tr = out.truth, er = out.error_arcsec;
  const cls = out.outcome === "correct" ? "correct" : "wrong";
  $("answer").innerHTML = `<p class="badge ${cls}">${out.outcome === "correct" ? "✓ Solved" : "✗ Wrong"}</p>
    <table class="cmp"><tr><th></th><th>truth</th><th>tracker</th><th>difference</th></tr>
      <tr><td>RA</td><td>${fmt(tr.ra, 4)}°</td><td>${fmt(e.ra, 4)}°</td><td>${fmt(angDiff(e.ra, tr.ra) * 3600 * Math.cos(tr.dec * D2R), 1)}″</td></tr>
      <tr><td>Dec</td><td>${fmt(tr.dec, 4)}°</td><td>${fmt(e.dec, 4)}°</td><td>${fmt((e.dec - tr.dec) * 3600, 1)}″</td></tr>
      <tr><td>Roll</td><td>${fmt(tr.roll, 3)}°</td><td>${fmt(e.roll, 3)}°</td><td>${fmt(angDiff(e.roll, tr.roll) * 3600, 1)}″</td></tr></table>
    <p class="big-err">${fmt(er.pointing, 1)}″ <span>pointing error (1 pixel ≈ 71″)</span></p>
    <p class="small muted">Roll error ${fmt(er.roll, 1)}″. Roll is weaker because all stars sit within 14° of the boresight.</p>`;
}

// ---------------------------------------------------------------- tabs
for (const b of document.querySelectorAll(".tabs button")) {
  b.addEventListener("click", () => {
    for (const o of document.querySelectorAll(".tabs button")) o.setAttribute("aria-selected", String(o === b));
    for (const s of document.querySelectorAll(".tab")) s.hidden = s.id !== "tab-" + b.dataset.tab;
    if (b.dataset.tab === "results" && window.renderResults) window.renderResults();
  });
}
