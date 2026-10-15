// Live tracker page: sky globe, camera image with overlays, the solve as a timeline.
// All the star-tracker computation happens in worker.js (the real Python code via Pyodide).
"use strict";

const D2R = Math.PI / 180, R2D = 180 / Math.PI, TAN10 = Math.tan(10 * D2R);
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- background starfield (drawn once)
(function drawSky() {
  const c = $("stars"), ctx = c.getContext("2d");
  const paint = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = innerWidth * dpr; c.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    let s = 20261015;                                               // fixed seed: the same sky every visit
    const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    const n = Math.round(innerWidth * innerHeight / 1800);
    for (let i = 0; i < n; i++) {
      const x = rnd() * innerWidth, y = rnd() * innerHeight, r = rnd();
      const size = r < 0.92 ? 0.6 + rnd() * 0.6 : 1.2 + rnd() * 1.3;
      const a = 0.25 + rnd() * 0.6;
      ctx.fillStyle = r < 0.1 ? `rgba(159,208,255,${a})` : r < 0.2 ? `rgba(74,163,255,${a})` : `rgba(232,240,255,${a})`;
      ctx.beginPath(); ctx.arc(x, y, size, 0, 2 * Math.PI); ctx.fill();
    }
  };
  paint();
  addEventListener("resize", paint);
})();

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
    for (const el of document.querySelectorAll(".needs-ready")) el.hidden = false;
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
    $("answer").innerHTML = `<p class="headline wrong">Error</p><p class="note">${escapeHtml(m.text)}</p>`;
  }
};

function setBusy(b) { $("solve").disabled = b; $("lost").textContent = b ? "Solving…" : "Lost in space"; }
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function solve() {
  if (!S.sky) return;
  if (S.busy) { S.pending = true; return; }
  S.busy = true;
  setBusy(true);
  S.seed += 1;                                   // fresh noise every solve
  worker.postMessage({ cmd: "solve", id: ++S.reqId, ra: S.ra, dec: S.dec, roll: S.roll, noise: S.noise, hot: S.hot, seed: S.seed });
}

function lostInSpace() {
  if (S.busy) { S.pendingRandom = true; return; }      // queue it: never drop a click silently
  S.busy = true;
  setBusy(true);
  worker.postMessage({ cmd: "random", id: ++S.reqId, seed: Math.floor(Math.random() * 1e9) });
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
$("lost").addEventListener("click", lostInSpace);
for (const id of ["showDots", "showNames", "showTruth"]) $(id).addEventListener("change", drawOverlay);

// ---------------------------------------------------------------- sky globe
const globe = $("globe"), g = globe.getContext("2d");
const ZOOM = 1.0;                                // orthographic sphere: the full hemisphere in front of the camera

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
  g.beginPath(); g.arc(cx, cy, R, 0, 2 * Math.PI);
  // the sphere: lit from the upper left, dark towards the limb
  const body = g.createRadialGradient(cx - 0.45 * R, cy - 0.45 * R, R * 0.05, cx - 0.1 * R, cy - 0.1 * R, R * 1.15);
  body.addColorStop(0, "#2b5aa0"); body.addColorStop(0.35, "#132d5c"); body.addColorStop(0.75, "#071431"); body.addColorStop(1, "#02050f");
  g.fillStyle = body; g.fill(); g.clip();

  // RA / Dec grid, fading towards the limb so the curvature reads
  const gridInk = g.createRadialGradient(cx, cy, 0, cx, cy, R);
  gridInk.addColorStop(0, "rgba(159,208,255,0.42)"); gridInk.addColorStop(0.8, "rgba(159,208,255,0.2)"); gridInk.addColorStop(1, "rgba(159,208,255,0.04)");
  g.strokeStyle = gridInk; g.lineWidth = 1;
  const line = (pts) => { g.beginPath(); let pen = false;
    for (const r of pts) { const [x, y, z] = project(axes, r, R, cx, cy);
      if (z > 0) { pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true; } else pen = false; }
    g.stroke(); };
  for (let d = -75; d <= 75; d += 15) line(Array.from({ length: 181 }, (_, i) => unit(i * 2, d)));
  for (let a = 0; a < 360; a += 15) line(Array.from({ length: 91 }, (_, i) => unit(a, -90 + i * 2)));

  // stars on the sphere: smaller and fainter near the limb
  const sky = S.sky;
  g.font = "500 12px Inter, system-ui";
  for (let i = 0; i < S.skyR.length; i++) {
    const [x, y, z] = project(axes, S.skyR[i], R, cx, cy);
    if (z <= 0) continue;
    const depth = 0.45 + 0.55 * z;
    const rad = Math.max(0.5, 3.6 - 0.6 * sky.mag[i]) * depth;
    g.globalAlpha = 0.35 + 0.65 * z;
    if (rad > 1.8) { g.shadowColor = "rgba(232,240,255,0.9)"; g.shadowBlur = 8; } else g.shadowBlur = 0;
    g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI); g.fillStyle = "#e8f0ff"; g.fill();
    g.shadowBlur = 0;
    if (sky.name[i] && z > 0.35) { g.fillStyle = css("--gold"); g.fillText(sky.name[i], x + 6, y - 5); }
  }
  g.globalAlpha = 1;

  // the camera's field of view: the pinhole's square, drawn on the sphere
  g.strokeStyle = css("--accent"); g.lineWidth = 2;
  g.shadowColor = "rgba(74,163,255,1)"; g.shadowBlur = 14;
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
  g.shadowBlur = 0;
  g.strokeStyle = "rgba(159,208,255,0.7)"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx - 7, cy); g.lineTo(cx + 7, cy); g.moveTo(cx, cy - 7); g.lineTo(cx, cy + 7); g.stroke();

  // limb darkening and a soft specular highlight: what makes it read as a ball
  const limb = g.createRadialGradient(cx, cy, R * 0.55, cx, cy, R);
  limb.addColorStop(0, "rgba(2,5,15,0)"); limb.addColorStop(1, "rgba(2,5,15,0.85)");
  g.fillStyle = limb; g.fillRect(0, 0, W, W);
  const spec = g.createRadialGradient(cx - 0.5 * R, cy - 0.5 * R, 0, cx - 0.5 * R, cy - 0.5 * R, R * 0.7);
  spec.addColorStop(0, "rgba(200,225,255,0.22)"); spec.addColorStop(1, "rgba(200,225,255,0)");
  g.fillStyle = spec; g.fillRect(0, 0, W, W);
  g.restore();
}

// drag: move the boresight opposite to the drag, in the camera's own axes (works at any roll, at the poles)
let drag = null;
globe.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; globe.setPointerCapture(e.pointerId); globe.classList.add("dragging"); });
globe.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const scale = globe.width / globe.getBoundingClientRect().width;
  const R = globe.width / 2 - 2;
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
    ov.lineWidth = 2.5;
    for (const d of out.dots) {
      ov.strokeStyle = d.verified ? css("--accent") : css("--warn");
      ov.beginPath(); ov.arc(d.u, d.v, 15, 0, 2 * Math.PI); ov.stroke();
    }
  }
  if ($("showNames").checked) {
    ov.font = "500 21px Inter, system-ui"; ov.fillStyle = css("--gold");
    ov.shadowColor = "rgba(0,0,0,0.95)"; ov.shadowBlur = 5;
    for (const d of out.dots) if (d.verified && d.name && d.mag !== null && d.mag < 3.2) ov.fillText(d.name, d.u + 18, d.v - 14);
    ov.shadowBlur = 0;
  }
}

// ---------------------------------------------------------------- timeline + answer
function fmt(x, dp = 2) { return x === null || x === undefined ? "–" : Number(x).toFixed(dp); }
function angDiff(a, b) { return ((a - b + 540) % 360) - 180; }

function showSteps(out) {
  const t = out.timing_ms, total = t.prepare + t.render + t.detect + t.identify;
  const solved = out.outcome !== "no solution";
  const steps = [
    ["Render", t.render, `${out.n_in_view} catalog stars drawn as blurred dots on 1,048,576 pixels, plus noise${S.hot ? ` and ${S.hot} fake stars` : ""}.`],
    ["Detect", t.detect, `${out.n_detected} dots found: 5σ above the background and brightest in their 3×3 neighbourhood; centres to ~0.1 px.`],
    ["Identify", t.identify, `${out.n_pairs.toLocaleString()} dot pairs voted against ${S.info.pairs.toLocaleString()} catalog pairs (±${S.info.eps_arcsec}″); ` +
      `names cross-checked, attitude solved by SVD, ${out.n_verified} dots verified.`],
    [solved ? "Solved" : "No solution", total, solved
      ? `${fmt(total, 0)} ms in your browser (Python as WebAssembly). Native Python on one laptop core: ~14 ms; the Tesla T4 renders + detects a frame in 0.32 ms.`
      : `Fewer than 4 dots could be verified, so the tracker says it doesn't know instead of guessing.`],
  ];
  $("steps").innerHTML = steps.map(([n, ms, what], i) =>
    `<li class="${i < 3 || solved ? "done" : ""}"><span class="name">${n}</span><span class="ms">${fmt(ms, i === 3 ? 0 : 1)} ms</span><span class="what">${what}</span></li>`).join("");

  const v = $("verdict");
  v.hidden = false;
  v.className = "verdict " + (out.outcome === "correct" ? "correct" : out.outcome === "wrong" ? "wrong" : "none");
  v.textContent = out.outcome === "correct" ? "✓ solved" : out.outcome === "wrong" ? "✗ wrong" : "no solution";

  if (!solved) {
    $("answer").innerHTML = `<p class="headline none">No solution<small>the honest answer</small></p>
      <p class="note">A spacecraft would keep using its gyros and try the next frame. Try less noise or fewer fake stars.</p>`;
    return;
  }
  const e = out.estimate, tr = out.truth, er = out.error_arcsec;
  $("answer").innerHTML = `<p class="headline ${out.outcome === "correct" ? "" : "wrong"}">${fmt(er.pointing, 1)}″<small>pointing error · 1 pixel ≈ 71″</small></p>
    <table class="cmp"><tr><th></th><th>truth</th><th>tracker</th><th>difference</th></tr>
      <tr><td>Right ascension</td><td>${fmt(tr.ra, 4)}°</td><td>${fmt(e.ra, 4)}°</td><td>${fmt(angDiff(e.ra, tr.ra) * 3600 * Math.cos(tr.dec * D2R), 1)}″</td></tr>
      <tr><td>Declination</td><td>${fmt(tr.dec, 4)}°</td><td>${fmt(e.dec, 4)}°</td><td>${fmt((e.dec - tr.dec) * 3600, 1)}″</td></tr>
      <tr><td>Roll</td><td>${fmt(tr.roll, 3)}°</td><td>${fmt(e.roll, 3)}°</td><td>${fmt(angDiff(e.roll, tr.roll) * 3600, 1)}″</td></tr></table>
    <p class="note">Roll error ${fmt(er.roll, 1)}″. Roll is the weaker axis because every star sits within 14° of the boresight, so a twist moves them very little.</p>`;
}

// ---------------------------------------------------------------- tabs
for (const b of document.querySelectorAll(".tabs button")) {
  b.addEventListener("click", () => {
    for (const o of document.querySelectorAll(".tabs button")) o.setAttribute("aria-selected", String(o === b));
    for (const s of document.querySelectorAll(".tab")) s.hidden = s.id !== "tab-" + b.dataset.tab;
    if (b.dataset.tab === "results" && window.renderResults) window.renderResults();
    window.scrollTo({ top: 0 });
  });
}
