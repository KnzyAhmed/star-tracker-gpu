// Results tab: STORED results from the Day 5 campaign and Day 6 benchmark (not computed live).
// Every number shown is read from the JSON files the experiments wrote.
"use strict";

(function () {
  let done = false;
  const pct = (k, n, dp = 2) => (100 * k / n).toFixed(dp);
  const ci = (lo, hi, dp = 2) => `[${(100 * lo).toFixed(dp)}, ${(100 * hi).toFixed(dp)}]`;

  function sweepTable(title, unit, s) {
    const rows = s.level.map((lv, i) =>
      `<tr><td>${lv}</td><td>${s.n[i].toLocaleString()}</td><td>${pct(s.correct[i], s.n[i])} %</td>` +
      `<td>${ci(s.correct_lo[i], s.correct_hi[i])}</td><td>${s.wrong[i]}</td><td>${s.no_solution[i].toLocaleString()}</td>` +
      `<td>${s.mean_detected[i].toFixed(1)}</td></tr>`).join("");
    return `<div class="panel res-table"><h2>${title}</h2><table class="cmp">
      <tr><th>${unit}</th><th>trials</th><th>correct</th><th>95 % CI</th><th>wrong</th><th>no solution</th><th>dots / frame</th></tr>
      ${rows}</table></div>`;
  }

  async function render() {
    if (done) return;
    done = true;
    const el = document.getElementById("results");
    try {
      const [s, cpu, gpu] = await Promise.all(["results/day5/summary.json", "results/day6/cpu_m1pro.json", "results/day6/gpu_colab_t4.json"]
        .map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); })));
      const b = s.baseline;
      // the sweeps' noise-5 / 0-hot-pixel points ARE baseline trials (same skies), so count them once
      const extra = (sw, base) => sw.level.reduce((acc, lv, i) => lv === base ? acc : { n: acc.n + sw.n[i], w: acc.w + sw.wrong[i] }, { n: 0, w: 0 });
      const xn = extra(s.read_noise, 5), xh = extra(s.hot_pixels, 0);
      const totalTrials = b.n + xn.n + xh.n, wrongAll = b.wrong + xn.w + xh.w;
      const thr = Object.fromEntries(cpu.throughput.map((r) => [`${r.workers}-${r.full}`, r.ms_per_frame]));
      const n = cpu.host.cores, g = gpu.gpu.ms_per_frame_median, gpuRD = g.pack + g.gpu_wall;
      el.innerHTML = `
        <div class="tiles">
          <div class="tile"><div class="num">${b.correct.toLocaleString()} / ${b.n.toLocaleString()}</div>
            <div class="lbl">random skies solved (≥ ${(100 * b.correct_ci[0]).toFixed(3)} % at 95 % confidence)</div></div>
          <div class="tile"><div class="num">${wrongAll} wrong</div><div class="lbl">in all ${totalTrials.toLocaleString()} trials, every noise and fake-star level: failures are always "no solution"</div></div>
          <div class="tile"><div class="num">${b.pointing_pct.p50.toFixed(1)}″</div><div class="lbl">median pointing error (p99 ${b.pointing_pct.p99.toFixed(1)}″; roll ${b.roll_pct.p50.toFixed(1)}″)</div></div>
          <div class="tile"><div class="num">${(thr[`${n}-false`] / gpuRD).toFixed(1)}×</div>
            <div class="lbl">Tesla T4 vs all ${n} cores of an ${cpu.host.cpu} on the image stages (${(thr["1-false"] / gpuRD).toFixed(0)}× vs one core)</div></div>
        </div>
        <p class="note">These are <b>stored</b> results. The ${totalTrials.toLocaleString()}-frame campaign and the benchmark ran on an NVIDIA Tesla T4 (Google Colab, 70 W)
          with hand-written CUDA kernels; a web page can't run those, so the live tab uses the CPU version of the same code.</p>
        <h2>Where and why it fails</h2>
        <div class="res-grid">
          <figure><img src="results/day5/failure_map.png" alt="Sky map of failures over star density" loading="lazy">
            <figcaption>Failures cluster in sparse sky, away from the Milky Way: fewer stars means a weaker vote.</figcaption></figure>
          <figure><img src="results/day5/success_vs_stars.png" alt="Success rate versus stars in view" loading="lazy">
            <figcaption>At 4× noise, success rises from 0 % (&lt; 10 stars in view) to 100 % (40+).</figcaption></figure>
        </div>
        <h2>Robustness</h2>
        <div class="res-grid">
          <figure><img src="results/day5/read_noise_sweep.png" alt="Success versus read noise" loading="lazy">
            <figcaption>Read-noise sweep, 10,000 skies per level (the same skies at every level).</figcaption></figure>
          <figure><img src="results/day5/hot_pixels_sweep.png" alt="Success versus hot pixels" loading="lazy">
            <figcaption>Fake stars (hot pixels) per frame, 10,000 skies per level.</figcaption></figure>
        </div>
        <div class="res-grid" style="margin-top:16px">
          ${sweepTable("Read-noise sweep", "read noise", s.read_noise)}
          ${sweepTable("Fake-star sweep (read noise 5)", "hot pixels", s.hot_pixels)}
        </div>
        <h2>Accuracy</h2>
        <div class="res-grid"><figure><img src="results/day5/accuracy_hist.png" alt="Pointing and roll error histograms" loading="lazy">
          <figcaption>Pointing and roll errors of the ${b.n.toLocaleString()} baseline trials.</figcaption></figure></div>
        <h2>CPU vs GPU</h2>
        <div class="res-grid"><figure><img src="results/day6/benchmark.png" alt="Benchmark chart" loading="lazy">
          <figcaption>Render + detect: ${thr["1-false"].toFixed(2)} ms on 1 core, ${thr[`${n}-false`].toFixed(2)} ms on ${n} cores,
          ${gpuRD.toFixed(2)} ms on the T4 (transfers included). End to end, identification on the CPU becomes the bottleneck (Amdahl's law).</figcaption></figure></div>
        <p class="small muted">CPU: ${cpu.host.cpu} (${cpu.host.cores_detail || n + " cores"}, ${cpu.host.power_source} power). GPU: ${gpu.host.gpu}.
          Colab host: ${gpu.host.cpu}, ${gpu.host.cores} vCPU.</p>`;
    } catch (err) {
      el.innerHTML = `<p class="badge wrong">Could not load results</p><p class="small muted">${String(err)}</p>`;
    }
  }
  window.renderResults = render;
})();
