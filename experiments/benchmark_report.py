"""Day 6: merge results/benchmark/*.json (CPU run on the Mac, GPU run on Colab) into a table + figure.

    python -m experiments.benchmark_report        # -> docs/results/day6/benchmark.md, benchmark.png

All numbers are MEASURED medians in ms per frame, except one row marked "projection".
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

SRC, OUT = Path("results/benchmark"), Path("docs/results/day6")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DARK = "--theme" in sys.argv and "dark" in sys.argv          # python -m experiments.benchmark_report --theme dark
if DARK:                                                      # website theme: transparent figure, light ink
    OUT = OUT / "dark"
    SURFACE, INK, INK2, GRID = "#0b0e24", "#eef0ff", "#b8bce0", "#2a2f58"
    BLUE, ORANGE, AQUA, YELLOW = "#9db4ff", "#ff9a6e", "#5fe3c0", "#f3c969"
SAVE = {"transparent": True} if DARK else {}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load(prefix):
    files = sorted(SRC.glob(f"{prefix}_*.json"))
    if not files:
        raise SystemExit(f"no {SRC}/{prefix}_*.json: run experiments.benchmark --device {prefix} first")
    return json.loads(files[-1].read_text())


def main():
    cpu, gpu = load("cpu"), load("gpu")
    mac = cpu["host"]
    n_mac = mac["cores"]
    thr = {(r["workers"], r["full"]): r["ms_per_frame"] for r in cpu["throughput"]}
    s_mac = {k: v["median_ms"] for k, v in cpu["single_core_stages"].items()}
    s_col = {k: v["median_ms"] for k, v in gpu["single_core_stages"].items()}
    g = gpu["gpu"]["ms_per_frame_median"]
    gpu_rd = g["pack"] + g["gpu_wall"]                         # upload + render + detect + download + sort
    mac_rd_all, mac_full_all = thr[(n_mac, False)], thr[(n_mac, True)]
    # projection: GPU render+detect + identify spread over the Mac's cores (identify scales like the full pipeline)
    ident_all = s_mac["identify"] * mac_full_all / thr[(1, True)]
    proj = g["prepare"] + gpu_rd + ident_all

    mac_name = f"{mac['cpu']} ({mac.get('cores_detail', str(n_mac) + ' cores')}, {mac.get('power_source', '?')} power)"
    col_name = f"{gpu['host']['cpu']}, {gpu['host']['cores']} vCPU (Colab)"
    gpu_name = gpu["host"]["gpu"]
    rows = [  # (configuration, render+detect ms, full pipeline ms, note)
        (f"{mac['cpu']}, 1 core", thr[(1, False)], thr[(1, True)], "measured"),
        (f"{mac['cpu']}, {n_mac} processes", mac_rd_all, mac_full_all, "measured"),
        (f"Colab Xeon 2.0 GHz, 1 core", s_col["render"] + s_col["detect"],
         sum(s_col.values()), "measured (sum of stage medians)"),
        (f"Tesla T4 + Colab 2 vCPU for identify", gpu_rd, g["total"], "measured, transfers included"),
        (f"Tesla T4 + {mac['cpu']} {n_mac} cores for identify", gpu_rd, proj, "PROJECTION (not run)"),
    ]
    L = ["# Day 6 benchmark: CPU vs GPU", "",
         f"- CPU: **{mac_name}**, NumPy, one process per core, fastest CPU noise (`rng.normal`).",
         f"- GPU: **{gpu_name}** (name, power limit, memory, driver, max SM clock), CuPy RawKernels, batches of "
         f"{gpu['gpu']['batch']}, 2 warm-up batches, timers stopped after synchronise.",
         f"- Colab host CPU: {col_name}.",
         "- Same skies on both (campaign seeding). Medians, ms per frame (lower is better).", "",
         "| configuration | render + detect | full pipeline | frames/s (full) | note |", "|---|---|---|---|---|"]
    for name, rd, full, note in rows:
        L.append(f"| {name} | {rd:.3f} | {full:.3f} | {1000 / full:,.0f} | {note} |")
    L += ["", "**Speed-ups for the image stages (render + detect, transfers included):**",
          f"- T4 vs {mac['cpu']} 1 core: **{thr[(1, False)] / gpu_rd:.0f}×**",
          f"- T4 vs {mac['cpu']} all {n_mac} cores: **{mac_rd_all / gpu_rd:.1f}×**",
          f"- T4 vs its own Colab CPU (1 core): {(s_col['render'] + s_col['detect']) / gpu_rd:.0f}×", "",
          "**GPU pipeline breakdown (ms/frame):** " + ", ".join(f"{k} {v:.3f}" for k, v in g.items()), "",
          f"Identification is {g['identify_pool'] / g['total'] * 100:.0f} % of the GPU pipeline's time on Colab's 2 vCPUs: "
          "after the GPU removes the image work, the CPU-side identify is the bottleneck (Amdahl's law).",
          f"The projection row assumes identify scales over the Mac's cores like the full CPU pipeline did "
          f"({thr[(1, True)] / mac_full_all:.1f}× on {n_mac} processes); it was not run as one system."]
    OUT.mkdir(parents=True, exist_ok=True)
    if not DARK:
        (OUT / "benchmark.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))

    # ---------------- figure: (a) render+detect per configuration, (b) where the GPU pipeline's time goes
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130, gridspec_kw={"width_ratios": [1.3, 1]})
    labels = [f"{mac['cpu']}\n1 core", f"{mac['cpu']}\n{n_mac} processes", "Tesla T4\n(incl. transfers)"]
    vals = [thr[(1, False)], mac_rd_all, gpu_rd]
    y = range(len(vals))[::-1]
    ax[0].barh(list(y), vals, color=BLUE, height=0.55)
    for yi, v in zip(y, vals):
        ax[0].text(v, yi, f"  {v:.2f} ms", va="center", color=INK, fontsize=9)
    ax[0].set_yticks(list(y), labels)
    ax[0].set_xlim(0, max(vals) * 1.25)
    ax[0].set(xlabel="ms per frame (lower is better)", title="Image stages: render + detect")
    ax[0].grid(axis="y", visible=False)

    parts = [("prepare", g["prepare"], AQUA), ("upload + render", g["pack"] + g["render_kernel"], BLUE),
             ("detect + download", g["detect_and_download"], ORANGE), ("identify (2 vCPU)", g["identify_pool"], YELLOW)]
    left = 0.0
    for name, v, color in parts:
        ax[1].barh([0], [v], left=left, color=color, height=0.5, edgecolor=SURFACE, linewidth=2, label=f"{name} {v:.2f}")
        left += v
    ax[1].set_yticks([])
    ax[1].set(xlabel="ms per frame", title=f"Tesla T4 pipeline, {g['total']:.2f} ms/frame end to end")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark.png", bbox_inches="tight", **SAVE)
    print(f"saved {OUT}/benchmark.md, benchmark.png")


if __name__ == "__main__":
    main()
