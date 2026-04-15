#!/usr/bin/env python3
"""
Scaling Test Runner for SPH Benchmark
======================================

Runs benchmark_sph.py for a matrix of grid sizes × thread counts,
collects the timing results, and produces:

  1. scaling_summary.txt  — full results table
  2. scaling_plots.png    — 4-panel figure:
     - ms/step vs grid size (one line per thread count)
     - parallel speedup vs thread count (one line per grid size)
     - parallel efficiency vs thread count
     - time breakdown stacked bars per grid size

Usage
-----
    python run_scaling.py

Configuration: edit the GRID_SIZES and THREAD_COUNTS lists below.
The script detects available cores automatically for the default.

PyCharm: just Run this script.  Set "Working directory" to the folder
containing benchmark_sph.py.  No environment variables needed —
the script sets them per subprocess.
"""

import subprocess
import os
import sys
import platform
import re
import numpy as np

# ======================================================================
# CONFIGURATION  — edit these
# ======================================================================
GRID_SIZES    = [51, 101, 201]       # Nx = Ny values to test
THREAD_COUNTS = [1, 2, 4, 8]        # NUMBA_NUM_THREADS values
N_STEPS       = 5000               # steps per benchmark (enough for timing)
BENCHMARK_SCRIPT = "benchmark_sph.py"
OUT_DIR       = "scaling_results"

# ======================================================================
# AUTO-DETECT
# ======================================================================
n_cores = os.cpu_count() or 4
# Filter thread counts to what's available
THREAD_COUNTS = [t for t in THREAD_COUNTS if t <= n_cores]
if not THREAD_COUNTS:
    THREAD_COUNTS = [1]

print(f"System         : {platform.processor() or platform.machine()}")
print(f"Cores          : {n_cores}")
print(f"Grid sizes     : {GRID_SIZES}")
print(f"Thread counts  : {THREAD_COUNTS}")
print(f"Steps/bench    : {N_STEPS}")
print(f"Output         : {OUT_DIR}/")

os.makedirs(OUT_DIR, exist_ok=True)

# Check benchmark script exists
if not os.path.exists(BENCHMARK_SCRIPT):
    # Try common locations
    for alt in ["benchmark_sph.py",
                os.path.join(os.path.dirname(__file__), "benchmark_sph.py"),
                os.path.join("../outputs", "benchmark_sph.py")]:
        if os.path.exists(alt):
            BENCHMARK_SCRIPT = alt
            break
    else:
        print(f"ERROR: {BENCHMARK_SCRIPT} not found.")
        sys.exit(1)

print(f"Benchmark      : {BENCHMARK_SCRIPT}")


# ======================================================================
# RUN BENCHMARKS
# ======================================================================

def parse_stats(filepath):
    """Read machine-readable footer from benchmark stats file."""
    d = {}
    with open(filepath) as f:
        in_machine = False
        for line in f:
            if "MACHINE_READABLE" in line:
                in_machine = True
                continue
            if in_machine and "=" in line:
                key, val = line.strip().split("=", 1)
                try:
                    d[key] = float(val)
                except ValueError:
                    d[key] = val
    return d


results = []   # list of dicts

total_runs = len(GRID_SIZES) * len(THREAD_COUNTS)
run_idx = 0

for nx in GRID_SIZES:
    for nt in THREAD_COUNTS:
        run_idx += 1
        tag = f"Nx{nx}_Tt{nt}"
        print(f"\n{'─'*60}")
        print(f"  Run {run_idx}/{total_runs}: Nx={nx}, threads={nt}")
        print(f"{'─'*60}")

        env = os.environ.copy()
        env["NUMBA_NUM_THREADS"] = str(nt)
        env["OMP_NUM_THREADS"]   = "1"     # prevent NumPy thread contention
        env["MKL_NUM_THREADS"]   = "1"

        cmd = [
            sys.executable, BENCHMARK_SCRIPT,
            "--nx", str(nx),
            "--nsteps", str(N_STEPS),
            "--outdir", OUT_DIR,
        ]

        try:
            proc = subprocess.run(cmd, env=env, capture_output=True,
                                  text=True, timeout=600)
            print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout)
            if proc.returncode != 0:
                print(f"  STDERR: {proc.stderr[-300:]}")
                print(f"  *** FAILED (exit code {proc.returncode}) ***")
                continue
        except subprocess.TimeoutExpired:
            print(f"  *** TIMEOUT (600 s) ***")
            continue

        # Parse results
        stats_path = os.path.join(OUT_DIR, f"benchmark_{tag}.txt")
        if os.path.exists(stats_path):
            d = parse_stats(stats_path)
            d["tag"] = tag
            results.append(d)
        else:
            print(f"  WARNING: stats file not found: {stats_path}")


if not results:
    print("\nNo successful runs.  Exiting.")
    sys.exit(1)

# ======================================================================
# SUMMARY TABLE
# ======================================================================

# Build table
header = f"{'Nx':>6} {'Threads':>7} {'Particles':>10} {'ms/step':>10} {'T_loop':>8} " \
         f"{'T_sph':>8} {'T_const':>8} {'T_bc':>8} {'L2w_final':>12}"
sep = "─" * len(header)

table_lines = [
    "SPH Three-Phase Benchmark — Scaling Results",
    f"Steps per run: {N_STEPS}",
    f"System: {platform.processor() or platform.machine()}, {n_cores} cores",
    "",
    header,
    sep,
]

for d in sorted(results, key=lambda x: (x.get("NX",0), x.get("THREADS",0))):
    line = (f"{int(d.get('NX',0)):>6} {int(d.get('THREADS',0)):>7} "
            f"{int(d.get('NPART',0)):>10} {d.get('MS_PER_STEP',0):>10.3f} "
            f"{d.get('T_LOOP',0):>8.3f} {d.get('T_SPH_KERNEL',0):>8.3f} "
            f"{d.get('T_CONSTITUTIVE',0):>8.3f} {d.get('T_UPDATE_BC',0):>8.3f} "
            f"{d.get('L2W_FINAL',0):>12.4e}")
    table_lines.append(line)

table_lines.append(sep)

# Compute speedups
table_lines.append("\nParallel Speedup (relative to 1 thread, same grid):")
table_lines.append(f"{'Nx':>6} {'Threads':>7} {'Speedup':>10} {'Efficiency':>10}")
table_lines.append("─" * 40)
for nx in GRID_SIZES:
    base = [d for d in results if int(d.get("NX",0))==nx and int(d.get("THREADS",0))==1]
    if not base:
        continue
    t_base = base[0].get("T_LOOP", 1)
    for d in sorted([d for d in results if int(d.get("NX",0))==nx],
                     key=lambda x: x.get("THREADS",0)):
        nt = int(d.get("THREADS",0))
        t  = d.get("T_LOOP", 1)
        speedup = t_base / max(t, 1e-10)
        eff = speedup / nt * 100
        table_lines.append(f"{nx:>6} {nt:>7} {speedup:>10.2f}x {eff:>9.1f}%")

# Physics consistency check
table_lines.append(f"\nPhysics Consistency (L2w at step {N_STEPS} — should be identical across threads):")
for nx in GRID_SIZES:
    runs = [d for d in results if int(d.get("NX",0))==nx]
    if len(runs) < 2:
        continue
    l2vals = [d.get("L2W_FINAL",0) for d in runs]
    spread = max(l2vals) - min(l2vals)
    table_lines.append(f"  Nx={nx}: L2w range = [{min(l2vals):.6e}, {max(l2vals):.6e}], "
                       f"spread = {spread:.2e}")

summary_path = os.path.join(OUT_DIR, "scaling_summary.txt")
with open(summary_path, "w") as f:
    for line in table_lines:
        f.write(line + "\n")

print(f"\n{'='*60}")
for line in table_lines:
    print(line)
print(f"{'='*60}")
print(f"\nSummary saved: {summary_path}")


# ======================================================================
# SCALING PLOTS
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

colors = plt.cm.tab10(np.linspace(0, 1, max(len(GRID_SIZES), len(THREAD_COUNTS))))

# ── Panel 1: ms/step vs grid size ──────────────────────────────────────
ax = axes[0, 0]
for idx, nt in enumerate(THREAD_COUNTS):
    runs = sorted([d for d in results if int(d.get("THREADS",0))==nt],
                   key=lambda x: x.get("NX",0))
    if not runs: continue
    nxs  = [int(d["NX"]) for d in runs]
    ms   = [d["MS_PER_STEP"] for d in runs]
    ax.plot(nxs, ms, "o-", color=colors[idx], lw=2, ms=8, label=f"{nt} thread(s)")
ax.set_xlabel("Nx (grid side)", fontsize=11)
ax.set_ylabel("ms / step", fontsize=11)
ax.set_title("Cost per Step vs Grid Size", fontsize=12)
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_yscale("log"); ax.set_xscale("log")

# ── Panel 2: Parallel speedup vs thread count ─────────────────────────
ax = axes[0, 1]
for idx, nx in enumerate(GRID_SIZES):
    base_runs = [d for d in results if int(d.get("NX",0))==nx and int(d.get("THREADS",0))==1]
    if not base_runs: continue
    t_base = base_runs[0]["T_LOOP"]
    runs = sorted([d for d in results if int(d.get("NX",0))==nx],
                   key=lambda x: x.get("THREADS",0))
    nts = [int(d["THREADS"]) for d in runs]
    spd = [t_base / max(d["T_LOOP"], 1e-10) for d in runs]
    ax.plot(nts, spd, "o-", color=colors[idx], lw=2, ms=8, label=f"Nx={nx}")
# Ideal line
max_t = max(THREAD_COUNTS)
ax.plot([1, max_t], [1, max_t], "k--", lw=1, alpha=0.5, label="Ideal")
ax.set_xlabel("Thread count", fontsize=11)
ax.set_ylabel("Speedup", fontsize=11)
ax.set_title("Parallel Speedup", fontsize=12)
ax.legend(); ax.grid(True, alpha=0.3)

# ── Panel 3: Parallel efficiency ──────────────────────────────────────
ax = axes[1, 0]
for idx, nx in enumerate(GRID_SIZES):
    base_runs = [d for d in results if int(d.get("NX",0))==nx and int(d.get("THREADS",0))==1]
    if not base_runs: continue
    t_base = base_runs[0]["T_LOOP"]
    runs = sorted([d for d in results if int(d.get("NX",0))==nx],
                   key=lambda x: x.get("THREADS",0))
    nts = [int(d["THREADS"]) for d in runs]
    eff = [t_base / max(d["T_LOOP"], 1e-10) / int(d["THREADS"]) * 100 for d in runs]
    ax.plot(nts, eff, "o-", color=colors[idx], lw=2, ms=8, label=f"Nx={nx}")
ax.axhline(100, color="k", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("Thread count", fontsize=11)
ax.set_ylabel("Efficiency [%]", fontsize=11)
ax.set_title("Parallel Efficiency", fontsize=12)
ax.set_ylim(0, 120)
ax.legend(); ax.grid(True, alpha=0.3)

# ── Panel 4: Time breakdown stacked bars ──────────────────────────────
ax = axes[1, 1]
# One group of bars per (Nx, threads) combination
tags = []
t_const_arr = []; t_sph_arr = []; t_bc_arr = []; t_met_arr = []
for d in sorted(results, key=lambda x: (x.get("NX",0), x.get("THREADS",0))):
    tags.append(f"{int(d['NX'])}×{int(d['NX'])}\n{int(d['THREADS'])}T")
    t_const_arr.append(d.get("T_CONSTITUTIVE", 0))
    t_sph_arr.append(d.get("T_SPH_KERNEL", 0))
    t_bc_arr.append(d.get("T_UPDATE_BC", 0))
    t_met_arr.append(d.get("T_METRICS", 0))

x_pos = np.arange(len(tags))
w = 0.6
b1 = ax.bar(x_pos, t_const_arr, w, label="Constitutive", color="#4e79a7")
b2 = ax.bar(x_pos, t_sph_arr, w, bottom=t_const_arr, label="SPH kernels", color="#e15759")
bot3 = np.array(t_const_arr) + np.array(t_sph_arr)
b3 = ax.bar(x_pos, t_bc_arr, w, bottom=bot3, label="Update+BCs", color="#76b7b2")
bot4 = bot3 + np.array(t_bc_arr)
b4 = ax.bar(x_pos, t_met_arr, w, bottom=bot4, label="Metrics", color="#b07aa1")

ax.set_xticks(x_pos)
ax.set_xticklabels(tags, fontsize=8)
ax.set_ylabel("Time [s]", fontsize=11)
ax.set_title("Time Breakdown per Configuration", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(f"SPH Scaling Benchmark — {N_STEPS} steps per run", fontsize=13, y=1.01)

plot_path = os.path.join(OUT_DIR, "scaling_plots.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plots saved: {plot_path}")
print("\nDone.")
