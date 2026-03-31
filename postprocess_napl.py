#!/usr/bin/env python3
"""
Post-processing & animation of SPH three-phase seepage snapshots
================================================================

Reads the HDF5 snapshot file produced by sph_seepage_napl.py and generates:

  1. anim_profiles.mp4 / .gif
     Vertical saturation profiles (Sw, Sn, Sa vs depth) at x = 5 m
     for every snapshot, with physical time label.

  2. anim_field.mp4 / .gif
     2D saturation field at every snapshot.  Each phase is a separate
     colour layer rendered with transparency proportional to its
     saturation, so overlapping phases blend naturally:
       - Water  →  blue   (alpha = Sw)
       - NAPL   →  red    (alpha = Sn / Sn_max for visibility)
       - Air    →  green   (alpha = Sa)

  3. Static summary figures (final-state profiles, mass balance, etc.)

Usage
-----
    python postprocess_napl.py [snapshot_file.h5]

If no argument is given, looks for sph_napl_snapshots.h5 in the current
directory or in /mnt/user-data/outputs/.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize

# ======================================================================
# 0.  LOCATE SNAPSHOT FILE
# ======================================================================

def find_snapshot_file(argv):
    """Find snapshot HDF5 file or NPZ directory from command line or defaults."""
    candidates = []
    if len(argv) > 1:
        candidates.append(argv[1])
    candidates += [
        "sph_napl_snapshots.h5",
        os.path.join("outputs", "sph_napl_snapshots.h5"),
        os.path.join("../data_sph/omp", "sph_napl_snapshots.h5"),
        # NPZ directory fallback
        "snapshots",
        os.path.join("outputs", "snapshots"),
        os.path.join("../data_sph/omp", "snapshots"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    print("ERROR: snapshot file/directory not found.  Searched:")
    for c in candidates:
        print(f"  {c}")
    sys.exit(1)

snap_source = find_snapshot_file(sys.argv)
# Put output in the parent of snapshots/ dir, or same dir as .h5 file
if os.path.isdir(snap_source):
    out_dir = os.path.dirname(os.path.abspath(snap_source))
else:
    out_dir = os.path.dirname(os.path.abspath(snap_source))
print(f"Snapshot source : {snap_source}")
print(f"Output dir      : {out_dir}")

# ======================================================================
# 1.  LOAD ALL SNAPSHOTS
# ======================================================================

print("Loading snapshots ...", flush=True)

snapshots = []

if os.path.isfile(snap_source) and snap_source.endswith(".h5"):
    # ── HDF5 format: single file, multiple groups ──
    import h5py
    with h5py.File(snap_source, "r") as f:
        group_names = sorted([k for k in f.keys() if k.startswith("step_")])
        print(f"  Found {len(group_names)} snapshot group(s) in HDF5")
        for gname in group_names:
            grp = f[gname]
            d = {}
            for attr in ["step", "time_s", "Nx", "Ny", "Lx", "Ly", "H_u", "H_d"]:
                if attr in grp.attrs:
                    d[attr] = float(grp.attrs[attr]) if attr != "step" else int(grp.attrs[attr])
            for key in ["x", "y", "h", "H", "Sw", "Sn", "kw", "kn",
                         "dhdt", "dSndt", "qx", "qy", "ptype", "is_source"]:
                if key in grp:
                    d[key] = grp[key][:]
            snapshots.append(d)

elif os.path.isdir(snap_source):
    # ── NPZ format: directory of individual files ──
    import glob
    npz_files = sorted(glob.glob(os.path.join(snap_source, "step_*.npz")))
    print(f"  Found {len(npz_files)} NPZ snapshot file(s)")
    for path in npz_files:
        raw = np.load(path, allow_pickle=True)
        d = {}
        for key in raw.files:
            val = raw[key]
            # Scalars stored as 0-d arrays
            if val.ndim == 0:
                d[key] = val.item()
            else:
                d[key] = val
        # Ensure required keys have correct types
        if "step" in d:   d["step"]   = int(d["step"])
        if "time_s" in d: d["time_s"] = float(d["time_s"])
        for k in ["Nx", "Ny"]:
            if k in d: d[k] = int(d[k])
        for k in ["Lx", "Ly", "H_u", "H_d"]:
            if k in d: d[k] = float(d[k])
        snapshots.append(d)
else:
    print(f"ERROR: {snap_source} is neither an HDF5 file nor a directory of NPZ files.")
    sys.exit(1)

if not snapshots:
    print("ERROR: no snapshots loaded.")
    sys.exit(1)

snapshots.sort(key=lambda s: s["step"])
n_snap = len(snapshots)
print(f"  Loaded {n_snap} snapshots, steps {snapshots[0]['step']} → {snapshots[-1]['step']}")
print(f"  Physical time: {snapshots[0]['time_s']:.2e} → {snapshots[-1]['time_s']:.2e} s")

# Grid info from first snapshot
Nx = snapshots[0]["Nx"]
Ny = snapshots[0]["Ny"]
Lx = snapshots[0]["Lx"]
Ly = snapshots[0]["Ly"]
H_u = snapshots[0]["H_u"]
H_d = snapshots[0]["H_d"]
xp = snapshots[0]["x"]
yp = snapshots[0]["y"]
X2d = xp.reshape(Nx, Ny)
Y2d = yp.reshape(Nx, Ny)

# Source region (detect from is_source mask)
is_src = snapshots[0].get("is_source", np.zeros(Nx*Ny, dtype=np.int8))
src_idx = np.where(is_src > 0)[0]
if len(src_idx) > 0:
    SRC_X0, SRC_X1 = xp[src_idx].min(), xp[src_idx].max()
    SRC_Y0, SRC_Y1 = yp[src_idx].min(), yp[src_idx].max()
    # Expand to cell edges
    dx = Lx / (Nx - 1)
    SRC_X0 -= 0.5 * dx; SRC_X1 += 0.5 * dx
    SRC_Y0 -= 0.5 * dx; SRC_Y1 += 0.5 * dx
else:
    SRC_X0 = SRC_X1 = SRC_Y0 = SRC_Y1 = 0

# Column index closest to x = 5 m (domain centre)
ix_mid = Nx // 2

# Determine time units for labels
t_final = snapshots[-1]["time_s"]
if t_final > 7200:
    t_unit, t_label = 3600.0, "h"
elif t_final > 120:
    t_unit, t_label = 60.0, "min"
else:
    t_unit, t_label = 1.0, "s"


def fmt_time(t_s):
    """Format physical time with appropriate unit."""
    return f"{t_s / t_unit:.2f} {t_label}"


# ======================================================================
# 2.  HELPER: extract 2D fields from a snapshot
# ======================================================================

def get_fields(snap):
    """Return (Sw_2d, Sn_2d, Sa_2d, h_2d) from a snapshot dict."""
    Sw = snap["Sw"].reshape(Nx, Ny)
    Sn = snap["Sn"].reshape(Nx, Ny)
    Sa = np.clip(1.0 - Sw - Sn, 0.0, 1.0)
    h  = snap["h"].reshape(Nx, Ny)
    return Sw, Sn, Sa, h


# ======================================================================
# 3.  ANIMATION 1: Vertical saturation profiles at x = x_mid
# ======================================================================

print("\nGenerating vertical profile animation ...", flush=True)

fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(16, 8),
                                          sharey=True, constrained_layout=True)
y_col = Y2d[ix_mid, :]

# Fixed axis limits
for ax in (ax1a, ax1b, ax1c):
    ax.set_ylim(0, Ly)
    ax.grid(True, alpha=0.3)
ax1a.set_xlim(-0.05, 1.05)
ax1b.set_xlim(-0.05, 1.05)
ax1c.set_xlim(-0.05, 1.05)

ax1a.set_xlabel(r"$S_w$ [-]", fontsize=12)
ax1a.set_ylabel("y [m]", fontsize=12)
ax1a.set_title("Water Saturation", fontsize=13)

ax1b.set_xlabel(r"$S_n$ [-]", fontsize=12)
ax1b.set_title("NAPL Saturation", fontsize=13)

ax1c.set_xlabel(r"$S_a$ [-]", fontsize=12)
ax1c.set_title("Air Saturation", fontsize=13)

line_w, = ax1a.plot([], [], "b-", lw=2)
line_n, = ax1b.plot([], [], "r-", lw=2)
line_a, = ax1c.plot([], [], "g-", lw=2)

# Water table marker (horizontal line)
wt_line_a = ax1a.axhline(0, color="cyan", ls="--", lw=1, label="WT")
wt_line_b = ax1b.axhline(0, color="cyan", ls="--", lw=1)
wt_line_c = ax1c.axhline(0, color="cyan", ls="--", lw=1)

# Source region shading
for ax in (ax1a, ax1b, ax1c):
    ax.axhspan(SRC_Y0, SRC_Y1, alpha=0.1, color="red")

time_text1 = fig1.suptitle("", fontsize=14)


def update_profiles(frame_idx):
    snap = snapshots[frame_idx]
    Sw, Sn, Sa, h_2d = get_fields(snap)

    line_w.set_data(Sw[ix_mid, :], y_col)
    line_n.set_data(Sn[ix_mid, :], y_col)
    line_a.set_data(Sa[ix_mid, :], y_col)

    # Find water table elevation at this x
    h_col = h_2d[ix_mid, :]
    crossings = np.where(np.diff(np.sign(h_col)))[0]
    if len(crossings) > 0:
        j = crossings[0]
        f = h_col[j] / (h_col[j] - h_col[j+1]) if h_col[j] != h_col[j+1] else 0.5
        ywt = y_col[j] + f * (y_col[j+1] - y_col[j])
    else:
        ywt = 0
    wt_line_a.set_ydata([ywt, ywt])
    wt_line_b.set_ydata([ywt, ywt])
    wt_line_c.set_ydata([ywt, ywt])

    time_text1.set_text(
        f"Saturation Profiles at x = {X2d[ix_mid,0]:.1f} m  —  "
        f"step {snap['step']},  t = {fmt_time(snap['time_s'])}")
    return line_w, line_n, line_a


anim1 = animation.FuncAnimation(fig1, update_profiles,
                                 frames=n_snap, interval=200, blit=False)

# Save as mp4 if ffmpeg available, else gif
try:
    anim1_path = os.path.join(out_dir, "anim_profiles.mp4")
    anim1.save(anim1_path, writer="ffmpeg", fps=8, dpi=120)
    print(f"  Saved {anim1_path}  ({n_snap} frames)")
except Exception:
    anim1_path = os.path.join(out_dir, "anim_profiles.gif")
    anim1.save(anim1_path, writer="pillow", fps=5, dpi=100)
    print(f"  Saved {anim1_path}  ({n_snap} frames)")
plt.close(fig1)


# ======================================================================
# 4.  ANIMATION 2: 2D RGBA saturation field with transparency blending
# ======================================================================
#
# Each phase is rendered as a colour layer with alpha = saturation:
#   Water  →  blue    RGBA = (0.1, 0.3, 1.0, Sw)
#   NAPL   →  red     RGBA = (1.0, 0.2, 0.0, Sn_scaled)
#   Air    →  green   RGBA = (0.3, 0.8, 0.2, Sa)
#
# Sn is scaled to [0, 1] using the max Sn across ALL snapshots
# so that even small amounts are visible.
#
# Layers are alpha-composited onto a white background.

print("Generating 2D saturation field animation ...", flush=True)

# Find global max Sn for colour scaling
sn_global_max = max(snap["Sn"].max() for snap in snapshots)
sn_global_max = max(sn_global_max, 0.01)   # avoid div by zero
print(f"  Global max Sn = {sn_global_max:.4f}")

# Phase colours (RGB, no alpha)
COL_W = np.array([0.1, 0.3, 1.0])    # blue
COL_N = np.array([1.0, 0.2, 0.0])    # red
COL_A = np.array([0.3, 0.8, 0.2])    # green


def composite_rgba(Sw_2d, Sn_2d, Sa_2d):
    """Alpha-composite three phase layers onto white background.

    Returns (Ny, Nx, 3) RGB array for imshow (transposed for display).
    """
    # Start with white background
    canvas = np.ones((Nx, Ny, 3))

    # Composite in order: air (back), then water, then NAPL (front)
    for col, alpha_2d in [(COL_A, Sa_2d),
                          (COL_W, Sw_2d),
                          (COL_N, np.clip(Sn_2d / sn_global_max, 0, 1))]:
        a = alpha_2d[:, :, np.newaxis]   # (Nx, Ny, 1)
        c = col[np.newaxis, np.newaxis, :]  # (1, 1, 3)
        canvas = canvas * (1.0 - a) + c * a

    # Transpose to (Ny, Nx, 3) for imshow with origin="lower"
    return np.clip(np.transpose(canvas, (1, 0, 2)), 0, 1)


fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, Ly)
ax2.set_aspect("equal")
ax2.set_xlabel("x [m]", fontsize=12)
ax2.set_ylabel("y [m]", fontsize=12)

# Initial frame
Sw0, Sn0, Sa0, h0 = get_fields(snapshots[0])
img2 = ax2.imshow(composite_rgba(Sw0, Sn0, Sa0),
                  extent=[0, Lx, 0, Ly], origin="lower",
                  aspect="equal", interpolation="bilinear")

# Water table contour (will be updated)
wt_contour = [None]

# Source rectangle
src_rect = Rectangle((SRC_X0, SRC_Y0), SRC_X1 - SRC_X0, SRC_Y1 - SRC_Y0,
                      lw=2, ec="white", fc="none", ls="--")
ax2.add_patch(src_rect)

# Legend
ax2.legend(handles=[
    Patch(fc=COL_W, alpha=0.8, label="Water $S_w$"),
    Patch(fc=COL_N, alpha=0.8, label="NAPL $S_n$"),
    Patch(fc=COL_A, alpha=0.8, label="Air $S_a$"),
], loc="upper right", fontsize=10, framealpha=0.8)

title2 = ax2.set_title("", fontsize=13)


def update_field(frame_idx):
    snap = snapshots[frame_idx]
    Sw, Sn, Sa, h_2d = get_fields(snap)

    img2.set_data(composite_rgba(Sw, Sn, Sa))

    # Update water table contour
    if wt_contour[0] is not None:
        # Remove previous contour (works in all matplotlib versions)
        try:
            wt_contour[0].remove()
        except (AttributeError, ValueError):
            # Fallback for older matplotlib
            for c in getattr(wt_contour[0], 'collections', []):
                c.remove()
    wt_contour[0] = ax2.contour(X2d, Y2d, h_2d, levels=[0.0],
                                 colors="white", linewidths=2,
                                 linestyles="--")

    title2.set_text(
        f"Three-Phase Saturation  —  "
        f"step {snap['step']},  t = {fmt_time(snap['time_s'])}")
    return [img2]


anim2 = animation.FuncAnimation(fig2, update_field,
                                 frames=n_snap, interval=200, blit=False)

try:
    anim2_path = os.path.join(out_dir, "anim_field.mp4")
    anim2.save(anim2_path, writer="ffmpeg", fps=8, dpi=120)
    print(f"  Saved {anim2_path}  ({n_snap} frames)")
except Exception:
    anim2_path = os.path.join(out_dir, "anim_field.gif")
    anim2.save(anim2_path, writer="pillow", fps=5, dpi=100)
    print(f"  Saved {anim2_path}  ({n_snap} frames)")
plt.close(fig2)


# ======================================================================
# 5.  STATIC SUMMARY FIGURES
# ======================================================================

print("\nGenerating summary figures ...", flush=True)

# ── Fig S1: Final-state vertical profiles at multiple x positions ──────
fig_s1, (ax_s1a, ax_s1b, ax_s1c) = plt.subplots(1, 3, figsize=(16, 7),
                                                   sharey=True,
                                                   constrained_layout=True)
snap_f = snapshots[-1]
Sw_f, Sn_f, Sa_f, h_f = get_fields(snap_f)

x_positions = [Nx//5, 2*Nx//5, Nx//2, 3*Nx//5, 4*Nx//5]
for ix_val in x_positions:
    label = f"x = {X2d[ix_val, 0]:.1f} m"
    ax_s1a.plot(Sw_f[ix_val, :], y_col, label=label)
    ax_s1b.plot(Sn_f[ix_val, :], y_col, label=label)
    ax_s1c.plot(Sa_f[ix_val, :], y_col, label=label)

ax_s1a.set_xlabel(r"$S_w$"); ax_s1a.set_ylabel("y [m]")
ax_s1a.set_title("Water"); ax_s1a.legend(fontsize=8); ax_s1a.grid(True, alpha=0.3)
ax_s1b.set_xlabel(r"$S_n$")
ax_s1b.set_title("NAPL"); ax_s1b.legend(fontsize=8); ax_s1b.grid(True, alpha=0.3)
ax_s1c.set_xlabel(r"$S_a$")
ax_s1c.set_title("Air"); ax_s1c.legend(fontsize=8); ax_s1c.grid(True, alpha=0.3)
fig_s1.suptitle(f"Final Saturation Profiles  (t = {fmt_time(snap_f['time_s'])})",
                fontsize=14)
fig_s1.savefig(os.path.join(out_dir, "pp_final_profiles.png"),
               dpi=150, bbox_inches="tight")
print("  Saved pp_final_profiles.png")
plt.close(fig_s1)

# ── Fig S2: NAPL mass balance over time ────────────────────────────────
fig_s2, (ax_s2a, ax_s2b) = plt.subplots(1, 2, figsize=(14, 5))

times  = np.array([s["time_s"] for s in snapshots])
# Compute NAPL volumes (need particle volume Vp and porosity)
dx_grid = Lx / (Nx - 1)
Vp = dx_grid * dx_grid
phi_0 = 0.43   # porosity (must match simulation)

napl_total = np.array([np.sum(s["Sn"]) * Vp * phi_0 for s in snapshots])
napl_nosrc = np.array([
    np.sum(s["Sn"][s.get("is_source", np.zeros(Nx*Ny, dtype=np.int8)) < 1]) * Vp * phi_0
    for s in snapshots
])

ax_s2a.plot(times / t_unit, napl_total, "k-", lw=1.5, label="Total (incl. source)")
ax_s2a.plot(times / t_unit, napl_nosrc, "r-", lw=1.5, label="Outside source")
ax_s2a.set_xlabel(f"Time [{t_label}]")
ax_s2a.set_ylabel(r"NAPL volume $\phi S_n V$  [m³/m]")
ax_s2a.set_title("NAPL Mass Balance")
ax_s2a.legend(); ax_s2a.grid(True, alpha=0.3)

if len(times) > 2:
    dt = np.diff(times)
    dm = np.diff(napl_nosrc)
    rate = dm / np.maximum(dt, 1e-30)
    ax_s2b.plot(times[1:] / t_unit, rate, "r-", lw=1)
    ax_s2b.set_xlabel(f"Time [{t_label}]")
    ax_s2b.set_ylabel("d(NAPL vol)/dt  [m³/m/s]")
    ax_s2b.set_title("Source Discharge Rate")
    ax_s2b.grid(True, alpha=0.3)

fig_s2.tight_layout()
fig_s2.savefig(os.path.join(out_dir, "pp_napl_mass_balance.png"),
               dpi=150, bbox_inches="tight")
print("  Saved pp_napl_mass_balance.png")
plt.close(fig_s2)

# ── Fig S3: Water table evolution ──────────────────────────────────────

def find_wt(h_2d):
    """Water table elevation per x-column (h=0 contour)."""
    nx, ny = h_2d.shape
    ywt = np.zeros(nx)
    for ix in range(nx):
        col = h_2d[ix, :]
        crossings = np.where(np.diff(np.sign(col)))[0]
        if len(crossings) > 0:
            j = crossings[0]
            f = col[j] / (col[j] - col[j+1]) if col[j] != col[j+1] else 0.5
            ywt[ix] = Y2d[ix, j] + f * (Y2d[ix, j+1] - Y2d[ix, j])
        else:
            ywt[ix] = Y2d[ix, 0] if col[0] < 0 else Y2d[ix, -1]
    return ywt

fig_s3, (ax_s3a, ax_s3b) = plt.subplots(1, 2, figsize=(14, 5))

cmap_wt = plt.cm.viridis
x_col = X2d[:, 0]
ywt_first = find_wt(snapshots[0]["h"].reshape(Nx, Ny))

for idx, snap in enumerate(snapshots):
    col = cmap_wt(idx / max(n_snap - 1, 1))
    ywt = find_wt(snap["h"].reshape(Nx, Ny))
    label = fmt_time(snap["time_s"]) if idx % max(1, n_snap // 6) == 0 else None
    ax_s3a.plot(x_col, ywt, color=col, lw=1.2, label=label)

ax_s3a.set_xlabel("x [m]"); ax_s3a.set_ylabel("z_WT [m]")
ax_s3a.set_title("Water Table Position Over Time")
ax_s3a.legend(fontsize=8); ax_s3a.grid(True, alpha=0.3)

# WT displacement relative to first snapshot
ywt_last = find_wt(snapshots[-1]["h"].reshape(Nx, Ny))
ax_s3b.plot(x_col, ywt_last - ywt_first, "k-", lw=2)
ax_s3b.axhline(0, color="gray", ls=":", lw=0.8)
ax_s3b.set_xlabel("x [m]"); ax_s3b.set_ylabel("ΔWT [m]")
ax_s3b.set_title("WT Displacement (final − initial)")
ax_s3b.grid(True, alpha=0.3)

fig_s3.tight_layout()
fig_s3.savefig(os.path.join(out_dir, "pp_water_table_evolution.png"),
               dpi=150, bbox_inches="tight")
print("  Saved pp_water_table_evolution.png")
plt.close(fig_s3)

# ── Fig S4: Convergence from snapshot data ─────────────────────────────
fig_s4, (ax_s4a, ax_s4b) = plt.subplots(1, 2, figsize=(14, 5))

steps = np.array([s["step"] for s in snapshots])
l2_dhdt  = np.array([np.sqrt(np.mean(s["dhdt"]**2)) for s in snapshots])
l2_dSndt = np.array([np.sqrt(np.mean(s["dSndt"]**2)) for s in snapshots])
sn_max   = np.array([
    s["Sn"][s.get("is_source", np.zeros(Nx*Ny, dtype=np.int8)) < 1].max()
    if np.any(s.get("is_source", np.zeros(Nx*Ny, dtype=np.int8)) < 1)
    else 0.0
    for s in snapshots
])

ax_s4a.semilogy(times / t_unit, l2_dhdt,  "b-", lw=1, label=r"L2($dh_w/dt$)")
ax_s4a.semilogy(times / t_unit, l2_dSndt, "r-", lw=1, label=r"L2($dS_n/dt$)")
ax_s4a.set_xlabel(f"Time [{t_label}]"); ax_s4a.set_ylabel("Residual")
ax_s4a.set_title("Convergence"); ax_s4a.legend()
ax_s4a.grid(True, which="both", alpha=0.3)

ax_s4b.plot(times / t_unit, sn_max, "r-", lw=1.5)
ax_s4b.set_xlabel(f"Time [{t_label}]")
ax_s4b.set_ylabel(r"max $S_n$ (outside source)")
ax_s4b.set_title("NAPL Front"); ax_s4b.grid(True, alpha=0.3)

fig_s4.tight_layout()
fig_s4.savefig(os.path.join(out_dir, "pp_convergence.png"),
               dpi=150, bbox_inches="tight")
print("  Saved pp_convergence.png")
plt.close(fig_s4)

# ── Fig S5: Sn evolution — vertical slice at x_mid over all snapshots ──
fig_s5, ax_s5 = plt.subplots(figsize=(8, 8), constrained_layout=True)
cmap_evo = plt.cm.inferno

for idx, snap in enumerate(snapshots):
    Sn_2d = snap["Sn"].reshape(Nx, Ny)
    col = cmap_evo(idx / max(n_snap - 1, 1))
    label = (fmt_time(snap["time_s"])
             if idx % max(1, n_snap // 8) == 0 or idx == n_snap - 1
             else None)
    ax_s5.plot(Sn_2d[ix_mid, :], y_col, color=col, lw=1.5, label=label)

ax_s5.axhspan(SRC_Y0, SRC_Y1, alpha=0.1, color="red")
# Initial WT at this x
zwt_mid = H_u + (H_d - H_u) * X2d[ix_mid, 0] / Lx
ax_s5.axhline(zwt_mid, color="blue", ls="--", lw=1, alpha=0.5, label="initial WT")
ax_s5.set_xlabel(r"$S_n$ [-]", fontsize=12)
ax_s5.set_ylabel("y [m]", fontsize=12)
ax_s5.set_title(f"NAPL Saturation Evolution at x = {X2d[ix_mid,0]:.1f} m",
                fontsize=13)
ax_s5.legend(fontsize=8, loc="upper right")
ax_s5.grid(True, alpha=0.3)
fig_s5.savefig(os.path.join(out_dir, "pp_Sn_evolution_vertical.png"),
               dpi=150, bbox_inches="tight")
print("  Saved pp_Sn_evolution_vertical.png")
plt.close(fig_s5)

# ── Fig S6: Final 2D composite (high-res static version) ──────────────
fig_s6, ax_s6 = plt.subplots(figsize=(10, 8), constrained_layout=True)
Sw_f, Sn_f, Sa_f, h_f = get_fields(snapshots[-1])
ax_s6.imshow(composite_rgba(Sw_f, Sn_f, Sa_f),
             extent=[0, Lx, 0, Ly], origin="lower",
             aspect="equal", interpolation="bilinear")
ax_s6.contour(X2d, Y2d, h_f, levels=[0.0], colors="white",
              linewidths=2, linestyles="--")
ax_s6.add_patch(Rectangle((SRC_X0, SRC_Y0), SRC_X1 - SRC_X0,
                            SRC_Y1 - SRC_Y0, lw=2, ec="white",
                            fc="none", ls="--"))
ax_s6.legend(handles=[
    Patch(fc=COL_W, alpha=0.8, label="Water $S_w$"),
    Patch(fc=COL_N, alpha=0.8, label="NAPL $S_n$"),
    Patch(fc=COL_A, alpha=0.8, label="Air $S_a$"),
], loc="upper right", fontsize=10, framealpha=0.8)
ax_s6.set_title(f"Final Three-Phase Saturation  (t = {fmt_time(snap_f['time_s'])})",
                fontsize=13)
ax_s6.set_xlabel("x [m]"); ax_s6.set_ylabel("y [m]")
fig_s6.savefig(os.path.join(out_dir, "pp_final_field.png"),
               dpi=200, bbox_inches="tight")
print("  Saved pp_final_field.png")
plt.close(fig_s6)


# ======================================================================
# DONE
# ======================================================================
print(f"\nPost-processing complete.  {n_snap} snapshots processed.")
print(f"Outputs in: {out_dir}/")
