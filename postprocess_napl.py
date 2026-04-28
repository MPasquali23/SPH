#!/usr/bin/env python3
"""
SPH Three-Phase Seepage — Animated post-processing from HDF5
=============================================================

Reads sph_napl_snapshots.h5 and produces animations:

  1. anim_slices_rgb.mp4   — XZ + YZ slice scatter at source centre.
                             RGB encoding: R=Sn, G=Sa, B=Sw.
                             Streamlines: water (blue), NAPL (red).
  2. anim_profile_xy.mp4   — Saturation profiles (Sw, Sn, Sa) vs z
                             at (x_src, y_src), one line per phase.

Usage
-----
    python postprocess_napl.py
    python postprocess_napl.py --input my_run.h5 --outdir figs/
    python postprocess_napl.py --skip 5         # use every 5th snapshot
    python postprocess_napl.py --fps 10
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False


# ======================================================================
# CLI
# ======================================================================
parser = argparse.ArgumentParser(description="Animated post-processing of SPH snapshots")
parser.add_argument("--input",  default="sph_napl_snapshots.h5",
                    help="HDF5 snapshot file (default: sph_napl_snapshots.h5)")
parser.add_argument("--outdir", default=".",
                    help="Output directory for animations (default: cwd)")
parser.add_argument("--skip",   type=int, default=1,
                    help="Use every Nth snapshot (default: 1)")
parser.add_argument("--fps",    type=int, default=8,
                    help="Animation frame rate (default: 8)")
parser.add_argument("--dpi",    type=int, default=120,
                    help="Output DPI (default: 120)")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

if not HAS_H5:
    print("NOTE: h5py not available — falling back to NPZ snapshots in 'snapshots/' directory.")

if HAS_H5 and not os.path.exists(args.input):
    print(f"NOTE: HDF5 file not found ({args.input}); trying NPZ snapshots ...")


# ======================================================================
# LOAD SNAPSHOTS  (HDF5 if available, else NPZ directory)
# ======================================================================
def load_from_hdf5(path):
    print(f"Reading {path} ...", flush=True)
    snaps = []
    with h5py.File(path, "r") as f:
        groups = sorted([k for k in f.keys() if k.startswith("step_")])
        if len(groups) == 0:
            raise RuntimeError(f"No step_* groups found in {path}")
        groups = groups[::args.skip]
        g0 = f[groups[0]]
        meta = {k: g0.attrs[k] for k in ("Nx","Ny","Nz","Lx","Ly","Lz")}
        meta["x"] = g0["x"][...]; meta["y"] = g0["y"][...]; meta["z"] = g0["z"][...]
        meta["is_source"] = g0["is_source"][...].astype(bool)
        meta["has_napl_q"] = ("qxn" in g0) and ("qyn" in g0) and ("qzn" in g0)
        for k in groups:
            g = f[k]
            d = {"step": int(g.attrs["step"]), "time": float(g.attrs["time_s"]),
                 "h": g["h"][...], "Sn": g["Sn"][...], "Sw": g["Sw"][...],
                 "qx": g["qx"][...], "qy": g["qy"][...], "qz": g["qz"][...]}
            if meta["has_napl_q"]:
                d["qxn"] = g["qxn"][...]; d["qyn"] = g["qyn"][...]; d["qzn"] = g["qzn"][...]
            snaps.append(d)
    return meta, snaps


def load_from_npz(snap_dir):
    """Read NPZ snapshots in <snap_dir>/step_*.npz ."""
    print(f"Reading NPZ snapshots from {snap_dir}/ ...", flush=True)
    files = sorted([f for f in os.listdir(snap_dir) if f.startswith("step_") and f.endswith(".npz")])
    if not files:
        raise RuntimeError(f"No step_*.npz files in {snap_dir}/")
    files = files[::args.skip]
    snaps = []
    f0 = np.load(os.path.join(snap_dir, files[0]))
    meta = {"Nx": int(f0["Nx"]), "Ny": int(f0["Ny"]), "Nz": int(f0["Nz"]),
            "Lx": float(f0["Lx"]), "Ly": float(f0["Ly"]), "Lz": float(f0["Lz"]),
            "x": f0["x"], "y": f0["y"], "z": f0["z"],
            "is_source": f0["is_source"].astype(bool),
            "has_napl_q": all(k in f0.files for k in ("qxn","qyn","qzn"))}
    for fname in files:
        f = np.load(os.path.join(snap_dir, fname))
        d = {"step": int(f["step"]), "time": float(f["time_s"]),
             "h": f["h"], "Sn": f["Sn"], "Sw": f["Sw"],
             "qx": f["qx"], "qy": f["qy"], "qz": f["qz"]}
        if meta["has_napl_q"]:
            d["qxn"] = f["qxn"]; d["qyn"] = f["qyn"]; d["qzn"] = f["qzn"]
        snaps.append(d)
    return meta, snaps


# Try HDF5 first, fall back to NPZ directory
try:
    if HAS_H5 and os.path.exists(args.input):
        meta, snaps = load_from_hdf5(args.input)
    else:
        snap_dir = os.path.join(os.path.dirname(args.input) or ".", "snapshots")
        meta, snaps = load_from_npz(snap_dir)
except Exception as e:
    print(f"ERROR loading snapshots: {e}")
    sys.exit(1)

Nx = int(meta["Nx"]); Ny = int(meta["Ny"]); Nz = int(meta["Nz"])
Lx = float(meta["Lx"]); Ly = float(meta["Ly"]); Lz = float(meta["Lz"])
x_flat = meta["x"]; y_flat = meta["y"]; z_flat = meta["z"]
is_source_flat = meta["is_source"]
has_napl_q = meta["has_napl_q"]

print(f"  Grid: {Nx} x {Ny} x {Nz}    Domain: {Lx} x {Ly} x {Lz} m")
print(f"  Loaded {len(snaps)} frames.   t = [{snaps[0]['time']:.1f}, {snaps[-1]['time']:.1f}] s")
if not has_napl_q:
    print("  NOTE: NAPL velocity not in snapshots — red NAPL streamlines will be skipped.")


# ======================================================================
# GEOMETRY: reshape, infer source location
# ======================================================================
def to3d(arr):
    return arr.reshape(Nx, Ny, Nz)

X3 = to3d(x_flat); Y3 = to3d(y_flat); Z3 = to3d(z_flat)
src3 = to3d(is_source_flat)

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

src_xs = X3[src3]; src_ys = Y3[src3]; src_zs = Z3[src3]
if len(src_xs) == 0:
    print("WARNING: is_source mask is empty — defaulting slice indices to grid centre")
    SRC_X0 = SRC_X1 = 0.5 * Lx
    SRC_Y0 = SRC_Y1 = 0.5 * Ly
    SRC_Z0 = SRC_Z1 = 0.5 * Lz
    ix_src = Nx // 2; iy_src = Ny // 2
else:
    SRC_X0, SRC_X1 = src_xs.min(), src_xs.max()
    SRC_Y0, SRC_Y1 = src_ys.min(), src_ys.max()
    SRC_Z0, SRC_Z1 = src_zs.min(), src_zs.max()
    # Pad bbox by half a cell so the rectangle visually contains the source markers
    SRC_X0 -= 0.5*dx; SRC_X1 += 0.5*dx
    SRC_Y0 -= 0.5*dy; SRC_Y1 += 0.5*dy
    SRC_Z0 -= 0.5*dz; SRC_Z1 += 0.5*dz
    cx = 0.5 * (src_xs.min() + src_xs.max())
    cy = 0.5 * (src_ys.min() + src_ys.max())
    ix_src = max(0, min(int(round(cx / dx)), Nx - 1))
    iy_src = max(0, min(int(round(cy / dy)), Ny - 1))

print(f"  Source bbox  x=[{SRC_X0:.2f},{SRC_X1:.2f}]  "
      f"y=[{SRC_Y0:.2f},{SRC_Y1:.2f}]  z=[{SRC_Z0:.2f},{SRC_Z1:.2f}]")
print(f"  Slice indices: ix_src={ix_src} (x={X3[ix_src,0,0]:.2f}m), "
      f"iy_src={iy_src} (y={Y3[0,iy_src,0]:.2f}m)")

x_axis = X3[:, 0, 0]
y_axis = Y3[0, :, 0]
z_axis = Z3[0, 0, :]


# ======================================================================
# Helper functions
# ======================================================================
def saturations_from(Sw_flat, Sn_flat):
    """Return (Sw, Sn, Sa) as 3D arrays clipped to [0,1]."""
    Sw3 = to3d(np.clip(Sw_flat, 0.0, 1.0))
    Sn3 = to3d(np.clip(Sn_flat, 0.0, 1.0))
    Sa3 = np.clip(1.0 - Sw3 - Sn3, 0.0, 1.0)
    return Sw3, Sn3, Sa3


def rgb_from(Sw, Sn, Sa):
    """Build RGB array from three saturation arrays (same shape).
    R = Sn (red, NAPL), G = Sa (green, air), B = Sw (blue, water).
    """
    rgb = np.zeros(Sw.shape + (3,), dtype=np.float32)
    rgb[..., 0] = Sn
    rgb[..., 1] = Sa
    rgb[..., 2] = Sw
    return np.clip(rgb, 0.0, 1.0)


def slice_yz(field3, ix):  return field3[ix, :, :]   # (Ny, Nz)
def slice_xz(field3, iy):  return field3[:, iy, :]   # (Nx, Nz)


# ======================================================================
# PLOT 1: animated XZ + YZ slice with RGB scatter and streamlines
# ======================================================================
print("\nBuilding Plot 1 (slice animation with streamlines) ...", flush=True)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

# Build cell-edge coordinates for pcolormesh (one more edge than centres).
# Centres are at i*dx for i=0..N-1; edges go at -0.5*dx + i*dx for i=0..N (i.e. straddling each centre).
def edges_from_axis(centres):
    n = len(centres)
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0]    = centres[0]  - 0.5 * (centres[1] - centres[0])
    edges[-1]   = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges

x_edges = edges_from_axis(x_axis)
y_edges = edges_from_axis(y_axis)
z_edges = edges_from_axis(z_axis)


def setup_axes_plot1(ax_yz, ax_xz):
    ax_yz.set_xlim(0, Ly); ax_yz.set_ylim(0, Lz); ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y [m]"); ax_yz.set_ylabel("z [m]")
    ax_xz.set_xlim(0, Lx); ax_xz.set_ylim(0, Lz); ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x [m]"); ax_xz.set_ylabel("z [m]")


def render_plot1_frame(snap):
    ax1a.clear(); ax1b.clear()
    setup_axes_plot1(ax1a, ax1b)

    Sw3, Sn3, Sa3 = saturations_from(snap["Sw"], snap["Sn"])
    qx3 = to3d(snap["qx"]); qy3 = to3d(snap["qy"]); qz3 = to3d(snap["qz"])

    if "qxn" in snap:
        qxn3 = to3d(snap["qxn"]); qyn3 = to3d(snap["qyn"]); qzn3 = to3d(snap["qzn"])
    else:
        qxn3 = qyn3 = qzn3 = None

    # ── YZ slice at x = x_axis[ix_src] ──
    Sw_yz = slice_yz(Sw3, ix_src); Sn_yz = slice_yz(Sn3, ix_src); Sa_yz = slice_yz(Sa3, ix_src)
    rgb_yz = rgb_from(Sw_yz, Sn_yz, Sa_yz)   # (Ny, Nz, 3)
    # pcolormesh expects (Nz, Ny, 3) when called as pcolormesh(y_edges, z_edges, ...)
    ax1a.pcolormesh(y_edges, z_edges, np.transpose(rgb_yz, (1, 0, 2)), shading="flat")

    qy_yz_w = slice_yz(qy3, ix_src); qz_yz_w = slice_yz(qz3, ix_src)
    if np.max(np.abs(qy_yz_w) + np.abs(qz_yz_w)) > 1e-25:
        ax1a.streamplot(y_axis, z_axis, qy_yz_w.T, qz_yz_w.T,
                         color="blue", linewidth=0.9, density=1.2,
                         arrowsize=1.0, arrowstyle="->")
    if qyn3 is not None:
        qy_yz_n = slice_yz(qyn3, ix_src); qz_yz_n = slice_yz(qzn3, ix_src)
        if np.max(np.abs(qy_yz_n) + np.abs(qz_yz_n)) > 1e-25:
            ax1a.streamplot(y_axis, z_axis, qy_yz_n.T, qz_yz_n.T,
                             color="red", linewidth=0.9, density=1.2,
                             arrowsize=1.0, arrowstyle="->")

    ax1a.add_patch(Rectangle((SRC_Y0, SRC_Z0), SRC_Y1-SRC_Y0, SRC_Z1-SRC_Z0,
                              lw=2, ec="k", fc="none", ls="--"))
    ax1a.set_title(f"YZ slice at x={x_axis[ix_src]:.2f} m")

    # ── XZ slice at y = y_axis[iy_src] ──
    Sw_xz = slice_xz(Sw3, iy_src); Sn_xz = slice_xz(Sn3, iy_src); Sa_xz = slice_xz(Sa3, iy_src)
    rgb_xz = rgb_from(Sw_xz, Sn_xz, Sa_xz)   # (Nx, Nz, 3)
    ax1b.pcolormesh(x_edges, z_edges, np.transpose(rgb_xz, (1, 0, 2)), shading="flat")

    qx_xz_w = slice_xz(qx3, iy_src); qz_xz_w = slice_xz(qz3, iy_src)
    if np.max(np.abs(qx_xz_w) + np.abs(qz_xz_w)) > 1e-25:
        ax1b.streamplot(x_axis, z_axis, qx_xz_w.T, qz_xz_w.T,
                         color="blue", linewidth=0.9, density=1.5,
                         arrowsize=1.0, arrowstyle="->")
    if qxn3 is not None:
        qx_xz_n = slice_xz(qxn3, iy_src); qz_xz_n = slice_xz(qzn3, iy_src)
        if np.max(np.abs(qx_xz_n) + np.abs(qz_xz_n)) > 1e-25:
            ax1b.streamplot(x_axis, z_axis, qx_xz_n.T, qz_xz_n.T,
                             color="red", linewidth=0.9, density=1.5,
                             arrowsize=1.0, arrowstyle="->")

    ax1b.add_patch(Rectangle((SRC_X0, SRC_Z0), SRC_X1-SRC_X0, SRC_Z1-SRC_Z0,
                              lw=2, ec="k", fc="none", ls="--"))
    ax1b.set_title(f"XZ slice at y={y_axis[iy_src]:.2f} m")

    fig1.suptitle(
        f"Saturation slices  (R=Sn, G=Sa, B=Sw)  "
        f"+ water (blue) / NAPL (red) streamlines  "
        f"—  step {snap['step']:>6d},  t = {snap['time']:7.1f} s",
        fontsize=11)


def animate_plot1(frame_idx):
    render_plot1_frame(snaps[frame_idx])

print(f"  Rendering {len(snaps)} frames ...", flush=True)
render_plot1_frame(snaps[0])
anim1 = animation.FuncAnimation(fig1, animate_plot1, frames=len(snaps),
                                  interval=1000//args.fps)

out_path1 = os.path.join(args.outdir, "anim_slices_rgb")
saved = False
for ext, writer_name in [("mp4", "ffmpeg"), ("gif", "pillow")]:
    try:
        full = out_path1 + "." + ext
        anim1.save(full, writer=writer_name, fps=args.fps, dpi=args.dpi)
        print(f"  Saved {full}")
        saved = True
        break
    except Exception as e:
        print(f"  ({writer_name} failed: {e})")
if not saved:
    print("  ERROR: could not save Plot 1 animation")
plt.close(fig1)


# ======================================================================
# PLOT 2: animated saturation profiles vs z at (x_src, y_src)
# ======================================================================
print("\nBuilding Plot 2 (saturation profiles animation) ...", flush=True)

fig2, ax2 = plt.subplots(figsize=(7, 8), constrained_layout=True)


def render_plot2_frame(snap):
    ax2.clear()
    Sw3, Sn3, Sa3 = saturations_from(snap["Sw"], snap["Sn"])
    Sw_col = Sw3[ix_src, iy_src, :]
    Sn_col = Sn3[ix_src, iy_src, :]
    Sa_col = Sa3[ix_src, iy_src, :]
    h_col  = to3d(snap["h"])[ix_src, iy_src, :]

    ax2.plot(Sw_col, z_axis, "b-",  lw=2,    label=r"$S_w$ (water)")
    ax2.plot(Sn_col, z_axis, "r-",  lw=2,    label=r"$S_n$ (NAPL)")
    ax2.plot(Sa_col, z_axis, "g--", lw=1.5,  label=r"$S_a$ (air)")

    # Water-table elevation: linear-interpolate where h crosses 0
    sign_change = np.where(np.diff(np.sign(h_col)))[0]
    if len(sign_change) > 0:
        i0 = sign_change[0]; i1 = i0 + 1
        h0, h1 = h_col[i0], h_col[i1]
        z0, z1 = z_axis[i0], z_axis[i1]
        if abs(h1 - h0) > 1e-30:
            z_wt = z0 - h0 * (z1 - z0) / (h1 - h0)
            ax2.axhline(z_wt, color="cyan", ls=":", lw=1, alpha=0.7,
                         label=f"WT  z={z_wt:.2f} m")

    ax2.set_xlabel("Saturation [-]")
    ax2.set_ylabel("z [m]")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, Lz)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_title(
        f"Saturation profiles at (x={x_axis[ix_src]:.2f}, y={y_axis[iy_src]:.2f}) m"
        f"\nstep {snap['step']:>6d},  t = {snap['time']:7.1f} s",
        fontsize=11)


def animate_plot2(frame_idx):
    render_plot2_frame(snaps[frame_idx])

print(f"  Rendering {len(snaps)} frames ...", flush=True)
render_plot2_frame(snaps[0])
anim2 = animation.FuncAnimation(fig2, animate_plot2, frames=len(snaps),
                                  interval=1000//args.fps)

out_path2 = os.path.join(args.outdir, "anim_profile_xy")
saved = False
for ext, writer_name in [("mp4", "ffmpeg"), ("gif", "pillow")]:
    try:
        full = out_path2 + "." + ext
        anim2.save(full, writer=writer_name, fps=args.fps, dpi=args.dpi)
        print(f"  Saved {full}")
        saved = True
        break
    except Exception as e:
        print(f"  ({writer_name} failed: {e})")
if not saved:
    print("  ERROR: could not save Plot 2 animation")
plt.close(fig2)


print(f"\n{'='*70}")
print("Done.")
print(f"{'='*70}")
print(f"Outputs in: {os.path.abspath(args.outdir)}/")
