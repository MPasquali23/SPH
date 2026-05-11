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

# Exceptions that can be raised by np.load() and array access on a
# corrupted NPZ file. zipfile.BadZipFile is NOT an OSError on Python
# 3.x, so we have to enumerate it explicitly.
import zipfile
_NPZ_ERRORS = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)
# Same idea for HDF5 — h5py raises OSError for I/O issues, but specific
# build versions can raise RuntimeError or h5py-specific exceptions.
_H5_ERRORS = (OSError, KeyError, ValueError, RuntimeError)


def _step_from_name(name):
    """Extract the integer step from a group name 'step_NNNNNN' or filename
    'step_NNNNNN.npz'.  Robust to any zero-padding width — needed because
    long runs (>= 1e6 steps) overflow the original 6-digit width and mix
    different widths in the same directory.
    Returns -1 on parse failure (so the entry sorts to the front and can
    be inspected separately).
    """
    try:
        base = name.split(".")[0]               # strip extension if any
        return int(base.split("_")[1])
    except (ValueError, IndexError):
        return -1


def format_time(t_seconds):
    """Format a physical time (in seconds) for plot titles.
    Picks the most readable representation based on magnitude:
      t < 1 s        →  "1.23e-04 s"   (preserve precision for early steps)
      t < 60 s       →  "42.3 s"
      t < 3600 s     →  "12m 34s"
      t < 86400 s    →  "3h 12m 34s"
      t >= 86400 s   →  "3d 12h 34m 56s"
    """
    if t_seconds < 1.0:
        return f"{t_seconds:.2e} s"
    if t_seconds < 60.0:
        return f"{t_seconds:.1f} s"

    # Decompose into integer d/h/m/s
    total_s = int(round(t_seconds))
    days,  rem = divmod(total_s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


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
parser.add_argument("--dpi",    type=int, default=200,
                    help="Output DPI for rasterized frames (default: 200). "
                         "Scatter plots with many small markers benefit from "
                         "higher DPI than line plots. 120 gives 1800x840 px "
                         "(HD-ish); 200 gives 3000x1400 (sharper); 250+ for "
                         "publication-grade.")
parser.add_argument("--alpha", type=float, default=0.7,
                    help="Marker transparency (0=invisible, 1=opaque). "
                         "Lower values reveal phase mixing more clearly. Default 0.7")
parser.add_argument("--stream-threshold", type=float, default=0.02,
                    help="Mask streamlines below this fraction of max |v| in each slice. "
                         "Default 0.02 (2%%); set to 0 to disable masking.")
parser.add_argument("--stream-density", type=float, default=1.2,
                    help="Streamline density factor (matplotlib streamplot 'density'). "
                         "Lower values = fewer streamlines = faster. Default 1.2")
parser.add_argument("--stream-grid", type=int, default=30,
                    help="Decimate the velocity field to AT MOST this many points "
                         "per axis when computing streamlines. Streamlines need much "
                         "less resolution than the scatter underneath, so this gives "
                         "a big speedup on fine grids. Set to 0 to disable decimation "
                         "(use full grid). Default 30.")
parser.add_argument("--no-streamlines", action="store_true",
                    help="Skip streamline rendering entirely. Big speedup for "
                         "preview / draft animations.")
parser.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers for frame rendering. "
                         "0 = auto (cpu_count - 1). 1 = serial mode (legacy "
                         "FuncAnimation). >1 = parallel mode (each frame rendered "
                         "in a worker process, then ffmpeg stitches). Default 0.")
parser.add_argument("--quality", choices=["high", "medium", "low"], default="high",
                    help="Video quality preset. 'high' = visually lossless "
                         "(CRF 18 for libx264, ~12 Mb/s for hardware encoders). "
                         "'medium' = good (CRF 23, ~6 Mb/s). 'low' = small files "
                         "(CRF 28, ~2 Mb/s). Default high — recommended for "
                         "scientific visualizations with sharp particle edges.")
parser.add_argument("--pix-fmt", choices=["yuv420p", "yuv444p"], default="yuv420p",
                    help="Video pixel format. yuv420p (default) uses 4:2:0 chroma "
                         "subsampling — universally playable, but discards 75%% of "
                         "color resolution which can muddy fine red/blue/green details "
                         "on a grey background. yuv444p keeps full chroma fidelity "
                         "(sharper colored particles) but is rejected by some players "
                         "(notably old QuickTime / iOS Safari). Use yuv444p for "
                         "scientific archival, yuv420p for sharing.")
args = parser.parse_args()

# Resolve --workers auto setting
import multiprocessing as _mp
if args.workers == 0:
    _N_WORKERS = max(1, _mp.cpu_count() - 1)
else:
    _N_WORKERS = max(1, args.workers)
print(f"Frame rendering: {'parallel' if _N_WORKERS > 1 else 'serial'} "
      f"({_N_WORKERS} worker{'s' if _N_WORKERS > 1 else ''})")

os.makedirs(args.outdir, exist_ok=True)

if not HAS_H5:
    print("NOTE: h5py not available — falling back to NPZ snapshots in 'snapshots/' directory.")

if HAS_H5 and not os.path.exists(args.input):
    print(f"NOTE: HDF5 file not found ({args.input}); trying NPZ snapshots ...")


# ======================================================================
# LOAD SNAPSHOTS  (HDF5 if available, else NPZ directory)
# ======================================================================
def _validate_h5_snapshot(group, expected_npart):
    """Return (ok, reason) for whether an HDF5 snapshot group is fully readable.

    Catches the common rsync / partial-write failure modes:
      - missing required attrs or datasets
      - truncated dataset (wrong shape)
      - I/O errors when actually fetching the data
    Cheap: only reads attrs and shapes, not full data arrays.
    """
    required_attrs = ("step", "time_s", "Nx", "Ny", "Nz", "Lx", "Ly", "Lz")
    required_dsets = ("x", "y", "z", "h", "Sn", "Sw", "qx", "qy", "qz", "is_source")
    try:
        for a in required_attrs:
            if a not in group.attrs:
                return False, f"missing attr '{a}'"
            _ = group.attrs[a]   # also verifies it's actually readable
        for d in required_dsets:
            if d not in group:
                return False, f"missing dataset '{d}'"
            shp = group[d].shape
            if expected_npart is not None and shp[0] != expected_npart:
                return False, f"dataset '{d}' has shape {shp}, expected ({expected_npart},)"
    except _H5_ERRORS as e:
        return False, f"I/O error: {type(e).__name__}: {e}"
    return True, None


def load_from_hdf5(path):
    print(f"Reading {path} ...", flush=True)

    # Step 1: try to open the file at all
    try:
        h5_file = h5py.File(path, "r")
    except (OSError, IOError) as e:
        raise RuntimeError(
            f"Cannot open HDF5 file {path}: {e}\n"
            f"  This typically means the file is incomplete or corrupted.\n"
            f"  Try:\n"
            f"    - Re-running rsync to ensure full transfer\n"
            f"    - 'h5ls {path}' to inspect structure\n"
            f"    - 'h5repack {path} {path}.repacked' to recover salvageable data"
        ) from e

    snaps = []
    skipped = []      # list of (group_name, reason)
    meta = None

    with h5_file as f:
        # Step 2: list groups (cheap; usually works even if some groups are bad)
        try:
            groups = sorted([k for k in f.keys() if k.startswith("step_")],
                             key=_step_from_name)
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Cannot list groups in {path}: {e}") from e

        if len(groups) == 0:
            raise RuntimeError(f"No step_* groups found in {path}")

        # Step 3: find the first VALID group to extract geometry metadata
        for k in groups:
            try:
                g = f[k]
                ok, reason = _validate_h5_snapshot(g, expected_npart=None)
                if not ok:
                    skipped.append((k, reason))
                    continue
                Nx_ = int(g.attrs["Nx"])
                Ny_ = int(g.attrs["Ny"])
                Nz_ = int(g.attrs["Nz"])
                expected_npart = Nx_ * Ny_ * Nz_
                meta = {
                    "Nx": Nx_, "Ny": Ny_, "Nz": Nz_,
                    "Lx": float(g.attrs["Lx"]),
                    "Ly": float(g.attrs["Ly"]),
                    "Lz": float(g.attrs["Lz"]),
                    "x": g["x"][...], "y": g["y"][...], "z": g["z"][...],
                    "is_source": g["is_source"][...].astype(bool),
                    "has_napl_q": ("qxn" in g) and ("qyn" in g) and ("qzn" in g),
                }
                break
            except _H5_ERRORS as e:
                skipped.append((k, f"metadata read failed: {type(e).__name__}: {e}"))
                continue

        if meta is None:
            raise RuntimeError(
                f"All {len(groups)} snapshot groups in {path} are unreadable.\n"
                f"  First failures: {skipped[:3]}"
            )

        expected_npart = meta["Nx"] * meta["Ny"] * meta["Nz"]
        # Apply --skip subsampling
        groups_subset = groups[::args.skip]

        # Snapshots that already failed metadata search are skipped
        already_failed = {k for k, _ in skipped}

        # Step 4: read each snapshot, skipping bad ones
        for k in groups_subset:
            if k in already_failed:
                continue
            try:
                g = f[k]
                ok, reason = _validate_h5_snapshot(g, expected_npart)
                if not ok:
                    skipped.append((k, reason))
                    continue

                # Now fetch the data. This is where I/O errors on truncated
                # datasets actually trigger. Catch and skip.
                d = {"step": int(g.attrs["step"]),
                     "time": float(g.attrs["time_s"]),
                     "h":  g["h"][...],  "Sn": g["Sn"][...], "Sw": g["Sw"][...],
                     "qx": g["qx"][...], "qy": g["qy"][...], "qz": g["qz"][...]}
                if meta["has_napl_q"]:
                    # NAPL velocity may be missing only on this snapshot
                    if "qxn" in g and "qyn" in g and "qzn" in g:
                        d["qxn"] = g["qxn"][...]
                        d["qyn"] = g["qyn"][...]
                        d["qzn"] = g["qzn"][...]
                snaps.append(d)
            except _H5_ERRORS as e:
                skipped.append((k, f"read failed: {type(e).__name__}: {e}"))
                continue

    # Report skipped snapshots
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} unreadable snapshot(s):")
        for k, reason in skipped[:5]:
            print(f"    - {k}: {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
        print(f"  Likely cause: incomplete rsync transfer, partial write, "
              f"or file corruption.")

    if len(snaps) == 0:
        raise RuntimeError(
            f"No usable snapshots in {path} after skipping {len(skipped)} bad ones."
        )

    return meta, snaps


def _validate_npz_keys(loader, expected_npart):
    """Return (ok, reason) for whether an NPZ snapshot has all required keys
    with correct shapes. Doesn't load full data arrays."""
    required = ("step", "time_s", "Nx", "Ny", "Nz", "Lx", "Ly", "Lz",
                "x", "y", "z", "h", "Sn", "Sw", "qx", "qy", "qz", "is_source")
    try:
        for k in required:
            if k not in loader.files:
                return False, f"missing key '{k}'"
        if expected_npart is not None:
            for k in ("h", "Sn", "Sw", "qx", "qy", "qz"):
                # Reading shape requires accessing the array, but NumPy lazy-
                # loads from NPZ, so this is cheap.
                shp = loader[k].shape
                if shp[0] != expected_npart:
                    return False, f"key '{k}' has shape {shp}, expected ({expected_npart},)"
    except _NPZ_ERRORS as e:
        return False, f"I/O error: {type(e).__name__}: {e}"
    return True, None


def load_from_npz(snap_dir):
    """Read NPZ snapshots in <snap_dir>/step_*.npz, skipping corrupt files."""
    print(f"Reading NPZ snapshots from {snap_dir}/ ...", flush=True)
    try:
        files = sorted([fn for fn in os.listdir(snap_dir)
                         if fn.startswith("step_") and fn.endswith(".npz")],
                        key=_step_from_name)
    except (OSError, FileNotFoundError) as e:
        raise RuntimeError(f"Cannot list snapshots in {snap_dir}: {e}") from e

    if not files:
        raise RuntimeError(f"No step_*.npz files in {snap_dir}/")

    skipped = []
    meta = None

    # Find first valid file for metadata
    for fname in files:
        path = os.path.join(snap_dir, fname)
        try:
            f0 = np.load(path)
            ok, reason = _validate_npz_keys(f0, expected_npart=None)
            if not ok:
                skipped.append((fname, reason))
                continue
            Nx_ = int(f0["Nx"]); Ny_ = int(f0["Ny"]); Nz_ = int(f0["Nz"])
            meta = {
                "Nx": Nx_, "Ny": Ny_, "Nz": Nz_,
                "Lx": float(f0["Lx"]), "Ly": float(f0["Ly"]), "Lz": float(f0["Lz"]),
                "x": np.asarray(f0["x"]), "y": np.asarray(f0["y"]),
                "z": np.asarray(f0["z"]),
                "is_source": np.asarray(f0["is_source"]).astype(bool),
                "has_napl_q": all(k in f0.files for k in ("qxn", "qyn", "qzn")),
            }
            break
        except _NPZ_ERRORS as e:
            skipped.append((fname, f"metadata read failed: {type(e).__name__}: {e}"))
            continue

    if meta is None:
        raise RuntimeError(
            f"All {len(files)} NPZ snapshot files in {snap_dir} are unreadable.\n"
            f"  First failures: {skipped[:3]}"
        )

    expected_npart = meta["Nx"] * meta["Ny"] * meta["Nz"]
    files_subset = files[::args.skip]
    snaps = []

    # Files that already failed metadata search are skipped
    already_failed = {fname for fname, _ in skipped}

    for fname in files_subset:
        if fname in already_failed:
            continue
        path = os.path.join(snap_dir, fname)
        try:
            f = np.load(path)
            ok, reason = _validate_npz_keys(f, expected_npart)
            if not ok:
                skipped.append((fname, reason))
                continue
            d = {"step": int(f["step"]), "time": float(f["time_s"]),
                 "h":  np.asarray(f["h"]),
                 "Sn": np.asarray(f["Sn"]),
                 "Sw": np.asarray(f["Sw"]),
                 "qx": np.asarray(f["qx"]),
                 "qy": np.asarray(f["qy"]),
                 "qz": np.asarray(f["qz"])}
            if meta["has_napl_q"] and all(k in f.files for k in ("qxn", "qyn", "qzn")):
                d["qxn"] = np.asarray(f["qxn"])
                d["qyn"] = np.asarray(f["qyn"])
                d["qzn"] = np.asarray(f["qzn"])
            snaps.append(d)
        except _NPZ_ERRORS as e:
            skipped.append((fname, f"read failed: {type(e).__name__}: {e}"))
            continue

    if skipped:
        print(f"  WARNING: skipped {len(skipped)} unreadable NPZ file(s):")
        for fname, reason in skipped[:5]:
            print(f"    - {fname}: {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    if len(snaps) == 0:
        raise RuntimeError(
            f"No usable NPZ snapshots in {snap_dir} after skipping "
            f"{len(skipped)} bad ones."
        )

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
print("\nBuilding Plot 1 (slice animation: particle scatter + streamlines) ...", flush=True)

import shutil
import subprocess
import tempfile

# ── Helpers used in BOTH serial and parallel paths ────────────────────

def mask_low_speed(u, v, threshold_frac):
    """Return (u, v) with cells below threshold_frac * max(|v|) replaced by NaN.
    streamplot skips integration through NaN cells, so faint flow regions are
    not cluttered with spurious streamlines.

    threshold_frac == 0  → no masking (all cells kept)
    """
    if threshold_frac <= 0.0:
        return u, v
    mag = np.sqrt(u*u + v*v)
    mmax = float(mag.max())
    if mmax < 1e-30:
        return u, v
    mask = mag < threshold_frac * mmax
    if not mask.any():
        return u, v
    u_out = np.where(mask, np.nan, u)
    v_out = np.where(mask, np.nan, v)
    return u_out, v_out


def decimate_for_streamplot(axis_a, axis_b, u, v, max_points):
    """Subsample (axis_a, axis_b, u, v) to at most max_points along each axis.
    Streamplot integration cost scales with grid size; the visual quality
    of streamlines barely changes between e.g. 50x50 and 25x25, so this is
    mostly free quality-wise but a big speedup for fine grids.

    max_points <= 0 disables decimation.
    Inputs assumed shaped: axis_a (Na,), axis_b (Nb,), u/v (Na, Nb).
    Returns (axis_a', axis_b', u', v') with at most max_points along each axis.
    """
    if max_points <= 0:
        return axis_a, axis_b, u, v
    Na, Nb = u.shape
    # ceil division: we want Na/step <= max_points, so step >= Na/max_points
    step_a = max(1, -(-Na // max_points))
    step_b = max(1, -(-Nb // max_points))
    if step_a == 1 and step_b == 1:
        return axis_a, axis_b, u, v
    return (axis_a[::step_a], axis_b[::step_b],
            u[::step_a, ::step_b], v[::step_a, ::step_b])


def draw_streamlines(ax, axis_a, axis_b, u, v, color, density,
                      stream_thresh, stream_grid, draw_streamlines_flag):
    """Compose mask + decimate + streamplot. Returns silently if disabled
    or if the field has no significant flow."""
    if not draw_streamlines_flag:
        return
    u_m, v_m = mask_low_speed(u, v, stream_thresh)
    if np.nanmax(np.abs(u_m) + np.abs(v_m)) < 1e-25:
        return
    a_d, b_d, u_d, v_d = decimate_for_streamplot(axis_a, axis_b, u_m, v_m,
                                                   stream_grid)
    ax.streamplot(a_d, b_d, u_d.T, v_d.T,
                   color=color, linewidth=0.9, density=density,
                   arrowsize=1.0, arrowstyle="->")


def setup_axes_plot1(ax_yz, ax_xz, geom):
    """Configure the two slice panels (idempotent)."""
    Lx_, Ly_, Lz_ = geom["Lx"], geom["Ly"], geom["Lz"]
    ax_yz.set_xlim(0, Ly_); ax_yz.set_ylim(0, Lz_); ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y [m]"); ax_yz.set_ylabel("z [m]")
    ax_yz.set_facecolor("#f4f4f4")
    ax_xz.set_xlim(0, Lx_); ax_xz.set_ylim(0, Lz_); ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x [m]"); ax_xz.set_ylabel("z [m]")
    ax_xz.set_facecolor("#f4f4f4")


def marker_size_for(ax, dx_axis, fill_fraction=0.65):
    """Compute scatter `s` (points^2) such that a square marker spans
    fill_fraction * dx_axis data units along the x-axis.

    fill_fraction < 1.0 leaves visible gaps between markers so the plot
    reads as a scatter of discrete particles rather than a tiled fill.
    """
    bbox = ax.get_window_extent()   # pixels in display coords
    xlim = ax.get_xlim()
    pixels_per_unit = bbox.width / (xlim[1] - xlim[0])
    pt_per_pixel = 72.0 / ax.figure.dpi
    diameter_pt = pixels_per_unit * pt_per_pixel * dx_axis * fill_fraction
    return diameter_pt ** 2


def _draw_plot1_into_axes(ax_yz, ax_xz, fig, snap, geom,
                          alpha_val, stream_thresh, stream_density,
                          stream_grid, draw_streamlines_flag):
    """Pure rendering function: clear & redraw both panels for one frame.
    Used by both the serial FuncAnimation path and the parallel worker path.
    Uses only its arguments — no module-level state — so it can run inside
    a worker process.
    """
    Nx_, Ny_, Nz_ = geom["Nx"], geom["Ny"], geom["Nz"]
    ix_src_ = geom["ix_src"]; iy_src_ = geom["iy_src"]
    x_axis_ = geom["x_axis"]; y_axis_ = geom["y_axis"]; z_axis_ = geom["z_axis"]
    SRC_X0_, SRC_X1_ = geom["SRC_X0"], geom["SRC_X1"]
    SRC_Y0_, SRC_Y1_ = geom["SRC_Y0"], geom["SRC_Y1"]
    SRC_Z0_, SRC_Z1_ = geom["SRC_Z0"], geom["SRC_Z1"]
    yz_y_flat_ = geom["yz_y_flat"]; yz_z_flat_ = geom["yz_z_flat"]
    xz_x_flat_ = geom["xz_x_flat"]; xz_z_flat_ = geom["xz_z_flat"]
    size_yz_ = geom["size_yz"]; size_xz_ = geom["size_xz"]

    ax_yz.clear(); ax_xz.clear()
    setup_axes_plot1(ax_yz, ax_xz, geom)

    Sw3 = np.clip(snap["Sw"], 0, 1).reshape(Nx_, Ny_, Nz_)
    Sn3 = np.clip(snap["Sn"], 0, 1).reshape(Nx_, Ny_, Nz_)
    Sa3 = np.clip(1.0 - Sw3 - Sn3, 0, 1)

    qx3 = snap["qx"].reshape(Nx_, Ny_, Nz_)
    qy3 = snap["qy"].reshape(Nx_, Ny_, Nz_)
    qz3 = snap["qz"].reshape(Nx_, Ny_, Nz_)
    has_napl = "qxn" in snap
    if has_napl:
        qxn3 = snap["qxn"].reshape(Nx_, Ny_, Nz_)
        qyn3 = snap["qyn"].reshape(Nx_, Ny_, Nz_)
        qzn3 = snap["qzn"].reshape(Nx_, Ny_, Nz_)

    # ── YZ slice ──
    rgb_yz = np.zeros((Ny_, Nz_, 3), dtype=np.float32)
    rgb_yz[..., 0] = Sn3[ix_src_, :, :]
    rgb_yz[..., 1] = Sa3[ix_src_, :, :]
    rgb_yz[..., 2] = Sw3[ix_src_, :, :]
    ax_yz.scatter(yz_y_flat_, yz_z_flat_, c=rgb_yz.reshape(-1, 3),
                  marker="o", s=size_yz_, edgecolors="none", alpha=alpha_val)

    draw_streamlines(ax_yz, y_axis_, z_axis_,
                      qy3[ix_src_, :, :], qz3[ix_src_, :, :],
                      "blue", stream_density,
                      stream_thresh, stream_grid, draw_streamlines_flag)
    if has_napl:
        draw_streamlines(ax_yz, y_axis_, z_axis_,
                          qyn3[ix_src_, :, :], qzn3[ix_src_, :, :],
                          "red", stream_density,
                          stream_thresh, stream_grid, draw_streamlines_flag)

    ax_yz.add_patch(Rectangle((SRC_Y0_, SRC_Z0_), SRC_Y1_-SRC_Y0_, SRC_Z1_-SRC_Z0_,
                               lw=2, ec="k", fc="none", ls="--"))
    ax_yz.set_title(f"YZ slice at x={x_axis_[ix_src_]:.2f} m")

    # ── XZ slice ──
    rgb_xz = np.zeros((Nx_, Nz_, 3), dtype=np.float32)
    rgb_xz[..., 0] = Sn3[:, iy_src_, :]
    rgb_xz[..., 1] = Sa3[:, iy_src_, :]
    rgb_xz[..., 2] = Sw3[:, iy_src_, :]
    ax_xz.scatter(xz_x_flat_, xz_z_flat_, c=rgb_xz.reshape(-1, 3),
                  marker="o", s=size_xz_, edgecolors="none", alpha=alpha_val)

    draw_streamlines(ax_xz, x_axis_, z_axis_,
                      qx3[:, iy_src_, :], qz3[:, iy_src_, :],
                      "blue", stream_density + 0.3,
                      stream_thresh, stream_grid, draw_streamlines_flag)
    if has_napl:
        draw_streamlines(ax_xz, x_axis_, z_axis_,
                          qxn3[:, iy_src_, :], qzn3[:, iy_src_, :],
                          "red", stream_density + 0.3,
                          stream_thresh, stream_grid, draw_streamlines_flag)

    ax_xz.add_patch(Rectangle((SRC_X0_, SRC_Z0_), SRC_X1_-SRC_X0_, SRC_Z1_-SRC_Z0_,
                               lw=2, ec="k", fc="none", ls="--"))
    ax_xz.set_title(f"XZ slice at y={y_axis_[iy_src_]:.2f} m")

    fig.suptitle(
        f"Saturation slices  (R=Sn, G=Sa, B=Sw)  "
        f"+ water (blue) / NAPL (red) streamlines  "
        f"—  step {snap['step']:>6d},  t = {format_time(snap['time'])}",
        fontsize=11)


def _render_frame_worker(task):
    """Worker: build a fresh figure, render one frame, save PNG, close.
    Top-level (picklable). All arguments come through `task`.
    """
    (idx, snap, geom, png_path, dpi, alpha_val,
     stream_thresh, stream_density, stream_grid, draw_streamlines_flag) = task
    # Each worker creates its own figure (matplotlib figures are NOT
    # safe to share across processes).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    fig, (ax_yz, ax_xz) = _plt.subplots(1, 2, figsize=(15, 7),
                                          constrained_layout=True)
    # First draw to give axes a real bbox so marker_size_for would work.
    # We don't recompute size here — geom carries the parent-computed values.
    setup_axes_plot1(ax_yz, ax_xz, geom)
    fig.canvas.draw()
    try:
        _draw_plot1_into_axes(ax_yz, ax_xz, fig, snap, geom,
                               alpha_val, stream_thresh, stream_density,
                               stream_grid, draw_streamlines_flag)
        fig.savefig(png_path, dpi=dpi)
    finally:
        _plt.close(fig)
    return idx


# ── Compute marker sizes from a one-shot reference figure ───────────
# (sizes depend on the actual rendered axis pixel extent, which depends
# on figsize and DPI; both are the same in serial and parallel paths)
_ref_fig, (_ref_yz, _ref_xz) = plt.subplots(1, 2, figsize=(15, 7),
                                              constrained_layout=True)

# Build the geom dict that travels everywhere (worker, serial path)
Y_yz_full, Z_yz_full = np.meshgrid(y_axis, z_axis, indexing="ij")
X_xz_full, Z_xz_full = np.meshgrid(x_axis, z_axis, indexing="ij")

geom = {
    "Nx": Nx, "Ny": Ny, "Nz": Nz,
    "Lx": Lx, "Ly": Ly, "Lz": Lz,
    "ix_src": ix_src, "iy_src": iy_src,
    "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
    "SRC_X0": SRC_X0, "SRC_X1": SRC_X1,
    "SRC_Y0": SRC_Y0, "SRC_Y1": SRC_Y1,
    "SRC_Z0": SRC_Z0, "SRC_Z1": SRC_Z1,
    "yz_y_flat": Y_yz_full.ravel(), "yz_z_flat": Z_yz_full.ravel(),
    "xz_x_flat": X_xz_full.ravel(), "xz_z_flat": Z_xz_full.ravel(),
    "size_yz": None, "size_xz": None,   # filled in next
}
setup_axes_plot1(_ref_yz, _ref_xz, geom)
_ref_fig.canvas.draw()
geom["size_yz"] = marker_size_for(_ref_yz, dy)
geom["size_xz"] = marker_size_for(_ref_xz, dx)
plt.close(_ref_fig)


# ── Decide rendering path ────────────────────────────────────────────
n_frames = len(snaps)
out_path1_base = os.path.join(args.outdir, "anim_slices_rgb")
saved = False

# Check for ffmpeg availability — required for the parallel path
_FFMPEG = shutil.which("ffmpeg")


def _probe_ffmpeg_encoders(ffmpeg_path):
    """Return a set of available video encoder names from `ffmpeg -encoders`.
    Empty set on failure (caller should treat as "unknown, try anyway").
    """
    try:
        proc = subprocess.run([ffmpeg_path, "-hide_banner", "-encoders"],
                              capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return set()
        names = set()
        for line in proc.stdout.splitlines():
            # Lines look like:  " V..... libx264              libx264 H.264 / AVC ..."
            # The encoder name is the first non-empty token after the flags column.
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].startswith(("V", "A", "S")):
                names.add(parts[1])
        return names
    except (OSError, subprocess.SubprocessError):
        return set()


# Quality-preset → encoder argument mapping.
# CRF (Constant Rate Factor) is x264/x265's quality control: lower = better.
# 18 is "visually lossless" for typical content; for sharp scientific viz with
# fine line/marker detail we use 18 by default to keep particle edges crisp.
# Hardware encoders (videotoolbox / nvenc) don't support CRF; we use bitrate.
_QUALITY_PRESETS = {
    # name : (crf_value, bitrate_for_hw, mpeg4_q, openh264_bitrate)
    "high":   (18, "12M", "2", "10M"),
    "medium": (23,  "6M", "5",  "5M"),
    "low":    (28,  "2M", "8",  "2M"),
}


def _build_encoder_candidates(quality, pix_fmt):
    """Build the (codec, ext, extra_args) candidate list for the chosen quality.
    Order = preference (best-quality H.264 first, ubiquitous fallbacks last).
    """
    crf, bitrate, mpeg4_q, oh264_bitrate = _QUALITY_PRESETS[quality]
    return [
        # libx264: CRF-based, highest quality at chosen point. -preset slow gives
        # better compression at small extra encode time vs medium/fast (we use
        # medium since this matters for parallel encode wall time on long runs).
        ("libx264",          "mp4", ["-pix_fmt", pix_fmt,
                                     "-crf", str(crf),
                                     "-preset", "medium"]),
        # videotoolbox: hardware H.264 on macOS. Bitrate-based; -allow_sw
        # permits software fallback if hardware path fails.
        ("h264_videotoolbox","mp4", ["-pix_fmt", pix_fmt,
                                     "-b:v", bitrate,
                                     "-allow_sw", "1"]),
        # nvenc: NVIDIA hardware. Use CQ (constant quality) mode roughly
        # matching CRF for visual equivalence.
        ("h264_nvenc",       "mp4", ["-pix_fmt", pix_fmt,
                                     "-rc", "vbr",
                                     "-cq", str(crf),
                                     "-b:v", bitrate]),
        # libopenh264: Cisco's H.264; bitrate-based.
        ("libopenh264",      "mp4", ["-pix_fmt", pix_fmt,
                                     "-b:v", oh264_bitrate]),
        # mpeg4: ubiquitous fallback. Larger files but plays everywhere.
        # mpeg4 does not support yuv444p; force yuv420p for these.
        ("mpeg4",            "mp4", ["-pix_fmt", "yuv420p", "-q:v", mpeg4_q]),
        ("mpeg4",            "mov", ["-pix_fmt", "yuv420p", "-q:v", mpeg4_q]),
    ]


# Candidate (extension, codec, extra_args) triples in preference order.
_ENCODER_CANDIDATES = _build_encoder_candidates(args.quality, args.pix_fmt)
_use_parallel = (_N_WORKERS > 1) and (_FFMPEG is not None)
if _N_WORKERS > 1 and _FFMPEG is None:
    print(f"  WARNING: ffmpeg not found on PATH; falling back to serial mode")

if _use_parallel:
    # ─── PARALLEL PATH ────────────────────────────────────────────────
    # Render each frame to a PNG in a worker process, then have ffmpeg
    # stitch them into an MP4. Roughly N_workers× faster than serial
    # for streamplot-heavy frames.
    with tempfile.TemporaryDirectory(prefix="sph_anim_", dir=args.outdir) as tmpdir:
        tasks = []
        for idx, snap in enumerate(snaps):
            png_path = os.path.join(tmpdir, f"frame_{idx:06d}.png")
            tasks.append((idx, snap, geom, png_path, args.dpi,
                          args.alpha, args.stream_threshold, args.stream_density,
                          args.stream_grid, not args.no_streamlines))

        print(f"  Rendering {n_frames} frames with {_N_WORKERS} workers ...",
              flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            ctx = _mp.get_context("fork")
        except ValueError:
            print(f"  NOTE: 'fork' start method unavailable; using 'spawn' "
                  f"(slower worker startup)")
            ctx = _mp.get_context("spawn")

        import time as _time
        _t_render_start = _time.time()
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=_N_WORKERS,
                                       mp_context=ctx) as pool:
                futures = [pool.submit(_render_frame_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        idx = fut.result()
                        completed += 1
                        if completed % max(1, n_frames // 10) == 0:
                            elapsed = _time.time() - _t_render_start
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (n_frames - completed) / rate if rate > 0 else 0
                            print(f"    ... {completed}/{n_frames} frames "
                                  f"({rate:.1f} fr/s, ETA {eta:.0f}s)", flush=True)
                    except Exception as e:
                        print(f"    WORKER FAILED: {type(e).__name__}: {e}")
            _t_render = _time.time() - _t_render_start
            print(f"  Rendered {completed}/{n_frames} frames in {_t_render:.1f}s "
                  f"({completed/_t_render:.1f} fr/s)", flush=True)
        except Exception as e:
            print(f"  ERROR in parallel rendering: {e}; falling back to serial")
            _use_parallel = False

        if _use_parallel and completed > 0:
            # ffmpeg stitch — try a chain of encoders, fall back through
            # codecs and containers as needed.  The first one that actually
            # produces a valid file wins.
            available = _probe_ffmpeg_encoders(_FFMPEG)
            if available:
                # Filter the candidate list to encoders we know are present.
                # If probing failed (empty set), try them all anyway — the
                # `-encoders` listing is informational; some builds may still
                # accept encoders that don't appear there.
                candidates = [(c, e, a) for (c, e, a) in _ENCODER_CANDIDATES
                              if c in available]
                if not candidates:
                    print(f"  WARNING: none of the preferred encoders are available "
                          f"in this ffmpeg build. Trying mpeg4 as last resort.")
                    candidates = [("mpeg4", "mp4", ["-q:v", "5"])]
            else:
                candidates = list(_ENCODER_CANDIDATES)

            print(f"  Encoding video ({len(candidates)} encoder candidate"
                  f"{'s' if len(candidates) > 1 else ''}) ...", flush=True)

            input_pattern = os.path.join(tmpdir, "frame_%06d.png")
            for codec, ext, extra in candidates:
                out_path = out_path1_base + "." + ext
                cmd = [_FFMPEG, "-y",
                       "-framerate", str(args.fps),
                       "-i", input_pattern,
                       "-c:v", codec,
                       *extra,
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       out_path]
                _t_enc = _time.time()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    _t_enc = _time.time() - _t_enc
                    if proc.returncode == 0:
                        print(f"  Saved {out_path}  "
                              f"(codec={codec}, encode: {_t_enc:.1f}s)")
                        saved = True
                        break
                    else:
                        # Show only the last meaningful error line
                        err_tail = proc.stderr.strip().splitlines()[-1] \
                                   if proc.stderr.strip() else "(no stderr)"
                        print(f"  encoder '{codec}' failed: {err_tail}")
                except Exception as e:
                    print(f"  encoder '{codec}' invocation failed: {e}")

            if not saved:
                print(f"  All ffmpeg encoder candidates failed; "
                      f"will fall back to serial mode (which can write GIF).")

if not saved:
    # ─── SERIAL PATH (FuncAnimation) ──────────────────────────────────
    # Either the user asked for serial, or parallel rendering failed.
    # Falls back to FuncAnimation; supports both MP4 (via ffmpeg) and
    # GIF (via Pillow) writers.
    print(f"  Rendering {n_frames} frames serially ...", flush=True)
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 7),
                                        constrained_layout=True)
    setup_axes_plot1(ax1a, ax1b, geom)
    fig1.canvas.draw()

    def _animate(frame_idx):
        _draw_plot1_into_axes(ax1a, ax1b, fig1, snaps[frame_idx], geom,
                               args.alpha, args.stream_threshold, args.stream_density,
                               args.stream_grid, not args.no_streamlines)

    _draw_plot1_into_axes(ax1a, ax1b, fig1, snaps[0], geom,
                           args.alpha, args.stream_threshold, args.stream_density,
                           args.stream_grid, not args.no_streamlines)
    anim1 = animation.FuncAnimation(fig1, _animate, frames=n_frames,
                                      interval=1000//args.fps)
    for ext, writer_name in [("mp4", "ffmpeg"), ("gif", "pillow")]:
        try:
            full = out_path1_base + "." + ext
            anim1.save(full, writer=writer_name, fps=args.fps, dpi=args.dpi)
            print(f"  Saved {full}")
            saved = True
            break
        except Exception as e:
            print(f"  ({writer_name} failed: {e})")
    plt.close(fig1)

if not saved:
    print("  ERROR: could not save Plot 1 animation")



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
        f"\nstep {snap['step']:>6d},  t = {format_time(snap['time'])}",
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
