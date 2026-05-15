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
    python postprocess_napl.py --input my_run/snapshots --outdir figs/
    python postprocess_napl.py --input my_run/           # auto-detect snapshots/ subdir
    python postprocess_napl.py --skip 5         # use every 5th snapshot
    python postprocess_napl.py --fps 10

The simulation writes one HDF5 (or NPZ) file per snapshot to
$OUTDIR/snapshots/step_NNNNNNNNNNNN.{h5,npz}.  Point --input at either
the snapshots/ directory itself or at the run's --outdir (we auto-find
the snapshots/ subdirectory in that case).
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
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


def format_dd_hh_mm_ss(t_seconds):
    """Format a physical time (in seconds) as 'dd:hh:mm:ss'.
    Always uses fixed 4-field format, suitable for static-figure panel
    titles where uniform width across panels is desirable.

    Sub-second times are shown as '00:00:00:00.xxxx' so we don't lose
    precision for early-time panels.
    """
    if t_seconds < 1.0 and t_seconds > 0:
        # Keep sub-second precision visible
        return f"00:00:00:00 ({t_seconds:.2e} s)"
    total_s = int(round(t_seconds))
    days,  rem = divmod(total_s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


# ======================================================================
# CLI
# ======================================================================
parser = argparse.ArgumentParser(description="Animated post-processing of SPH snapshots")
parser.add_argument("--input",  default="snapshots",
                    help="Path to per-snapshot files. Either the snapshots/ "
                         "directory directly (containing step_*.h5 or "
                         "step_*.npz files), or a run's --outdir (we look for "
                         "snapshots/ inside it). Default: ./snapshots")
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
parser.add_argument("--mp-start-method", choices=["spawn", "fork", "forkserver"],
                    default="spawn",
                    help="Multiprocessing start method for workers. Default 'spawn' "
                         "is safest — each worker is a fresh Python process. 'fork' "
                         "is faster to start but copies the parent's memory state, "
                         "which can corrupt thread-using libraries (HDF5, OpenMP, "
                         "matplotlib) and cause bus errors / segfaults on HPC. "
                         "Use 'fork' only if you've verified it works on your "
                         "cluster and need the faster startup.")


# ====================================================================
# FUNCTION DEFINITIONS (at module scope so spawn-mode workers can re-import
# them without re-running the runtime work)
# ====================================================================

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


def _validate_npz_keys(loader, expected_npart):
    """Return (ok, reason) for whether an NPZ snapshot has all required keys
    with correct first-axis shape. Cheap: doesn't load full data arrays.

    Catches the same partial-write / truncated-file failure modes as the
    HDF5 validator: missing keys, wrong shape, or low-level I/O errors.
    """
    required = ("step", "time_s", "Nx", "Ny", "Nz", "Lx", "Ly", "Lz",
                "x", "y", "z", "h", "Sn", "Sw", "qx", "qy", "qz", "is_source")
    try:
        for k in required:
            if k not in loader.files:
                return False, f"missing key '{k}'"
        if expected_npart is not None:
            for k in ("h", "Sn", "Sw", "qx", "qy", "qz"):
                shp = loader[k].shape
                if shp[0] != expected_npart:
                    return False, (f"key '{k}' has shape {shp}, "
                                   f"expected ({expected_npart},)")
    except _NPZ_ERRORS as e:
        return False, f"I/O error: {type(e).__name__}: {e}"
    return True, None


def _read_h5_snapshot_file(path, expected_npart=None, want_napl_q=False):
    """Open ONE per-snapshot HDF5 file and read its data.

    Returns (ok, payload_or_reason).  ok=True payload is a dict with
    step, time, h, Sn, Sw, qx, qy, qz (and qxn,qyn,qzn if present).
    ok=False payload is a string explaining the failure.
    """
    try:
        with h5py.File(path, "r") as f:
            ok, reason = _validate_h5_snapshot(f, expected_npart)
            if not ok:
                return False, reason
            d = {"step": int(f.attrs["step"]),
                 "time": float(f.attrs["time_s"]),
                 "h":  f["h"][...],  "Sn": f["Sn"][...], "Sw": f["Sw"][...],
                 "qx": f["qx"][...], "qy": f["qy"][...], "qz": f["qz"][...]}
            if want_napl_q and all(k in f for k in ("qxn", "qyn", "qzn")):
                d["qxn"] = f["qxn"][...]
                d["qyn"] = f["qyn"][...]
                d["qzn"] = f["qzn"][...]
            return True, d
    except _H5_ERRORS as e:
        return False, f"I/O error: {type(e).__name__}: {e}"


def _read_h5_snapshot_meta(path):
    """Extract geometry/coords metadata from a single HDF5 snapshot file."""
    try:
        with h5py.File(path, "r") as f:
            ok, reason = _validate_h5_snapshot(f, expected_npart=None)
            if not ok:
                return None, reason
            Nx_ = int(f.attrs["Nx"]); Ny_ = int(f.attrs["Ny"]); Nz_ = int(f.attrs["Nz"])
            meta = {
                "Nx": Nx_, "Ny": Ny_, "Nz": Nz_,
                "Lx": float(f.attrs["Lx"]),
                "Ly": float(f.attrs["Ly"]),
                "Lz": float(f.attrs["Lz"]),
                "x": f["x"][...], "y": f["y"][...], "z": f["z"][...],
                "is_source": f["is_source"][...].astype(bool),
                "has_napl_q": all(k in f for k in ("qxn", "qyn", "qzn")),
            }
            return meta, None
    except _H5_ERRORS as e:
        return None, f"metadata read failed: {type(e).__name__}: {e}"


def _read_npz_snapshot_file(path, expected_npart=None, want_napl_q=False):
    """Open ONE per-snapshot NPZ file and read its data."""
    try:
        f = np.load(path)
        ok, reason = _validate_npz_keys(f, expected_npart)
        if not ok:
            return False, reason
        d = {"step": int(f["step"]), "time": float(f["time_s"]),
             "h":  np.asarray(f["h"]),
             "Sn": np.asarray(f["Sn"]),
             "Sw": np.asarray(f["Sw"]),
             "qx": np.asarray(f["qx"]),
             "qy": np.asarray(f["qy"]),
             "qz": np.asarray(f["qz"])}
        if want_napl_q and all(k in f.files for k in ("qxn", "qyn", "qzn")):
            d["qxn"] = np.asarray(f["qxn"])
            d["qyn"] = np.asarray(f["qyn"])
            d["qzn"] = np.asarray(f["qzn"])
        return True, d
    except _NPZ_ERRORS as e:
        return False, f"I/O error: {type(e).__name__}: {e}"


def _read_npz_snapshot_meta(path):
    """Extract geometry metadata from a single NPZ snapshot file."""
    try:
        f0 = np.load(path)
        ok, reason = _validate_npz_keys(f0, expected_npart=None)
        if not ok:
            return None, reason
        Nx_ = int(f0["Nx"]); Ny_ = int(f0["Ny"]); Nz_ = int(f0["Nz"])
        meta = {
            "Nx": Nx_, "Ny": Ny_, "Nz": Nz_,
            "Lx": float(f0["Lx"]), "Ly": float(f0["Ly"]), "Lz": float(f0["Lz"]),
            "x": np.asarray(f0["x"]), "y": np.asarray(f0["y"]),
            "z": np.asarray(f0["z"]),
            "is_source": np.asarray(f0["is_source"]).astype(bool),
            "has_napl_q": all(k in f0.files for k in ("qxn", "qyn", "qzn")),
        }
        return meta, None
    except _NPZ_ERRORS as e:
        return None, f"metadata read failed: {type(e).__name__}: {e}"


def _read_step_time_only(path, fmt):
    """Read just (step, time_s) from a snapshot file. Cheap — does not
    touch any per-particle data. Returns (step, time) or (None, None) on
    failure."""
    try:
        if fmt == "h5":
            with h5py.File(path, "r") as f:
                return int(f.attrs["step"]), float(f.attrs["time_s"])
        else:
            with np.load(path) as ld:
                return int(ld["step"]), float(ld["time_s"])
    except (_H5_ERRORS if fmt == "h5" else _NPZ_ERRORS) as e:
        return None, None
    except Exception:
        return None, None


def scan_snapshots(snap_dir, skip=1):
    """Scan a per-snapshot directory and return a lightweight catalog.

    No per-particle data is loaded — only file paths, step numbers, and
    times. Use read_one_snapshot(path, fmt, ...) to load actual data on
    demand inside workers.

    Returns (meta, frames):
      - meta: dict with grid/domain info and positions, from the FIRST
        valid snapshot. ~140 MB at 1M particles — only loaded once.
      - frames: list of dicts in chronological order:
          {"step": int, "time": float, "path": str, "fmt": "h5"|"npz"}
        No per-particle arrays. ~100 bytes per entry. Cheap.

    Bad/corrupt files are reported and skipped.
    """
    print(f"Scanning per-snapshot files in {snap_dir}/ ...", flush=True)
    if not os.path.isdir(snap_dir):
        raise RuntimeError(
            f"Snapshot directory does not exist: {snap_dir}\n"
            f"  Point --input at the directory containing step_*.h5 or step_*.npz files,\n"
            f"  or at its parent (looks for ./snapshots inside)."
        )

    by_step = {}
    try:
        all_entries = os.listdir(snap_dir)
    except OSError as e:
        raise RuntimeError(f"Cannot list {snap_dir}: {e}") from e

    for entry in all_entries:
        if not entry.startswith("step_"):
            continue
        if entry.endswith(".h5"):
            step = _step_from_name(entry)
            by_step[step] = ("h5", entry)
        elif entry.endswith(".npz"):
            step = _step_from_name(entry)
            if step not in by_step:
                by_step[step] = ("npz", entry)

    if not by_step:
        raise RuntimeError(
            f"No step_*.h5 or step_*.npz files in {snap_dir}.\n"
            f"  If you have a legacy single-file HDF5 snapshot, that format is no\n"
            f"  longer supported (it caused bus errors on Lustre at large sizes).\n"
            f"  The simulation now writes one HDF5 file per snapshot."
        )

    steps_sorted = sorted(by_step.keys())
    print(f"  Found {len(steps_sorted)} candidate snapshot file(s)")

    skipped = []
    meta = None

    # Pass 1: find first valid file for metadata (grid + positions)
    for step in steps_sorted:
        fmt, name = by_step[step]
        path = os.path.join(snap_dir, name)
        if fmt == "h5":
            m, reason = _read_h5_snapshot_meta(path)
        else:
            m, reason = _read_npz_snapshot_meta(path)
        if m is None:
            skipped.append((name, reason))
            continue
        meta = m
        break

    if meta is None:
        raise RuntimeError(
            f"All {len(steps_sorted)} snapshot files in {snap_dir} are unreadable.\n"
            f"  First failures: {skipped[:3]}"
        )

    already_failed = {name for name, _ in skipped}

    # Pass 2: build lightweight catalog (step, time, path, fmt) per frame.
    # Each entry reads only the (step, time_s) attrs — no array data.
    steps_subset = steps_sorted[::skip]
    frames = []
    for step in steps_subset:
        fmt, name = by_step[step]
        if name in already_failed:
            continue
        path = os.path.join(snap_dir, name)
        s, t = _read_step_time_only(path, fmt)
        if s is None:
            skipped.append((name, "couldn't read step/time attrs"))
            continue
        frames.append({"step": s, "time": t, "path": path, "fmt": fmt})

    if skipped:
        print(f"  WARNING: skipped {len(skipped)} unreadable snapshot file(s):")
        for name, reason in skipped[:5]:
            print(f"    - {name}: {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
        print(f"  Likely cause: incomplete rsync transfer, partial write, "
              f"or file corruption.")

    if len(frames) == 0:
        raise RuntimeError(
            f"No usable snapshots in {snap_dir} after skipping {len(skipped)} bad ones."
        )

    # ─── Detect particle ordering and compute lex permutation ──────────
    # Particles in the GPU script are reordered with Z-order (Morton curve)
    # for memory coalescence. CPU/MPI snapshots are in lex order.
    # We always want lex order downstream (so 1D->3D reshape works for slice
    # rendering). Detect the order from positions and compute a permutation
    # that puts particles in lex order: idx_lex = i*Ny*Nz + j*Nz + k.
    Nx_m = int(meta["Nx"]); Ny_m = int(meta["Ny"]); Nz_m = int(meta["Nz"])
    Lx_m = float(meta["Lx"]); Ly_m = float(meta["Ly"]); Lz_m = float(meta["Lz"])
    dx_m = Lx_m / (Nx_m - 1)
    dy_m = Ly_m / (Ny_m - 1)
    dz_m = Lz_m / (Nz_m - 1)
    ii = np.round(meta["x"] / dx_m).astype(np.int64)
    jj = np.round(meta["y"] / dy_m).astype(np.int64)
    kk = np.round(meta["z"] / dz_m).astype(np.int64)
    lex_index = ii * (Ny_m * Nz_m) + jj * Nz_m + kk
    perm = np.argsort(lex_index, kind="stable")
    identity = np.arange(len(perm), dtype=perm.dtype)
    if np.array_equal(perm, identity):
        meta["lex_perm"] = None
        print(f"  Particle order: lexicographic (no reordering needed)")
    else:
        meta["lex_perm"] = perm
        print(f"  Particle order: non-lex (e.g. Z-order from GPU runs); "
              f"will apply permutation to restore lex order on read")
        # Apply to meta arrays so downstream code sees lex-ordered positions
        meta["x"] = meta["x"][perm]
        meta["y"] = meta["y"][perm]
        meta["z"] = meta["z"][perm]
        meta["is_source"] = meta["is_source"][perm]

    return meta, frames


def read_one_snapshot(path, fmt, expected_npart, want_napl_q, lex_perm=None):
    """Read one snapshot's data from disk. Returns the snap dict directly
    (the data is yours to use and discard). Raises on I/O failure with a
    descriptive message; the caller decides how to handle it.

    If lex_perm is provided (non-None), the read arrays are reordered by
    that permutation before being returned. Used to convert GPU Z-ordered
    snapshots into lex order so 1D→3D reshape works correctly.

    This is the function workers call. After it returns, the snap dict's
    arrays are referenced only by the worker — when the worker function
    returns, they go out of scope and are garbage-collected promptly.
    """
    if fmt == "h5":
        ok, payload = _read_h5_snapshot_file(path, expected_npart, want_napl_q)
    else:
        ok, payload = _read_npz_snapshot_file(path, expected_npart, want_napl_q)
    if not ok:
        raise RuntimeError(f"Failed to read {path}: {payload}")
    if lex_perm is not None:
        # Apply lex permutation to all per-particle arrays in the payload.
        # Keys whose values have shape (N_part,) are reordered; scalar keys
        # (step, time) are passed through.
        n = expected_npart
        for k, v in list(payload.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                payload[k] = v[lex_perm]
    return payload


# Resolve --input to a snapshot directory.
# The simulation writes per-snapshot files to $OUTDIR/snapshots/, so --input
# can be either that snapshots/ directory directly, or the run's --outdir.
def _resolve_snap_dir(input_path):
    # Legacy single-file HDF5 is no longer supported
    if os.path.isfile(input_path) and input_path.endswith(".h5"):
        sys.exit(
            f"ERROR: --input points at a single HDF5 file ({input_path}).\n"
            f"  This single-file format is no longer supported; the simulation\n"
            f"  now writes one file per snapshot.  Point --input at the\n"
            f"  snapshots/ directory containing step_*.h5 files."
        )
    if not os.path.isdir(input_path):
        sys.exit(f"ERROR: --input directory does not exist: {input_path}")

    # If the path contains step_*.{h5,npz} files directly, use it
    import glob as _g
    direct_h5  = _g.glob(os.path.join(input_path, "step_*.h5"))
    direct_npz = _g.glob(os.path.join(input_path, "step_*.npz"))
    if direct_h5 or direct_npz:
        return input_path
    # Otherwise look for a snapshots/ subdir
    sub = os.path.join(input_path, "snapshots")
    if os.path.isdir(sub):
        sub_h5  = _g.glob(os.path.join(sub, "step_*.h5"))
        sub_npz = _g.glob(os.path.join(sub, "step_*.npz"))
        if sub_h5 or sub_npz:
            return sub
    sys.exit(
        f"ERROR: no per-snapshot files found at {input_path} or "
        f"{input_path}/snapshots/.\n  Expected step_*.h5 or step_*.npz files."
    )

def to3d(arr):
    return arr.reshape(Nx, Ny, Nz)

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
    """Worker: open ONE snapshot file, render its frame to a PNG, close,
    release memory, return.

    Top-level (picklable). After this function returns, the snap dict's
    arrays go out of scope inside this worker process and are garbage
    collected.  Peak memory per worker is ~1 snapshot worth of data plus
    matplotlib's rendering buffers.
    """
    (idx, path, fmt, expected_npart, want_napl_q, lex_perm, geom, png_path, dpi,
     alpha_val, stream_thresh, stream_density, stream_grid,
     draw_streamlines_flag) = task
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import gc as _gc

    # Read this frame's data on demand (the whole reason for streaming).
    # lex_perm reorders Z-order GPU snapshots into lex order; None = no-op.
    snap = read_one_snapshot(path, fmt, expected_npart, want_napl_q, lex_perm)

    fig, (ax_yz, ax_xz) = _plt.subplots(1, 2, figsize=(15, 7),
                                          constrained_layout=True)
    setup_axes_plot1(ax_yz, ax_xz, geom)
    fig.canvas.draw()
    try:
        _draw_plot1_into_axes(ax_yz, ax_xz, fig, snap, geom,
                               alpha_val, stream_thresh, stream_density,
                               stream_grid, draw_streamlines_flag)
        fig.savefig(png_path, dpi=dpi)
    finally:
        _plt.close(fig)
        # Explicitly drop refs and request GC before returning so the
        # worker process's RSS settles between tasks.  Important for
        # spawn-mode pools where the worker is reused for many frames.
        del snap
        _gc.collect()
    return idx


# ── Compute marker sizes from a one-shot reference figure ───────────
# (sizes depend on the actual rendered axis pixel extent, which depends
# on figsize and DPI; both are the same in serial and parallel paths)
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


def _stitch_pngs_to_video(ffmpeg_path, png_dir, out_base, fps,
                          encoder_candidates, png_pattern="frame_%06d.png"):
    """Stitch a directory of numbered PNGs into a video.

    Tries each entry from encoder_candidates in order (tuples of
    (codec, ext, extra_args)). The first one that produces a valid file
    wins. Includes the `-vf pad=ceil(iw/2)*2:ceil(ih/2)*2` filter, required
    by libx264 with yuv420p when frame dimensions aren't even.

    Returns (success, output_path) — output_path is None on failure.
    Prints progress and the last line of ffmpeg's stderr on per-encoder
    failure, so the user sees WHY a candidate didn't work.

    Used by both Plot 1 (slice animation) and Plot 2 (profile animation).
    """
    if ffmpeg_path is None:
        return False, None

    available = _probe_ffmpeg_encoders(ffmpeg_path)
    if available:
        candidates = [(c, e, a) for (c, e, a) in encoder_candidates
                      if c in available]
        if not candidates:
            print(f"  WARNING: none of the preferred encoders are available "
                  f"in this ffmpeg build. Trying mpeg4 as last resort.")
            candidates = [("mpeg4", "mp4", ["-q:v", "5"])]
    else:
        # Probing returned nothing — try them all anyway (some builds don't
        # advertise encoders in the standard format).
        candidates = list(encoder_candidates)

    print(f"  Encoding video ({len(candidates)} encoder candidate"
          f"{'s' if len(candidates) > 1 else ''}) ...", flush=True)

    input_pattern = os.path.join(png_dir, png_pattern)
    import time as _time
    for codec, ext, extra in candidates:
        out_path = out_base + "." + ext
        cmd = [ffmpeg_path, "-y",
               "-framerate", str(fps),
               "-i", input_pattern,
               "-c:v", codec,
               *extra,
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               out_path]
        _t_enc = _time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600)
            _t_enc = _time.time() - _t_enc
            if (proc.returncode == 0
                    and os.path.exists(out_path)
                    and os.path.getsize(out_path) > 1000):
                print(f"  Saved {out_path}  "
                      f"(codec={codec}, encode: {_t_enc:.1f}s)")
                return True, out_path
            # Show the last meaningful stderr line so user can diagnose
            err_tail = (proc.stderr.strip().splitlines()[-1]
                        if proc.stderr.strip() else "(no stderr)")
            print(f"  encoder '{codec}' failed: {err_tail}")
        except Exception as e:
            print(f"  encoder '{codec}' invocation failed: {e}")

    return False, None


def _stitch_pngs_to_gif(png_dir, out_base, fps, png_pattern_glob="*.png"):
    """Last-resort GIF fallback via Pillow. Returns (success, output_path)."""
    try:
        from PIL import Image
        import glob as _g
        pngs = sorted(_g.glob(os.path.join(png_dir, png_pattern_glob)))
        if not pngs:
            return False, None
        imgs = [Image.open(p) for p in pngs]
        out_path = out_base + ".gif"
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                      duration=int(1000 / fps), loop=0)
        print(f"  Saved {out_path}  (pillow GIF fallback)")
        return True, out_path
    except Exception as e:
        print(f"  GIF fallback failed: {e}")
        return False, None


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
def _draw_plot2_into_axes(ax, snap, geom_p2):
    """Render saturation profiles (Sw, Sn, Sa) vs z into a given axes.
    Same drawing logic used by both the animation worker and the 4-panel
    static figure. `snap` is one snapshot's data dict; geom_p2 carries
    ix_src, iy_src, z_axis, Lz, x_axis, y_axis, Nx, Ny, Nz.
    """
    ix_src_ = geom_p2["ix_src"]; iy_src_ = geom_p2["iy_src"]
    z_axis_ = geom_p2["z_axis"]; Lz_ = geom_p2["Lz"]
    Nx_ = geom_p2["Nx"]; Ny_ = geom_p2["Ny"]; Nz_ = geom_p2["Nz"]
    x_axis_ = geom_p2["x_axis"]; y_axis_ = geom_p2["y_axis"]

    Sw3 = snap["Sw"].reshape(Nx_, Ny_, Nz_)
    Sn3 = snap["Sn"].reshape(Nx_, Ny_, Nz_)
    Sa3 = np.clip(1.0 - Sw3 - Sn3, 0.0, 1.0)
    Sw_col = Sw3[ix_src_, iy_src_, :]
    Sn_col = Sn3[ix_src_, iy_src_, :]
    Sa_col = Sa3[ix_src_, iy_src_, :]
    h_col  = snap["h"].reshape(Nx_, Ny_, Nz_)[ix_src_, iy_src_, :]

    ax.plot(Sw_col, z_axis_, "b-",  lw=2,    label=r"$S_w$ (water)")
    ax.plot(Sn_col, z_axis_, "r-",  lw=2,    label=r"$S_n$ (NAPL)")
    ax.plot(Sa_col, z_axis_, "g--", lw=1.5,  label=r"$S_a$ (air)")

    # Water-table elevation: linear-interpolate where h crosses 0
    sign_change = np.where(np.diff(np.sign(h_col)))[0]
    if len(sign_change) > 0:
        i0 = sign_change[0]; i1 = i0 + 1
        h0, h1 = h_col[i0], h_col[i1]
        z0, z1 = z_axis_[i0], z_axis_[i1]
        if abs(h1 - h0) > 1e-30:
            z_wt = z0 - h0 * (z1 - z0) / (h1 - h0)
            ax.axhline(z_wt, color="cyan", ls=":", lw=1, alpha=0.7,
                        label=f"WT  z={z_wt:.2f} m")

    ax.set_xlabel("Saturation [-]")
    ax.set_ylabel("z [m]")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, Lz_)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def _render_plot2_frame_worker(task):
    """Worker for Plot 2 (profile vs z): open one snapshot, render one PNG,
    release memory, return. Streaming pattern."""
    (idx, path, fmt, expected_npart, want_napl_q, lex_perm,
     geom_p2, png_path, dpi) = task
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import gc as _gc

    snap = read_one_snapshot(path, fmt, expected_npart, want_napl_q, lex_perm)

    fig, ax = _plt.subplots(figsize=(7, 8), constrained_layout=True)
    try:
        _draw_plot2_into_axes(ax, snap, geom_p2)
        x_ax = geom_p2["x_axis"]; y_ax = geom_p2["y_axis"]
        ix_s = geom_p2["ix_src"]; iy_s = geom_p2["iy_src"]
        ax.set_title(
            f"Saturation profiles at (x={x_ax[ix_s]:.2f}, y={y_ax[iy_s]:.2f}) m"
            f"\nstep {snap['step']:>6d},  t = {format_time(snap['time'])}",
            fontsize=11)
        fig.savefig(png_path, dpi=dpi)
    finally:
        _plt.close(fig)
        del snap
        _gc.collect()
    return idx


# Build geom dict for Plot 2 (lightweight — no per-particle data)
def _pick_panel_indices(n):
    """Pick 4 frame indices: first, 1/3, 2/3, last. Works for any n>=1."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    if n == 3:
        return [0, 1, 2]
    return [0, n // 3, (2 * n) // 3, n - 1]




# ====================================================================
# MAIN ENTRY POINT (everything runtime-side lives inside main() so that
# spawn-mode worker re-imports don't re-execute the rendering pipeline)
# ====================================================================

def main():
    # Names used by module-level helper functions (to3d, saturations_from)
    # must live at module scope so those functions can find them when
    # called from inside main(). Declaring them global here means main()'s
    # assignments below populate the module namespace.
    global Nx, Ny, Nz

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
        print("NOTE: h5py not available — only NPZ snapshots can be read.")


    # ======================================================================
    # LOAD SNAPSHOTS  (HDF5 if available, else NPZ directory)
    # ======================================================================
    snap_dir = _resolve_snap_dir(args.input)

    try:
        meta, frames = scan_snapshots(snap_dir, skip=args.skip)
    except Exception as e:
        print(f"ERROR scanning snapshots: {e}")
        sys.exit(1)

    Nx = int(meta["Nx"]); Ny = int(meta["Ny"]); Nz = int(meta["Nz"])
    Lx = float(meta["Lx"]); Ly = float(meta["Ly"]); Lz = float(meta["Lz"])
    x_flat = meta["x"]; y_flat = meta["y"]; z_flat = meta["z"]
    is_source_flat = meta["is_source"]
    has_napl_q = meta["has_napl_q"]
    # lex_perm: array that permutes per-particle data into lex order, or
    # None if the data is already lex-ordered. Threaded through worker
    # tasks so each frame's read is reordered consistently.
    _lex_perm = meta.get("lex_perm")

    print(f"  Grid: {Nx} x {Ny} x {Nz}    Domain: {Lx} x {Ly} x {Lz} m")
    print(f"  Catalog: {len(frames)} frames.   "
          f"t = [{frames[0]['time']:.1f}, {frames[-1]['time']:.1f}] s")
    if not has_napl_q:
        print("  NOTE: NAPL velocity not in snapshots — red NAPL streamlines will be skipped.")

    # Memory budget hint: in the new streaming workflow each worker holds ONE
    # snapshot's data at a time, not all of them. Peak per-worker memory is
    # the raw snapshot size (positions + ~14 fields) plus matplotlib's
    # rendering buffers (~3x raw).
    _npart = Nx * Ny * Nz
    _n_fields_per_snap = 14
    _bytes_per_snap = _npart * _n_fields_per_snap * 8
    _per_worker_estimate_mb = (_bytes_per_snap * 4) / (1024**2)
    _total_estimate_mb = _per_worker_estimate_mb * _N_WORKERS
    print(f"  Streaming workflow: each worker holds 1 snapshot at a time "
          f"(~{_per_worker_estimate_mb:.0f} MB / worker).")
    print(f"  Total in-flight memory: ~{_total_estimate_mb:.0f} MB across "
          f"{_N_WORKERS} worker{'s' if _N_WORKERS > 1 else ''}.")


    # ======================================================================
    # GEOMETRY: reshape, infer source location
    # ======================================================================
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
    print("\nBuilding Plot 1 (slice animation: particle scatter + streamlines) ...", flush=True)

    # ── Helpers used in BOTH serial and parallel paths ────────────────────

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
    n_frames = len(frames)
    out_path1_base = os.path.join(args.outdir, "anim_slices_rgb")
    saved = False

    # Check for ffmpeg availability — required for the parallel path
    _FFMPEG = shutil.which("ffmpeg")


    _ENCODER_CANDIDATES = _build_encoder_candidates(args.quality, args.pix_fmt)
    _use_parallel = (_N_WORKERS > 1) and (_FFMPEG is not None)
    if _N_WORKERS > 1 and _FFMPEG is None:
        print(f"  WARNING: ffmpeg not found on PATH; falling back to serial mode")

    if _use_parallel:
        # ─── PARALLEL PATH (streaming) ───────────────────────────────────
        # Each worker opens one snapshot file, renders one PNG, releases
        # memory, and returns. No data is held in the parent process between
        # tasks (only the lightweight `frames` catalog).
        with tempfile.TemporaryDirectory(prefix="sph_anim_", dir=args.outdir) as tmpdir:
            _expected_npart = Nx * Ny * Nz
            tasks = []
            for idx, fr in enumerate(frames):
                png_path = os.path.join(tmpdir, f"frame_{idx:06d}.png")
                tasks.append((idx, fr["path"], fr["fmt"], _expected_npart, has_napl_q,
                              _lex_perm,
                              geom, png_path, args.dpi,
                              args.alpha, args.stream_threshold, args.stream_density,
                              args.stream_grid, not args.no_streamlines))

            print(f"  Rendering {n_frames} frames with {_N_WORKERS} workers "
                  f"(start method: {args.mp_start_method}) ...",
                  flush=True)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from concurrent.futures.process import BrokenProcessPool
            try:
                ctx = _mp.get_context(args.mp_start_method)
            except ValueError:
                print(f"  NOTE: '{args.mp_start_method}' start method unavailable; "
                      f"falling back to 'spawn'")
                ctx = _mp.get_context("spawn")

            import time as _time
            _t_render_start = _time.time()
            completed = 0
            _broken_pool = False
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
                        except BrokenProcessPool as e:
                            if not _broken_pool:
                                print(f"\n    *** WORKER POOL BROKEN: {e}")
                                print(f"    A worker process died unexpectedly "
                                      f"(SIGBUS / SIGSEGV / SIGKILL).")
                                print(f"    Likely causes:")
                                print(f"      1. Out-of-memory: too little --mem for "
                                      f"{_N_WORKERS} workers. Each worker needs roughly "
                                      f"3-4x snapshot size (matplotlib intermediate buffers).")
                                print(f"      2. fork+threaded-library mismatch: try "
                                      f"--mp-start-method spawn (current: "
                                      f"{args.mp_start_method})")
                                print(f"      3. HDF5/Lustre I/O failure: stage data to "
                                      f"$SLURM_TMPDIR (node-local) and retry")
                                print(f"    Will fall back to serial rendering.\n")
                                _broken_pool = True
                        except Exception as e:
                            print(f"    WORKER FAILED: {type(e).__name__}: {e}")
                if _broken_pool:
                    _use_parallel = False
                else:
                    _t_render = _time.time() - _t_render_start
                    print(f"  Rendered {completed}/{n_frames} frames in {_t_render:.1f}s "
                          f"({completed/_t_render:.1f} fr/s)", flush=True)
            except BrokenProcessPool as e:
                print(f"  ERROR: worker pool broken before any frames completed: {e}")
                print(f"  Falling back to serial mode. Consider --workers 1 or more --mem.")
                _use_parallel = False
            except Exception as e:
                print(f"  ERROR in parallel rendering: {e}; falling back to serial")
                _use_parallel = False

            if _use_parallel and completed > 0:
                # ffmpeg stitch via shared helper (so Plot 1 and Plot 2 use
                # the exact same encoder fallback logic — no chance of one
                # path silently misbehaving while the other works).
                saved, _ = _stitch_pngs_to_video(
                    _FFMPEG, tmpdir, out_path1_base, args.fps,
                    _ENCODER_CANDIDATES)
                if not saved:
                    print(f"  All ffmpeg encoder candidates failed; "
                          f"will fall back to serial mode (which can write GIF).")

    if not saved:
        # ─── SERIAL PATH (streaming PNG → ffmpeg or pillow GIF) ───────────
        # User asked for serial OR the parallel path failed.  We still do
        # streaming-PNGs-then-stitch instead of FuncAnimation, because
        # FuncAnimation builds up all frames in memory before saving — which
        # is exactly what we're trying to avoid.
        print(f"  Rendering {n_frames} frames serially (streaming) ...", flush=True)
        _expected_npart = Nx * Ny * Nz
        with tempfile.TemporaryDirectory(prefix="sph_anim_", dir=args.outdir) as tmpdir:
            # Render each frame to its own PNG, one at a time
            import time as _time
            _t_render_start = _time.time()
            for idx, fr in enumerate(frames):
                png_path = os.path.join(tmpdir, f"frame_{idx:06d}.png")
                task = (idx, fr["path"], fr["fmt"], _expected_npart, has_napl_q,
                        _lex_perm,
                        geom, png_path, args.dpi,
                        args.alpha, args.stream_threshold, args.stream_density,
                        args.stream_grid, not args.no_streamlines)
                try:
                    _render_frame_worker(task)
                except Exception as e:
                    print(f"    frame {idx} ({fr['path']}): {type(e).__name__}: {e}")
                    continue
                if (idx + 1) % max(1, n_frames // 10) == 0:
                    elapsed = _time.time() - _t_render_start
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_frames - idx - 1) / rate if rate > 0 else 0
                    print(f"    ... {idx+1}/{n_frames} frames "
                          f"({rate:.1f} fr/s, ETA {eta:.0f}s)", flush=True)
            _t_render = _time.time() - _t_render_start
            print(f"  Rendered {n_frames} frames in {_t_render:.1f}s "
                  f"({n_frames/_t_render:.1f} fr/s)", flush=True)

            # Stitch PNGs into video via shared helper
            saved, _ = _stitch_pngs_to_video(
                _FFMPEG, tmpdir, out_path1_base, args.fps,
                _ENCODER_CANDIDATES)
            # If ffmpeg unavailable or all encoders failed: pillow GIF
            if not saved:
                saved, _ = _stitch_pngs_to_gif(tmpdir, out_path1_base, args.fps)

    if not saved:
        print("  ERROR: could not save Plot 1 animation")



    # ======================================================================
    # PLOT 2: animated saturation profiles vs z at (x_src, y_src)
    # ======================================================================
    print("\nBuilding Plot 2 (saturation profiles animation, streaming) ...", flush=True)


    geom_p2 = {
        "ix_src": ix_src, "iy_src": iy_src,
        "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
    }

    out_path2_base = os.path.join(args.outdir, "anim_profile_xy")
    saved2 = False
    _expected_npart = Nx * Ny * Nz

    if _use_parallel:
        with tempfile.TemporaryDirectory(prefix="sph_p2_", dir=args.outdir) as tmpdir2:
            tasks2 = []
            for idx, fr in enumerate(frames):
                png_path = os.path.join(tmpdir2, f"frame_{idx:06d}.png")
                tasks2.append((idx, fr["path"], fr["fmt"], _expected_npart,
                                has_napl_q, _lex_perm,
                                geom_p2, png_path, args.dpi))

            print(f"  Rendering {n_frames} frames with {_N_WORKERS} workers "
                  f"(start method: {args.mp_start_method}) ...", flush=True)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from concurrent.futures.process import BrokenProcessPool
            try:
                ctx = _mp.get_context(args.mp_start_method)
            except ValueError:
                ctx = _mp.get_context("spawn")

            import time as _time
            _t_render_start = _time.time()
            completed2 = 0
            _broken2 = False
            try:
                with ProcessPoolExecutor(max_workers=_N_WORKERS, mp_context=ctx) as pool:
                    futures = [pool.submit(_render_plot2_frame_worker, t) for t in tasks2]
                    for fut in as_completed(futures):
                        try:
                            idx = fut.result()
                            completed2 += 1
                            if completed2 % max(1, n_frames // 10) == 0:
                                elapsed = _time.time() - _t_render_start
                                rate = completed2 / elapsed if elapsed > 0 else 0
                                eta = (n_frames - completed2) / rate if rate > 0 else 0
                                print(f"    ... {completed2}/{n_frames} frames "
                                      f"({rate:.1f} fr/s, ETA {eta:.0f}s)", flush=True)
                        except BrokenProcessPool as e:
                            if not _broken2:
                                print(f"\n    *** WORKER POOL BROKEN (Plot 2): {e}")
                                _broken2 = True
                        except Exception as e:
                            print(f"    WORKER FAILED: {type(e).__name__}: {e}")
                if _broken2:
                    _use_parallel_p2 = False
                else:
                    _t_render = _time.time() - _t_render_start
                    print(f"  Rendered {completed2}/{n_frames} frames in "
                          f"{_t_render:.1f}s ({completed2/_t_render:.1f} fr/s)",
                          flush=True)
                    # Stitch via shared helper (same encoder fallback chain
                    # as Plot 1 — keeps both animations in sync)
                    saved2, _ = _stitch_pngs_to_video(
                        _FFMPEG, tmpdir2, out_path2_base, args.fps,
                        _ENCODER_CANDIDATES)
            except Exception as e:
                print(f"  Plot 2 parallel error: {e}")

    if not saved2:
        # Streaming serial fallback for Plot 2
        print(f"  Rendering {n_frames} frames serially ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="sph_p2_", dir=args.outdir) as tmpdir2:
            import time as _time
            _t_render_start = _time.time()
            for idx, fr in enumerate(frames):
                png_path = os.path.join(tmpdir2, f"frame_{idx:06d}.png")
                task = (idx, fr["path"], fr["fmt"], _expected_npart,
                        has_napl_q, _lex_perm,
                        geom_p2, png_path, args.dpi)
                try:
                    _render_plot2_frame_worker(task)
                except Exception as e:
                    print(f"    frame {idx} ({fr['path']}): {type(e).__name__}: {e}")
                    continue
                if (idx + 1) % max(1, n_frames // 10) == 0:
                    elapsed = _time.time() - _t_render_start
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_frames - idx - 1) / rate if rate > 0 else 0
                    print(f"    ... {idx+1}/{n_frames} frames "
                          f"({rate:.1f} fr/s, ETA {eta:.0f}s)", flush=True)
            _t_render = _time.time() - _t_render_start
            print(f"  Rendered {n_frames} frames in {_t_render:.1f}s", flush=True)

            # Stitch via shared helper
            saved2, _ = _stitch_pngs_to_video(
                _FFMPEG, tmpdir2, out_path2_base, args.fps,
                _ENCODER_CANDIDATES)
            if not saved2:
                saved2, _ = _stitch_pngs_to_gif(tmpdir2, out_path2_base, args.fps)

    if not saved2:
        print("  ERROR: could not save Plot 2 animation")


    # ======================================================================
    # STATIC FIGURE 1: 2x2 slice panels at 4 time points
    # ======================================================================
    print("\nBuilding static Figure 1 (slice panels at 4 time points) ...", flush=True)

    _panel_indices = _pick_panel_indices(n_frames)
    _panel_times_str = ", ".join(f"{frames[i]['time']:.1f}s" for i in _panel_indices)
    print(f"  Selected frames: {_panel_indices}  (t = {_panel_times_str})")

    # 2x2 layout with each panel containing the same YZ+XZ slice pair as Plot 1.
    # Each panel is a sub-figure: we need a grid of 4 (rows) x 2 (cols) actually,
    # or 2x2 with each cell holding two side-by-side subplots. The latter is
    # what fits the description "4 panels each representing 4 moments". I use
    # a 4x2 grid with shared row spacing — looks like 4 panels stacked.
    # Actually to match "4 panels": a 2x2 layout where each panel is a single
    # composite (YZ+XZ stacked vertically) makes the most sense visually.
    # 4 rows (one per time point) x 2 cols (YZ slice | XZ slice).
    # Each row's leftmost ylabel carries the panel time, so the time is
    # both immediately visible and naturally aligned with its row.
    n_panels = len(_panel_indices[:4])
    fig_static1, axes_static1 = plt.subplots(
        n_panels, 2, figsize=(13, 4.5 * n_panels), constrained_layout=True)
    if n_panels == 1:
        axes_static1 = np.atleast_2d(axes_static1)

    fig_static1.suptitle(
        "Saturation slices at 4 time points\n"
        "(R=Sn, G=Sa, B=Sw)  +  water (blue) / NAPL (red) streamlines",
        fontsize=13, fontweight="bold")

    for panel_i, idx in enumerate(_panel_indices[:n_panels]):
        ax_yz = axes_static1[panel_i, 0]
        ax_xz = axes_static1[panel_i, 1]
        fr = frames[idx]
        try:
            snap = read_one_snapshot(fr["path"], fr["fmt"], _expected_npart,
                                      has_napl_q, _lex_perm)
        except Exception as e:
            ax_yz.text(0.5, 0.5, f"Failed to read\n{fr['path']}\n{e}",
                       ha="center", va="center", transform=ax_yz.transAxes)
            continue
        setup_axes_plot1(ax_yz, ax_xz, geom)
        fig_static1.canvas.draw()
        _draw_plot1_into_axes(ax_yz, ax_xz, fig_static1, snap, geom,
                               args.alpha, args.stream_threshold,
                               args.stream_density, args.stream_grid,
                               not args.no_streamlines)
        # _draw_plot1_into_axes sets a per-frame suptitle on the figure;
        # for the static figure we want our own overall suptitle instead,
        # so re-set it after each draw (the last write wins, but they
        # all set the same text).
        fig_static1.suptitle(
            "Saturation slices at 4 time points\n"
            "(R=Sn, G=Sa, B=Sw)  +  water (blue) / NAPL (red) streamlines",
            fontsize=13, fontweight="bold")
        # Compact per-axes titles. Encode time in the ylabel of the left
        # panel so it acts as a row label.
        ax_yz.set_title(f"YZ at x={x_axis[ix_src]:.2f} m", fontsize=10)
        ax_xz.set_title(f"XZ at y={y_axis[iy_src]:.2f} m", fontsize=10)
        ax_yz.set_ylabel(
            f"step {snap['step']}\nt = {format_time(snap['time'])}\n\nz [m]",
            fontsize=10)
        del snap

    out_static1 = os.path.join(args.outdir, "static_slices_4panels.png")
    fig_static1.savefig(out_static1, dpi=args.dpi)
    plt.close(fig_static1)
    print(f"  Saved {out_static1}")


    # ======================================================================
    # STATIC FIGURE 2: 2x2 saturation profile panels at 4 time points
    # ======================================================================
    print("\nBuilding static Figure 2 (profile panels at 4 time points) ...", flush=True)

    fig_static2, axes2 = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    fig_static2.suptitle(
        f"Saturation profiles vs z at (x={x_axis[ix_src]:.2f}, "
        f"y={y_axis[iy_src]:.2f}) m  —  4 time points", fontsize=13)

    for panel_i, idx in enumerate(_panel_indices[:4]):
        ax = axes2.flat[panel_i]
        fr = frames[idx]
        try:
            snap = read_one_snapshot(fr["path"], fr["fmt"], _expected_npart,
                                      has_napl_q, _lex_perm)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to read\n{fr['path']}\n{e}",
                     ha="center", va="center", transform=ax.transAxes)
            continue
        _draw_plot2_into_axes(ax, snap, geom_p2)
        ax.set_title(f"step {snap['step']:>6d},  "
                      f"t = {format_time(snap['time'])}",
                      fontsize=11)
        del snap

    # Fill any unused panels with a blank message (only when n_frames < 4)
    for panel_i in range(len(_panel_indices), 4):
        ax = axes2.flat[panel_i]
        ax.text(0.5, 0.5, "(not enough frames)", ha="center", va="center",
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    out_static2 = os.path.join(args.outdir, "static_profiles_4panels.png")
    fig_static2.savefig(out_static2, dpi=args.dpi)
    plt.close(fig_static2)
    print(f"  Saved {out_static2}")


    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")
    print(f"Outputs in: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
