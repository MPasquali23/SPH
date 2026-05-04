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
parser.add_argument("--alpha", type=float, default=0.7,
                    help="Marker transparency (0=invisible, 1=opaque). "
                         "Lower values reveal phase mixing more clearly. Default 0.7")
parser.add_argument("--stream-threshold", type=float, default=0.02,
                    help="Mask streamlines below this fraction of max |v| in each slice. "
                         "Default 0.02 (2%%); set to 0 to disable masking.")
args = parser.parse_args()

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


fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

# Coordinate flat arrays for scatter (constant across frames)
Y_yz, Z_yz = np.meshgrid(y_axis, z_axis, indexing="ij")    # (Ny, Nz)
X_xz, Z_xz = np.meshgrid(x_axis, z_axis, indexing="ij")    # (Nx, Nz)
yz_y_flat = Y_yz.ravel(); yz_z_flat = Z_yz.ravel()
xz_x_flat = X_xz.ravel(); xz_z_flat = Z_xz.ravel()


def setup_axes_plot1(ax_yz, ax_xz):
    ax_yz.set_xlim(0, Ly); ax_yz.set_ylim(0, Lz); ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y [m]"); ax_yz.set_ylabel("z [m]")
    ax_yz.set_facecolor("#f4f4f4")     # neutral grey so gaps between particles read as 'between'
    ax_xz.set_xlim(0, Lx); ax_xz.set_ylim(0, Lz); ax_xz.set_aspect("equal")
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


# Force one draw to give the axes a real window extent so marker_size_for works.
setup_axes_plot1(ax1a, ax1b)
fig1.canvas.draw()
size_yz = marker_size_for(ax1a, dy)    # YZ slice: marker spans dy in y-direction
size_xz = marker_size_for(ax1b, dx)    # XZ slice: marker spans dx in x-direction


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
    rgb_yz = rgb_from(Sw_yz, Sn_yz, Sa_yz).reshape(-1, 3)
    ax1a.scatter(yz_y_flat, yz_z_flat, c=rgb_yz, marker="o",
                 s=size_yz, edgecolors="none", alpha=args.alpha)

    qy_yz_w = slice_yz(qy3, ix_src); qz_yz_w = slice_yz(qz3, ix_src)
    qy_yz_w_m, qz_yz_w_m = mask_low_speed(qy_yz_w, qz_yz_w, args.stream_threshold)
    if np.nanmax(np.abs(qy_yz_w_m) + np.abs(qz_yz_w_m)) > 1e-25:
        ax1a.streamplot(y_axis, z_axis, qy_yz_w_m.T, qz_yz_w_m.T,
                         color="blue", linewidth=0.9, density=1.2,
                         arrowsize=1.0, arrowstyle="->")
    if qyn3 is not None:
        qy_yz_n = slice_yz(qyn3, ix_src); qz_yz_n = slice_yz(qzn3, ix_src)
        qy_yz_n_m, qz_yz_n_m = mask_low_speed(qy_yz_n, qz_yz_n, args.stream_threshold)
        if np.nanmax(np.abs(qy_yz_n_m) + np.abs(qz_yz_n_m)) > 1e-25:
            ax1a.streamplot(y_axis, z_axis, qy_yz_n_m.T, qz_yz_n_m.T,
                             color="red", linewidth=0.9, density=1.2,
                             arrowsize=1.0, arrowstyle="->")

    ax1a.add_patch(Rectangle((SRC_Y0, SRC_Z0), SRC_Y1-SRC_Y0, SRC_Z1-SRC_Z0,
                              lw=2, ec="k", fc="none", ls="--"))
    ax1a.set_title(f"YZ slice at x={x_axis[ix_src]:.2f} m")

    # ── XZ slice at y = y_axis[iy_src] ──
    Sw_xz = slice_xz(Sw3, iy_src); Sn_xz = slice_xz(Sn3, iy_src); Sa_xz = slice_xz(Sa3, iy_src)
    rgb_xz = rgb_from(Sw_xz, Sn_xz, Sa_xz).reshape(-1, 3)
    ax1b.scatter(xz_x_flat, xz_z_flat, c=rgb_xz, marker="o",
                 s=size_xz, edgecolors="none", alpha=args.alpha)

    qx_xz_w = slice_xz(qx3, iy_src); qz_xz_w = slice_xz(qz3, iy_src)
    qx_xz_w_m, qz_xz_w_m = mask_low_speed(qx_xz_w, qz_xz_w, args.stream_threshold)
    if np.nanmax(np.abs(qx_xz_w_m) + np.abs(qz_xz_w_m)) > 1e-25:
        ax1b.streamplot(x_axis, z_axis, qx_xz_w_m.T, qz_xz_w_m.T,
                         color="blue", linewidth=0.9, density=1.5,
                         arrowsize=1.0, arrowstyle="->")
    if qxn3 is not None:
        qx_xz_n = slice_xz(qxn3, iy_src); qz_xz_n = slice_xz(qzn3, iy_src)
        qx_xz_n_m, qz_xz_n_m = mask_low_speed(qx_xz_n, qz_xz_n, args.stream_threshold)
        if np.nanmax(np.abs(qx_xz_n_m) + np.abs(qz_xz_n_m)) > 1e-25:
            ax1b.streamplot(x_axis, z_axis, qx_xz_n_m.T, qz_xz_n_m.T,
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
