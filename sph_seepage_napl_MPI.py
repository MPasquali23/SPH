#!/usr/bin/env python3
"""
SPH framework for steady-state seepage flow through a 10 m × 10 m aquifer
==========================================================================

Based on:
  Lian, Bui, Nguyen, Tran & Haque (2021)
  "A general SPH framework for transient seepage flows through
   unsaturated porous media considering anisotropic diffusion"
  Comput. Methods Appl. Mech. Engrg. 387, 114169.

Physics
-------
* Two-phase: water (wetting) + air (non-wetting) in rigid porous medium.
* Prescribed hydraulic head  H_u = 8 m (left)  and  H_d = 6 m (right).
* Impermeable (no-flux) top and bottom boundaries.
* Isotropic permeability  =>  corrected Laplacian Eq. 38 is sufficient.

Governing equation  (paper Eq. 29, rigid skeleton, v_s = 0, no gas phase):

    dh/dt  =  (1 / C_tilde_Sr) * div[ k * grad(H) ]          H = h + z

Darcy velocity (specific discharge):

    q  =  -k * grad(H)                                 (negative sign!)

The code integrates from an *approximate* initial condition until the true
steady state  div[k * grad(H)] ~ 0  is reached.  Convergence is monitored via
the L2-norm of dh/dt, which must *plateau* at a small residual once the
true steady state is attained.

Output
------
* One HDF5 (or NPZ) file per snapshot, in OUTDIR/snapshots/step_NNNNNNNNNNNN.{h5,npz}.
  This avoids the metadata-cache / mmap bus-error problems that occur on
  parallel filesystems (Lustre/GPFS) with single multi-GB HDF5 files.
* Matplotlib diagnostic figures.
"""

import numpy as np
import os
import sys
import time as wall_time
import argparse


# ======================================================================
# COMMAND-LINE ARGUMENTS
# ======================================================================
# All physical/numerical parameters can be overridden from the command line.
# Defaults match the original hard-coded values.

_parser = argparse.ArgumentParser(
    description="3D SPH three-phase seepage with NAPL source",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# --- Grid & domain ---
_g_grid = _parser.add_argument_group("Grid & domain")
_g_grid.add_argument("--nx", type=int, default=51, help="Particles in x")
_g_grid.add_argument("--ny", type=int, default=51, help="Particles in y")
_g_grid.add_argument("--nz", type=int, default=51, help="Particles in z (gravity)")
_g_grid.add_argument("--lx", type=float, default=10.0, help="Domain length x [m]")
_g_grid.add_argument("--ly", type=float, default=10.0, help="Domain length y [m]")
_g_grid.add_argument("--lz", type=float, default=10.0, help="Domain length z [m]")

# --- Boundary conditions ---
_g_bc = _parser.add_argument_group("Water Dirichlet BCs")
_g_bc.add_argument("--h-left",  type=float, default=8.0, help="Hydraulic head H at x=0 [m]")
_g_bc.add_argument("--h-right", type=float, default=6.0, help="Hydraulic head H at x=Lx [m]")

# --- NAPL source ---
_g_src = _parser.add_argument_group("NAPL source (cuboid)")
_g_src.add_argument("--src-x0", type=float, default=4.5, help="Source x lower bound [m]")
_g_src.add_argument("--src-x1", type=float, default=5.5, help="Source x upper bound [m]")
_g_src.add_argument("--src-y0", type=float, default=4.5, help="Source y lower bound [m]")
_g_src.add_argument("--src-y1", type=float, default=5.5, help="Source y upper bound [m]")
_g_src.add_argument("--src-z0", type=float, default=8.5, help="Source z lower bound [m]")
_g_src.add_argument("--src-z1", type=float, default=9.5, help="Source z upper bound [m]")
_g_src.add_argument("--sn-source", type=float, default=0.80,
                     help="NAPL saturation at source (held fixed)")

# --- Time integration ---
_g_time = _parser.add_argument_group("Time integration")
_g_time.add_argument("--n-steps", type=int, default=2000,
                      help="Maximum number of time steps")
_g_time.add_argument("--cfl", type=float, default=0.25, help="CFL factor for stable dt")

# --- I/O ---
_g_io = _parser.add_argument_group("I/O cadence")
_g_io.add_argument("--snapshot-every", type=int, default=200,
                    help="HDF5 snapshot every N steps")
_g_io.add_argument("--ckpt-every", type=int, default=500,
                    help="Checkpoint every N steps")
_g_io.add_argument("--print-every", type=int, default=100,
                    help="Console log every N steps")
_g_io.add_argument("--sn-snap-every", type=int, default=0,
                    help="In-memory Sn snapshot every N steps for animation. "
                         "0 = auto (n_steps/40)")

# --- Output paths ---
_g_path = _parser.add_argument_group("Output paths")
_g_path.add_argument("--outdir", type=str, default=".",
                      help="Directory for HDF5/NPZ snapshots, checkpoints, figures")
_g_path.add_argument("--snapshot-file", type=str, default="sph_napl_snapshots.h5",
                      help="DEPRECATED, ignored. Snapshots now go to OUTDIR/snapshots/ "
                           "as one file per step (step_NNNNNNNNNNNN.h5 or .npz). "
                           "This argument is accepted for backward CLI compatibility "
                           "but has no effect.")

# MPI / multi-node decomposition.
# MPI mode is activated automatically when launched with mpirun -np N (N > 1).
# No --use-mpi flag: launching method is the switch. Options below only tune
# the 2D pencil decomposition layout (Px × Py = N_ranks).
_g_mpi = _parser.add_argument_group(
    "MPI / multi-node (active when launched via mpirun -np N, N > 1)")
_g_mpi.add_argument("--mpi-px", type=int, default=0,
                     help="Number of MPI ranks along x. 0 = auto. "
                          "Px * Py must equal total rank count.")
_g_mpi.add_argument("--mpi-py", type=int, default=0,
                     help="Number of MPI ranks along y. 0 = auto.")
_g_mpi.add_argument("--mpi-validate", action="store_true",
                     help="Run extra consistency checks on ghost exchange "
                          "(position-match after roundtrip). Slow; for "
                          "first-run debugging only.")

args = _parser.parse_args()

# Validation
if args.src_x0 >= args.src_x1 or args.src_y0 >= args.src_y1 or args.src_z0 >= args.src_z1:
    sys.exit("ERROR: source bbox bounds must satisfy X0<X1, Y0<Y1, Z0<Z1")
if not (0.0 < args.sn_source <= 1.0):
    sys.exit("ERROR: --sn-source must be in (0, 1]")
if args.nx < 3 or args.ny < 3 or args.nz < 3:
    sys.exit("ERROR: grid sizes must be at least 3 (need interior particles)")
if args.lx <= 0 or args.ly <= 0 or args.lz <= 0:
    sys.exit("ERROR: domain dimensions must be positive")
if not (args.src_x1 <= args.lx and args.src_y1 <= args.ly and args.src_z1 <= args.lz):
    sys.exit("ERROR: source bbox extends outside domain")
if args.src_x0 < 0 or args.src_y0 < 0 or args.src_z0 < 0:
    sys.exit("ERROR: source bbox cannot have negative bounds")


# ======================================================================
# MPI INITIALIZATION (transparent when launched single-process)
# ======================================================================
# MPI mode activates automatically when the script is launched with
# mpirun / srun and the resulting communicator has size > 1. In single-
# process mode (no mpirun, or mpirun -np 1, or mpi4py not installed), all
# MPI helpers reduce to trivial no-ops and behaviour is bit-identical to
# the original single-process code.

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    MPI_RANK = _comm.Get_rank()
    MPI_NRANK = _comm.Get_size()
    HAS_MPI = True
except ImportError:
    _comm = None
    MPI_RANK = 0
    MPI_NRANK = 1
    HAS_MPI = False

USE_MPI = HAS_MPI and MPI_NRANK > 1


def log_root(msg="", **kwargs):
    """Print only on rank 0 (suppress on other ranks)."""
    if MPI_RANK == 0:
        print(msg, **kwargs)


def log_all(msg="", **kwargs):
    """Print on every rank, prefixed with [rank/nrank]."""
    print(f"[rank {MPI_RANK}/{MPI_NRANK}] {msg}", **kwargs)


def _factor_2d(n, prefer_aspect=1.0):
    """Pick (Px, Py) with Px*Py = n and aspect Px:Py ~ prefer_aspect."""
    best, best_score = None, None
    for px in range(1, int(np.sqrt(n)) + 2):
        if n % px != 0:
            continue
        py = n // px
        score = abs(np.log(px / py) - np.log(prefer_aspect))
        if best is None or score < best_score:
            best, best_score = (px, py), score
    return best


if USE_MPI:
    # Resolve (Px, Py) — user can override via --mpi-px / --mpi-py
    if args.mpi_px > 0 and args.mpi_py > 0:
        if args.mpi_px * args.mpi_py != MPI_NRANK:
            if MPI_RANK == 0:
                print(f"ERROR: --mpi-px ({args.mpi_px}) * --mpi-py "
                      f"({args.mpi_py}) = {args.mpi_px * args.mpi_py} "
                      f"!= MPI rank count {MPI_NRANK}")
            _comm.Abort(1)
        Px, Py = args.mpi_px, args.mpi_py
    elif args.mpi_px > 0:
        if MPI_NRANK % args.mpi_px != 0:
            if MPI_RANK == 0:
                print(f"ERROR: --mpi-px ({args.mpi_px}) does not divide "
                      f"rank count ({MPI_NRANK})")
            _comm.Abort(1)
        Px, Py = args.mpi_px, MPI_NRANK // args.mpi_px
    elif args.mpi_py > 0:
        if MPI_NRANK % args.mpi_py != 0:
            if MPI_RANK == 0:
                print(f"ERROR: --mpi-py ({args.mpi_py}) does not divide "
                      f"rank count ({MPI_NRANK})")
            _comm.Abort(1)
        Py, Px = args.mpi_py, MPI_NRANK // args.mpi_py
    else:
        # Auto: match domain aspect ratio
        Px, Py = _factor_2d(MPI_NRANK, prefer_aspect=args.lx / args.ly)

    PX_IDX = MPI_RANK %  Px
    PY_IDX = MPI_RANK // Px

    # Owned global (i, j) ranges  [IX_LO, IX_HI) × [IY_LO, IY_HI)
    _ix_split = np.linspace(0, args.nx, Px + 1, dtype=np.int64)
    _iy_split = np.linspace(0, args.ny, Py + 1, dtype=np.int64)
    IX_LO = int(_ix_split[PX_IDX]); IX_HI = int(_ix_split[PX_IDX + 1])
    IY_LO = int(_iy_split[PY_IDX]); IY_HI = int(_iy_split[PY_IDX + 1])

    # Ghost width: SPH kernel support is 2*h_sml = 2.6 dx → need 3 cells
    GHOST_W = 3

    # Extended ranges including ghosts (clipped at global domain edges)
    IX_GL = max(0,       IX_LO - GHOST_W)
    IX_GR = min(args.nx, IX_HI + GHOST_W)
    IY_GF = max(0,       IY_LO - GHOST_W)
    IY_GB = min(args.ny, IY_HI + GHOST_W)

    def _rank_of(px, py):
        if px < 0 or px >= Px or py < 0 or py >= Py:
            return MPI.PROC_NULL
        return py * Px + px

    NB_L = _rank_of(PX_IDX - 1, PY_IDX)
    NB_R = _rank_of(PX_IDX + 1, PY_IDX)
    NB_F = _rank_of(PX_IDX,     PY_IDX - 1)
    NB_B = _rank_of(PX_IDX,     PY_IDX + 1)
else:
    # Single-process: rank 0 owns everything; no ghosts
    Px = Py = 1
    PX_IDX = PY_IDX = 0
    IX_LO, IX_HI = 0, args.nx
    IY_LO, IY_HI = 0, args.ny
    IX_GL, IX_GR = IX_LO, IX_HI
    IY_GF, IY_GB = IY_LO, IY_HI
    GHOST_W = 0
    NB_L = NB_R = NB_F = NB_B = -1  # unused


def comm_allreduce_sum(x):
    """Sum a scalar / numpy array across ranks. No-op in single-process."""
    return _comm.allreduce(x, op=MPI.SUM) if USE_MPI else x

def comm_allreduce_max(x):
    return _comm.allreduce(x, op=MPI.MAX) if USE_MPI else x

def comm_allreduce_min(x):
    return _comm.allreduce(x, op=MPI.MIN) if USE_MPI else x

log_root(f"\n{'='*70}")
log_root(f"  3D SPH Three-Phase Seepage — configuration")
log_root(f"{'='*70}")
log_root(f"  Grid           : {args.nx} x {args.ny} x {args.nz} = "
         f"{args.nx*args.ny*args.nz} particles")
log_root(f"  Domain         : {args.lx} x {args.ly} x {args.lz} m")
log_root(f"  Water BCs      : H_left = {args.h_left} m,  H_right = {args.h_right} m")
log_root(f"  NAPL source    : x[{args.src_x0},{args.src_x1}]  "
         f"y[{args.src_y0},{args.src_y1}]  z[{args.src_z0},{args.src_z1}]  "
         f"Sn={args.sn_source}")
log_root(f"  Time           : up to {args.n_steps} steps, CFL={args.cfl}")
log_root(f"  I/O            : snapshot every {args.snapshot_every},  "
         f"ckpt every {args.ckpt_every},  print every {args.print_every}")
log_root(f"  Output         : {os.path.abspath(args.outdir)}")
if USE_MPI:
    log_root(f"  MPI mode       : {MPI_NRANK} ranks  ({Px}×{Py} pencil, "
             f"ghost = {GHOST_W} cells)")
else:
    log_root(f"  MPI mode       : single-process "
             f"(launch with `mpirun -np N` to enable)")
log_root(f"{'='*70}")

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
    log_root("Numba available — JIT-accelerated particle loops enabled.")
except ImportError:
    HAS_NUMBA = False
    log_root("WARNING: Numba not available — falling back to pure Python loops.")

# ======================================================================
# 0.  OUTPUT DIRECTORY
# ======================================================================
OUTPUT_DIR = args.outdir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================
# 1.  PHYSICAL PARAMETERS
# ======================================================================
k_abs   = 2.059e-11   # absolute (intrinsic) permeability  [m^2]
phi_0   = 0.43        # porosity  [-]
c_R     = 4.35e-7     # rock compressibility  [1/Pa]
rho_W   = 1000.0      # water density  [kg/m^3]
rho_A   = 1.225       # air density  [kg/m^3]
mu_w    = 1.0e-3      # water dynamic viscosity  [Pa*s]
mu_a    = 1.8e-5      # air dynamic viscosity  [Pa*s]
g_acc   = 9.81        # gravitational acceleration  [m/s^2]
n_vG    = 2.68        # Van Genuchten  n
p_caw0  = 676.55      # characteristic capillary pressure  [Pa]
p_0     = 101325.0    # atmospheric pressure  [Pa]

# Derived
gamma_w = rho_W * g_acc                       # specific weight of water
k_sat   = k_abs * rho_W * g_acc / mu_w        # saturated hydraulic conductivity [m/s]
m_vG    = 1.0 - 1.0 / n_vG                    # Van Genuchten m
g_a     = gamma_w / p_caw0                     # alpha_vG  [1/m]

S_sat   = 1.0
S_res   = 0.045
g_l     = 0.5          # Mualem pore-connectivity parameter

# Specific storage  (paper Eq. 30)
K_s       = 1.0 / c_R           # bulk modulus of skeleton
K_sat_l   = 2.0e9               # bulk modulus of water  [Pa]
alpha_hat = (1.0 - phi_0) / K_s
beta_hat  = 1.0 / K_sat_l
C_l       = gamma_w * (phi_0 * S_sat * beta_hat + alpha_hat)

print(f"k_sat              = {k_sat:.6e} m/s")
print(f"Van Genuchten:  n  = {n_vG:.3f},  m = {m_vG:.4f},  alpha = {g_a:.4f} 1/m")
print(f"Specific storage   = {C_l:.6e} 1/m")

# ======================================================================
# 1b. NAPL (LNAPL) PHYSICAL PARAMETERS  — diesel oil
# ======================================================================
rho_N    = 830.0        # NAPL density  [kg/m^3]
mu_n     = 3.61e-3      # NAPL dynamic viscosity  [Pa*s]
gamma_n  = rho_N * g_acc
k_sat_n  = k_abs * rho_N * g_acc / mu_n      # NAPL hydraulic conductivity

# Parker-Lenhard capillary scaling  (same VG shape, scaled alpha)
#   beta_nw  = sigma_aw / sigma_nw    (interfacial tension ratio)
#   alpha_nw = beta_nw * alpha_aw     (VG alpha for NAPL-water curve)
sigma_aw = 0.065        # air-water surface tension  [N/m]
sigma_nw = 0.030        # NAPL-water surface tension  [N/m]
beta_nw  = sigma_aw / sigma_nw                # ~ 2.167
alpha_nw = beta_nw * g_a                      # [1/m]  (g_a = alpha_aw)

print(f"k_sat_n (NAPL)     = {k_sat_n:.6e} m/s")
print(f"rho_N = {rho_N:.0f} kg/m^3,  mu_n = {mu_n:.4e} Pa*s")
print(f"sigma_aw = {sigma_aw}  sigma_nw = {sigma_nw}  beta_nw = {beta_nw:.4f}")
print(f"alpha_aw = {g_a:.4f}  alpha_nw = {alpha_nw:.4f}  1/m")

# ======================================================================
# 2.  DOMAIN  &  SPH DISCRETISATION   (3D — z is the gravity axis)
# ======================================================================
Lx, Ly, Lz = args.lx, args.ly, args.lz

Nx = args.nx       # particles in x  (horizontal flow direction)
Ny = args.ny       # particles in y  (horizontal, perpendicular to flow)
Nz = args.nz       # particles in z  (vertical, gravity direction)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

log_root(f"\nDomain  {Lx} x {Ly} x {Lz} m    grid  {Nx} x {Ny} x {Nz}    "
         f"dx = {dx:.4f}  dy = {dy:.4f}  dz = {dz:.4f} m")

# ----------------------------------------------------------------------
# Local particle generation with optional 2D pencil decomposition.
#
# Local-array layout per rank (concatenated, in this order):
#   Block 0 (OWNED):      i in [IX_LO, IX_HI), j in [IY_LO, IY_HI), k in [0, Nz)
#   Block 1 (X_LEFT  gh): i in [IX_GL, IX_LO), j in [IY_LO, IY_HI), k in [0, Nz)
#   Block 2 (X_RIGHT gh): i in [IX_HI, IX_GR), j in [IY_LO, IY_HI), k in [0, Nz)
#   Block 3 (Y_FRONT gh): i in [IX_GL, IX_GR), j in [IY_GF, IY_LO), k in [0, Nz)
#   Block 4 (Y_BACK  gh): i in [IX_GL, IX_GR), j in [IY_HI, IY_GB), k in [0, Nz)
#
# In single-process mode blocks 1..4 are all empty.
# Blocks 3 & 4 use the EXTENDED x-range so that the two-pass ghost
# exchange (x first, then y) propagates corner data without explicit
# diagonal MPI calls.
# ----------------------------------------------------------------------

def _block_indices(i_range, j_range, k_range):
    """Cartesian product of three index ranges. Returns 1-D arrays of i,j,k."""
    ii, jj, kk = np.meshgrid(i_range, j_range, k_range, indexing="ij")
    return ii.ravel(), jj.ravel(), kk.ravel()


_blocks_def = [
    (np.arange(IX_LO, IX_HI), np.arange(IY_LO, IY_HI)),   # owned
    (np.arange(IX_GL, IX_LO), np.arange(IY_LO, IY_HI)),   # x_left  ghost
    (np.arange(IX_HI, IX_GR), np.arange(IY_LO, IY_HI)),   # x_right ghost
    (np.arange(IX_GL, IX_GR), np.arange(IY_GF, IY_LO)),   # y_front ghost (extended x)
    (np.arange(IX_GL, IX_GR), np.arange(IY_HI, IY_GB)),   # y_back  ghost (extended x)
]
_k_full = np.arange(0, Nz)

# Generate each block; record sizes
_block_ii, _block_jj, _block_kk, _block_sizes = [], [], [], []
for (bi, bj) in _blocks_def:
    if len(bi) == 0 or len(bj) == 0:
        _block_sizes.append(0)
        continue
    ii, jj, kk = _block_indices(bi, bj, _k_full)
    _block_ii.append(ii); _block_jj.append(jj); _block_kk.append(kk)
    _block_sizes.append(len(ii))

# Concatenated (i,j,k) arrays for all local particles
if _block_ii:
    _all_ii = np.concatenate(_block_ii)
    _all_jj = np.concatenate(_block_jj)
    _all_kk = np.concatenate(_block_kk)
else:
    _all_ii = _all_jj = _all_kk = np.empty(0, dtype=np.int64)

# Physical positions
xp = _all_ii.astype(np.float64) * dx
yp = _all_jj.astype(np.float64) * dy
zp = _all_kk.astype(np.float64) * dz

# Block sizes / offsets
N_OWNED    = _block_sizes[0]
N_GHOST_XL = _block_sizes[1]
N_GHOST_XR = _block_sizes[2]
N_GHOST_YF = _block_sizes[3]
N_GHOST_YB = _block_sizes[4]
N_GHOST    = N_GHOST_XL + N_GHOST_XR + N_GHOST_YF + N_GHOST_YB
N_LOCAL    = N_OWNED + N_GHOST

OFF_OWNED = 0
OFF_XL    = OFF_OWNED + N_OWNED
OFF_XR    = OFF_XL    + N_GHOST_XL
OFF_YF    = OFF_XR    + N_GHOST_XR
OFF_YB    = OFF_YF    + N_GHOST_YF

# (i,j,k) -> local index map.  Used to compute ghost-exchange send buffers.
ijk_to_local = {}
for li in range(N_LOCAL):
    ijk_to_local[(int(_all_ii[li]), int(_all_jj[li]), int(_all_kk[li]))] = li

# Global particle index (i*Ny*Nz + j*Nz + k) for each local particle.
# Used by the snapshot gather to place owned particles at their true
# global positions in the assembled array.
global_idx_local = (_all_ii.astype(np.int64) * (Ny * Nz)
                    + _all_jj.astype(np.int64) * Nz
                    + _all_kk.astype(np.int64))
global_idx_owned = global_idx_local[:N_OWNED].copy()

# Legacy name: N_part now means LOCAL count (all per-particle arrays
# allocated with length N_part still work as before).
N_part = N_LOCAL
N_part_global = Nx * Ny * Nz

# idx_3d: local lookup table from (i,j,k) -> local index; -1 if not in
# this rank's owned+ghost region.  Used by BC mirror-index construction.
idx_3d = np.full((Nx, Ny, Nz), -1, dtype=np.int64)
for (i, j, k), li in ijk_to_local.items():
    idx_3d[i, j, k] = li

Vp     = dx * dy * dz       # particle volume (3-D)
h_sml  = 1.3 * dx           # smoothing length

if USE_MPI:
    log_all(f"Pencil ({PX_IDX},{PY_IDX}): owned={N_OWNED}, ghost={N_GHOST}, "
            f"local total={N_LOCAL}")
log_root(f"Particles (global): {N_part_global}    h_sml = {h_sml:.4f} m")

# ======================================================================
# 3.  CUBIC SPLINE KERNEL  (3-D)  [standard Monaghan]
# ======================================================================
#   alpha_d = 3 / (2 * pi * h^3)   — 3D normalisation
alpha_d = 3.0 / (2.0 * np.pi * h_sml**3)

def W_val(r):
    q = r / h_sml
    if q >= 2.0:
        return 0.0
    if q < 1.0:
        return alpha_d * (2.0/3.0 - q*q + 0.5*q**3)
    return alpha_d * (1.0/6.0) * (2.0 - q)**3

def dW_dr(r):
    """dW/dr  (always <= 0 for r > 0)."""
    q = r / h_sml
    if q < 1e-14 or q >= 2.0:
        return 0.0
    if q < 1.0:
        return alpha_d * (-2.0*q + 1.5*q*q) / h_sml
    return alpha_d * (-0.5) * (2.0 - q)**2 / h_sml

# ======================================================================
# 4.  NEIGHBOUR LISTS  (precomputed - particles are fixed)
# ======================================================================
#
# Convention (matching the paper):
#   r_ji  =  x_j - x_i
#   grad_i W(x_i - x_j)  =  dW/dr * (x_i - x_j)/r   (gradient w.r.t. x_i)
#
# F_hat_ij  =  r_ji . grad_i W_ij  /  |r_ji|^2           [Eq. 37]

support_r = 2.0 * h_sml

print("Building neighbour lists (cKDTree) ...", flush=True)

from scipy.spatial import cKDTree

tree = cKDTree(np.column_stack([xp, yp, zp]))
nbl  = tree.query_ball_tree(tree, r=support_r)      # list of lists

# Each entry: (j, r, x_ji, y_ji, z_ji, F_hat, gradWx, gradWy, gradWz)
neighbours = [[] for _ in range(N_part)]

for i in range(N_part):
    xi, yi, zi = xp[i], yp[i], zp[i]
    for j in nbl[i]:
        if j == i:
            continue
        xji = xp[j] - xi
        yji = yp[j] - yi
        zji = zp[j] - zi
        r   = np.sqrt(xji**2 + yji**2 + zji**2)
        if r < 1e-30:
            continue

        dWr = dW_dr(r)
        gWx = dWr * (-xji) / r
        gWy = dWr * (-yji) / r
        gWz = dWr * (-zji) / r

        dot  = xji * gWx + yji * gWy + zji * gWz
        Fhat = dot / (r * r)

        neighbours[i].append((j, r, xji, yji, zji, Fhat, gWx, gWy, gWz))

print("  done.  Converting to CSR arrays ...", flush=True)

# --- Convert neighbour lists to CSR arrays for Numba ---
_total_nbrs = sum(len(nb) for nb in neighbours)
nbr_j    = np.empty(_total_nbrs, dtype=np.int64)
nbr_Fhat = np.empty(_total_nbrs, dtype=np.float64)
nbr_gWx  = np.empty(_total_nbrs, dtype=np.float64)
nbr_gWy  = np.empty(_total_nbrs, dtype=np.float64)
nbr_gWz  = np.empty(_total_nbrs, dtype=np.float64)
nbr_ptr  = np.empty(N_part + 1, dtype=np.int64)

_offset = 0
for i in range(N_part):
    nbr_ptr[i] = _offset
    for (j, rij, xji, yji, zji, Fhat, gWx, gWy, gWz) in neighbours[i]:
        nbr_j[_offset]    = j
        nbr_Fhat[_offset] = Fhat
        nbr_gWx[_offset]  = gWx
        nbr_gWy[_offset]  = gWy
        nbr_gWz[_offset]  = gWz
        _offset += 1
nbr_ptr[N_part] = _offset
print(f"  CSR: {_total_nbrs} entries, avg {_total_nbrs/N_part:.1f} per particle")

# ======================================================================
# 5.  PRECOMPUTE SPH CORRECTION QUANTITIES
# ======================================================================
#
# Corrected gradient  (Eq. 36):
#   L_ij  =  [ sum_j V_j (x_j-x_i)^m  (grad_i W)^n ]^{-1}
#
# Corrected Laplacian  (Eq. 38):
#   K^norm  = 0.5 * sum_j V_j (x_ji^2 + y_ji^2 + z_ji^2) F_hat_ij
#   err^m   = sum_j V_j r^m_ji F_hat_ij

print("Precomputing correction matrices (3x3) ...", flush=True)

K_norm = np.zeros(N_part)
err_x  = np.zeros(N_part)
err_y  = np.zeros(N_part)
err_z  = np.zeros(N_part)
L_inv  = np.zeros((N_part, 3, 3))

for i in range(N_part):
    Kn = 0.0; ex = 0.0; ey = 0.0; ez = 0.0
    M  = np.zeros((3, 3))
    for (j, rij, xji, yji, zji, Fhat, gWx, gWy, gWz) in neighbours[i]:
        Kn += Vp * (xji**2 + yji**2 + zji**2) * Fhat
        ex += Vp * xji * Fhat
        ey += Vp * yji * Fhat
        ez += Vp * zji * Fhat
        # M_mn = sum_j V_j * r_ji^m * (grad_W)^n
        M[0, 0] += Vp * xji * gWx
        M[0, 1] += Vp * xji * gWy
        M[0, 2] += Vp * xji * gWz
        M[1, 0] += Vp * yji * gWx
        M[1, 1] += Vp * yji * gWy
        M[1, 2] += Vp * yji * gWz
        M[2, 0] += Vp * zji * gWx
        M[2, 1] += Vp * zji * gWy
        M[2, 2] += Vp * zji * gWz

    K_norm[i] = 0.5 * Kn
    err_x[i]  = ex
    err_y[i]  = ey
    err_z[i]  = ez

    # 3x3 inversion via numpy (one-time setup cost)
    try:
        L_inv[i] = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        L_inv[i] = np.eye(3)

print("  done.  (neighbour list freed — using CSR arrays from now on)")
del neighbours

# ======================================================================
# 6.  VAN GENUCHTEN  SWRC + MUALEM   [Paper Eqs. 31-32]
# ======================================================================

def compute_Sr(h):
    """Degree of saturation.  h < 0 => unsaturated."""
    return np.where(
        h >= 0.0,
        S_sat,
        S_res + (S_sat - S_res)
        * (1.0 + (g_a * np.abs(h))**n_vG)**(-m_vG)
    )

def compute_Se(Sr):
    return np.clip((Sr - S_res) / (S_sat - S_res), 0.0, 1.0)

def compute_k(h):
    """Hydraulic conductivity  k(h).  Returns array.

    For h >= 0 (saturated zone) we return k_sat directly — this
    avoids numerical noise from Se^(-1/m) near Se = 1.
    """
    k_arr = np.full_like(h, k_sat)          # saturated default
    mask  = h < 0.0                          # unsaturated particles only
    if np.any(mask):
        Sr_u = compute_Sr(h[mask])
        Se_u = compute_Se(Sr_u)
        Se_u = np.clip(Se_u, 1e-14, 1.0)
        inner = np.clip(1.0 - Se_u**(1.0 / m_vG), 0.0, None)
        kr = Se_u**g_l * (1.0 - inner**m_vG)**2
        k_arr[mask] = k_sat * np.clip(kr, 0.0, 1.0)
    return k_arr

def compute_Cs(h):
    """Specific moisture capacity  C_s = n * dS_r/dh   [Eq. 30]."""
    Cs = np.zeros_like(h)
    mask = h < 0.0
    ah = np.abs(h[mask])
    ahn = (g_a * ah)**n_vG
    dSr = ((S_sat - S_res) * m_vG * n_vG * g_a
           * (g_a * ah)**(n_vG - 1.0)
           / (1.0 + ahn)**(m_vG + 1.0))
    Cs[mask] = phi_0 * dSr
    return Cs


# ======================================================================
# 6b. THREE-PHASE CONSTITUTIVE RELATIONS  (Parker-Lenhard / VG)
# ======================================================================
#
# Parker, Lenhard & Kuppusamy (1987), Water Resour. Res. 23(4).
#
# Both S_ew and S_et follow the Van Genuchten functional form with the
# same shape parameters (n, m) but different alpha:
#
#   S_w  from air-water VG curve applied to h_w, capped at (1 - S_n):
#        S_w = min( S_wir + (1-S_wir) * VG(alpha_aw, h_w),  1 - S_n )
#
#   S_ew = (S_w  - S_wir) / (1 - S_wir)     effective water sat.
#   S_et = (S_w + S_n - S_wir) / (1 - S_wir) effective TOTAL liquid sat.
#          ^^^ from ACTUAL saturations, not from VG(h_w) ^^^
#
# Capillary head (NAPL-water), by inverting VG with alpha_nw:
#   h_cnw = (1/alpha_nw) * [S_ew^(-1/m) - 1]^(1/n)
#
# Relative permeabilities (Mualem-VG, three-phase):
#   kr_w = S_ew^l  *  [1 - (1 - S_ew^(1/m))^m]^2
#   kr_n = (S_et - S_ew)^l
#          * [(1 - S_ew^(1/m))^m  -  (1 - S_et^(1/m))^m]^2
#
# The wetting order is  water > NAPL > air  (water most wetting).

S_wir = S_res   # irreducible (residual) water saturation

# Precompute ratio (used in H_n calculation)
_gamma_ratio = gamma_w / gamma_n
_inv_alpha_nw = 1.0 / alpha_nw
_inv_m = 1.0 / m_vG
_inv_n = 1.0 / n_vG

# ── Scalar Numba constitutive helpers ────────────────────────────────
#
# These are called per-particle inside prange loops.  They access
# module-level scalar constants (k_sat, m_vG, etc.) which Numba
# captures at compile time.  inline='always' eliminates call overhead.

if HAS_NUMBA:
    @njit(inline='always')
    def _s_Sw(hw, sn):
        """Scalar water saturation."""
        if hw >= 0.0:
            sw = S_sat
        else:
            sw = S_wir + (1.0 - S_wir) * (1.0 + (g_a * abs(hw))**n_vG)**(-m_vG)
        cap = 1.0 - sn
        if sw > cap:
            sw = cap
        if sw < S_wir:
            sw = S_wir
        if sw > S_sat:
            sw = S_sat
        return sw

    @njit(inline='always')
    def _s_Sew(sw):
        """Scalar effective water saturation."""
        v = (sw - S_wir) / (1.0 - S_wir)
        if v < 1e-14:  v = 1e-14
        if v > 1.0 - 1e-10:  v = 1.0 - 1e-10
        return v

    @njit(inline='always')
    def _s_Set(sw, sn):
        """Scalar effective total-liquid saturation."""
        v = (sw + sn - S_wir) / (1.0 - S_wir)
        if v < 1e-14:  v = 1e-14
        if v > 1.0 - 1e-10:  v = 1.0 - 1e-10
        return v

    @njit(inline='always')
    def _s_kw(hw, sn):
        """Scalar water hydraulic conductivity."""
        if hw >= 0.0 and sn < 1e-12:
            return k_sat
        sw = _s_Sw(hw, sn)
        sew = _s_Sew(sw)
        inner = 1.0 - sew**_inv_m
        if inner < 0.0: inner = 0.0
        if inner > 1.0: inner = 1.0
        kr = sew**g_l * (1.0 - inner**m_vG)**2
        if kr < 0.0: kr = 0.0
        if kr > 1.0: kr = 1.0
        return k_sat * kr

    @njit(inline='always')
    def _s_kn(hw, sn):
        """Scalar NAPL hydraulic conductivity."""
        sw = _s_Sw(hw, sn)
        sew = _s_Sew(sw)
        st  = _s_Set(sw, sn)
        se_n = st - sew
        if se_n < 1e-14: se_n = 1e-14
        tw = (1.0 - sew**_inv_m)**m_vG
        if tw < 0.0: tw = 0.0
        if tw > 1.0: tw = 1.0
        tt = (1.0 - st**_inv_m)**m_vG
        if tt < 0.0: tt = 0.0
        if tt > 1.0: tt = 1.0
        diff = tw - tt
        if diff < 0.0: diff = 0.0
        kr = se_n**g_l * diff**2
        if kr < 0.0: kr = 0.0
        if kr > 1.0: kr = 1.0
        return k_sat_n * kr

    @njit(inline='always')
    def _s_Hn(hw, sn, z_i):
        """Scalar NAPL hydraulic head (z is gravity coordinate)."""
        sw = _s_Sw(hw, sn)
        sew = _s_Sew(sw)
        base = sew**(-_inv_m) - 1.0
        if base < 0.0: base = 0.0
        if base > 1e12: base = 1e12
        h_cnw = _inv_alpha_nw * base**_inv_n
        return _gamma_ratio * (hw + h_cnw) + z_i

    @njit(inline='always')
    def _s_Ctilde(hw):
        """Scalar total storage coefficient C_tilde = max(C_l + Cs, C_l)."""
        if hw >= 0.0:
            return C_l
        ah = abs(hw)
        ahn = (g_a * ah)**n_vG
        dSr = ((S_sat - S_res) * m_vG * n_vG * g_a
               * (g_a * ah)**(n_vG - 1.0)
               / (1.0 + ahn)**(m_vG + 1.0))
        Cs = phi_0 * dSr
        ct = C_l + Cs
        if ct < C_l:
            ct = C_l
        return ct

    # ── Fused precompute: all 5 constitutive fields in one prange ────
    @njit(cache=True, parallel=True)
    def _precompute_fields(h_w, S_n, zp_arr, N,
                           kw_out, Hw_out, Ct_out, kn_out, Hn_out):
        """Compute kw, Hw, C_tilde, kn, Hn for all particles in parallel.
        z is the gravity coordinate (3D)."""
        for i in prange(N):
            hw_i = h_w[i]
            sn_i = S_n[i]
            kw_out[i] = _s_kw(hw_i, sn_i)
            Hw_out[i] = hw_i + zp_arr[i]
            Ct_out[i] = _s_Ctilde(hw_i)
            kn_out[i] = _s_kn(hw_i, sn_i)
            Hn_out[i] = _s_Hn(hw_i, sn_i, zp_arr[i])


# ── NumPy vectorised constitutive functions (kept for fallback + plots) ─


def compute_Sw_3ph(h_w, S_n):
    """Water saturation from VG(h_w), capped so S_w + S_n <= 1."""
    Sw_vg = np.where(
        h_w >= 0.0, S_sat,
        S_wir + (1.0 - S_wir)
        * (1.0 + (g_a * np.abs(h_w))**n_vG)**(-m_vG)
    )
    Sw = np.minimum(Sw_vg, 1.0 - S_n)
    return np.clip(Sw, S_wir, S_sat)


def compute_Sew(h_w, S_n):
    """Effective water saturation  S_ew = (S_w - S_wir) / (1 - S_wir)."""
    Sw = compute_Sw_3ph(h_w, S_n)
    return np.clip((Sw - S_wir) / (1.0 - S_wir), 1e-14, 1.0 - 1e-10)


def compute_Set(h_w, S_n):
    """Effective total-liquid saturation from ACTUAL saturations.
    S_et = (S_w + S_n - S_wir) / (1 - S_wir)
    """
    Sw = compute_Sw_3ph(h_w, S_n)
    return np.clip((Sw + S_n - S_wir) / (1.0 - S_wir), 1e-14, 1.0 - 1e-10)


def compute_krw_3ph(h_w, S_n):
    """Water relative permeability.
    kr_w = S_ew^l * [1 - (1 - S_ew^(1/m))^m]^2
    """
    Sew = compute_Sew(h_w, S_n)
    inner = np.clip(1.0 - Sew**(1.0 / m_vG), 0.0, 1.0)
    kr = Sew**g_l * (1.0 - inner**m_vG)**2
    return np.clip(kr, 0.0, 1.0)


def compute_krn(h_w, S_n):
    """NAPL relative permeability.
    kr_n = (S_et - S_ew)^l * [(1-S_ew^(1/m))^m - (1-S_et^(1/m))^m]^2
    """
    Sew = compute_Sew(h_w, S_n)
    Set = compute_Set(h_w, S_n)

    Se_n = np.clip(Set - Sew, 1e-14, None)

    term_w = np.clip((1.0 - Sew**(1.0 / m_vG))**m_vG, 0.0, 1.0)
    term_t = np.clip((1.0 - Set**(1.0 / m_vG))**m_vG, 0.0, 1.0)
    diff   = np.clip(term_w - term_t, 0.0, None)

    kr = Se_n**g_l * diff**2
    return np.clip(kr, 0.0, 1.0)


def compute_kw_3ph(h_w, S_n):
    """Water hydraulic conductivity  k_w = k_sat * kr_w.
    For h_w >= 0 and S_n ~ 0, returns k_sat exactly.
    """
    k_arr = k_sat * compute_krw_3ph(h_w, S_n)
    sat_clean = (h_w >= 0.0) & (S_n < 1e-12)
    k_arr[sat_clean] = k_sat
    return k_arr


def compute_kn_field(h_w, S_n):
    """NAPL hydraulic conductivity  k_n = k_sat_n * kr_n."""
    return k_sat_n * compute_krn(h_w, S_n)


def compute_Hn_field(h_w, S_n):
    """NAPL hydraulic head.
    H_n = (gamma_w / gamma_n) * (h_w + h_cnw) + z

    Capillary head (NAPL-water), inverted VG with alpha_nw:
        h_cnw = (1/alpha_nw) * [S_ew^(-1/m) - 1]^(1/n)
    """
    Sew   = compute_Sew(h_w, S_n)

    base  = np.clip(Sew**(-1.0 / m_vG) - 1.0, 0.0, 1e12)
    h_cnw = (1.0 / alpha_nw) * base**(1.0 / n_vG)

    return (gamma_w / gamma_n) * (h_w + h_cnw) + zp


# ======================================================================
# 7.  PARTICLE CLASSIFICATION  &  BOUNDARY CONDITIONS
# ======================================================================
#
#   LEFT   x = 0   :  Dirichlet  H = H_u  = 8 m
#   RIGHT  x = Lx  :  Dirichlet  H = H_d  = 6 m
#   BOTTOM z = 0   :  Impermeable  q.n = 0
#   TOP    z = Lz  :  Impermeable  q.n = 0
#   FRONT  y = 0   :  Impermeable  q.n = 0   (3D no-flux)
#   BACK   y = Ly  :  Impermeable  q.n = 0   (3D no-flux)
#
# ptype:  0 = interior, 1 = left, 2 = right,
#         3 = bottom, 4 = top, 5 = front, 6 = back

H_u = args.h_left
H_d = args.h_right

ptype = np.zeros(N_part, dtype=int)
tol_bc = 0.5 * dx

for i in range(N_part):
    if   xp[i] < tol_bc:         ptype[i] = 1   # left  (x=0)
    elif xp[i] > Lx - tol_bc:    ptype[i] = 2   # right (x=Lx)
    elif zp[i] < tol_bc:         ptype[i] = 3   # bottom (z=0)
    elif zp[i] > Lz - tol_bc:    ptype[i] = 4   # top    (z=Lz)
    elif yp[i] < tol_bc:         ptype[i] = 5   # front  (y=0)
    elif yp[i] > Ly - tol_bc:    ptype[i] = 6   # back   (y=Ly)

n_left_local  = int(np.sum((ptype == 1) & (np.arange(N_part) < N_OWNED)))
n_right_local = int(np.sum((ptype == 2) & (np.arange(N_part) < N_OWNED)))
n_bot_local   = int(np.sum((ptype == 3) & (np.arange(N_part) < N_OWNED)))
n_top_local   = int(np.sum((ptype == 4) & (np.arange(N_part) < N_OWNED)))
n_front_local = int(np.sum((ptype == 5) & (np.arange(N_part) < N_OWNED)))
n_back_local  = int(np.sum((ptype == 6) & (np.arange(N_part) < N_OWNED)))
n_int_local   = int(np.sum((ptype == 0) & (np.arange(N_part) < N_OWNED)))
n_left  = comm_allreduce_sum(n_left_local)
n_right = comm_allreduce_sum(n_right_local)
n_bot   = comm_allreduce_sum(n_bot_local)
n_top   = comm_allreduce_sum(n_top_local)
n_front = comm_allreduce_sum(n_front_local)
n_back  = comm_allreduce_sum(n_back_local)
n_int   = comm_allreduce_sum(n_int_local)
log_root(f"\nParticle types (global):  interior={n_int}  left={n_left}  "
         f"right={n_right}  bot={n_bot}  top={n_top}  front={n_front}  back={n_back}")

# -- Initial condition --
#
# Linear hydraulic head H(x) = H_u + (H_d - H_u) x/Lx, with h_w = H - z.
# Gravity is along -z, so h = pressure head, H = h + z.

h_w = H_u + (H_d - H_u) * xp / Lx - zp      # vectorised, z is gravity

# Enforce Dirichlet BCs on left/right (vectorised)
h_w[ptype == 1] = H_u - zp[ptype == 1]
h_w[ptype == 2] = H_d - zp[ptype == 2]

h_init = h_w.copy()

# -- NAPL source region (3D cube) --
# 1 m x 1 m x 1 m cube, centred horizontally in x AND y, near top in z
SRC_X0, SRC_X1 = args.src_x0, args.src_x1
SRC_Y0, SRC_Y1 = args.src_y0, args.src_y1
SRC_Z0, SRC_Z1 = args.src_z0, args.src_z1
SN_SOURCE      = args.sn_source       # fixed NAPL saturation at source

is_source = ((xp >= SRC_X0) & (xp <= SRC_X1) &
             (yp >= SRC_Y0) & (yp <= SRC_Y1) &
             (zp >= SRC_Z0) & (zp <= SRC_Z1))

# Only OWNED source particles are this rank's responsibility for source
# maintenance.  Ghost source particles (if any are within this rank's
# extended pencil) get their values refreshed via the ghost exchange.
n_src_owned = int(np.sum(is_source[:N_OWNED]))
n_src_global = comm_allreduce_sum(n_src_owned)
log_root(f"NAPL source particles (global): {n_src_global}  "
         f"([{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}] x [{SRC_Z0},{SRC_Z1}], "
         f"Sn = {SN_SOURCE})")

# NAPL saturation field
S_n = np.zeros(N_part)
S_n[is_source] = SN_SOURCE

# ======================================================================
# 8.  BOUNDARY ENFORCEMENT  (vectorised with precomputed index arrays)
# ======================================================================
#
# 6 boundary face groups + source.  All loops eliminated via fancy indexing.
# BC enforcement applies only to OWNED particles; ghost values are
# overwritten by exchange each step.

_owned_mask = np.arange(N_part) < N_OWNED   # convenience mask

_idx_left  = np.where((ptype == 1) & _owned_mask)[0]
_idx_right = np.where((ptype == 2) & _owned_mask)[0]
_idx_bot   = np.where((ptype == 3) & _owned_mask)[0]
_idx_top   = np.where((ptype == 4) & _owned_mask)[0]
_idx_front = np.where((ptype == 5) & _owned_mask)[0]
_idx_back  = np.where((ptype == 6) & _owned_mask)[0]
_idx_src   = np.where(is_source & _owned_mask)[0]

# Precompute interior mirror indices for impermeable BCs.
# bottom (z=0)  mirrors from row k=1
# top    (z=Lz) mirrors from row k=Nz-2
# front  (y=0)  mirrors from row j=1
# back   (y=Ly) mirrors from row j=Ny-2

def _mirror_idx_for(ptype_value, mirror_jk):
    """Build mirror index array for a face. mirror_jk(i) returns (ix, jy, kz) of interior mirror."""
    out = np.empty(np.sum(ptype == ptype_value), dtype=np.int64)
    src_indices = np.where(ptype == ptype_value)[0]
    for n, i in enumerate(src_indices):
        ix, jy, kz = mirror_jk(i)
        ix = min(ix, Nx - 1); jy = min(jy, Ny - 1); kz = min(kz, Nz - 1)
        out[n] = idx_3d[ix, jy, kz]
    return out

_mirror_bot   = _mirror_idx_for(3,  # z=0  → mirror from k=1
    lambda i: (int(round(xp[i]/dx)), int(round(yp[i]/dy)), 1))
_mirror_top   = _mirror_idx_for(4,  # z=Lz → mirror from k=Nz-2
    lambda i: (int(round(xp[i]/dx)), int(round(yp[i]/dy)), Nz-2))
_mirror_front = _mirror_idx_for(5,  # y=0  → mirror from j=1
    lambda i: (int(round(xp[i]/dx)), 1,                       int(round(zp[i]/dz))))
_mirror_back  = _mirror_idx_for(6,  # y=Ly → mirror from j=Ny-2
    lambda i: (int(round(xp[i]/dx)), Ny-2,                    int(round(zp[i]/dz))))

# Precompute z-elevation differences for impermeable mirror (only z-faces matter for h)
#   h_bnd = h_int + (z_int - z_bnd)
# For y-faces (front/back), z_int == z_bnd so dz = 0 — but compute anyway for symmetry.
_dz_bot   = zp[_mirror_bot]   - zp[_idx_bot]
_dz_top   = zp[_mirror_top]   - zp[_idx_top]
_dz_front = zp[_mirror_front] - zp[_idx_front]   # = 0 for y-faces
_dz_back  = zp[_mirror_back]  - zp[_idx_back]    # = 0 for y-faces

# Precompute Dirichlet values (constant, never change)
_h_dirichlet_left  = H_u - zp[_idx_left]
_h_dirichlet_right = H_d - zp[_idx_right]

# Precompute "source-conflict" indices for impermeable boundaries.
# A boundary cell whose interior mirror is a source particle would otherwise
# inherit Sn = SN_SOURCE through the zero-gradient mirror BC, making the
# source visually (and physically) leak through the impermeable wall.
# We override these cells with Sn = 0 to keep the source strictly inside.
# This is a pure-resolution safeguard: at fine grids it's a no-op.
_idx_bot_srcconf   = _idx_bot[is_source[_mirror_bot]]
_idx_top_srcconf   = _idx_top[is_source[_mirror_top]]
_idx_front_srcconf = _idx_front[is_source[_mirror_front]]
_idx_back_srcconf  = _idx_back[is_source[_mirror_back]]
_n_srcconf = (len(_idx_bot_srcconf) + len(_idx_top_srcconf)
              + len(_idx_front_srcconf) + len(_idx_back_srcconf))

print(f"BC indices precomputed:  left={len(_idx_left)}  right={len(_idx_right)}  "
      f"bot={len(_idx_bot)}  top={len(_idx_top)}  "
      f"front={len(_idx_front)}  back={len(_idx_back)}  source={len(_idx_src)}")
if _n_srcconf > 0:
    print(f"  WARNING: {_n_srcconf} impermeable-boundary cells have a source as mirror "
          f"(will be forced to Sn=0 to avoid wall leakage; consider finer grid)")


def enforce_impermeable_bc(h_field):
    """Mirror H from interior to all 4 impermeable faces (vectorised)."""
    h_field[_idx_bot]   = h_field[_mirror_bot]   + _dz_bot
    h_field[_idx_top]   = h_field[_mirror_top]   + _dz_top
    h_field[_idx_front] = h_field[_mirror_front] + _dz_front
    h_field[_idx_back]  = h_field[_mirror_back]  + _dz_back
    return h_field


def enforce_dirichlet_bc(h_field):
    """Impose Dirichlet h on left/right boundaries (vectorised)."""
    h_field[_idx_left]  = _h_dirichlet_left
    h_field[_idx_right] = _h_dirichlet_right
    return h_field


def enforce_napl_bc(Sn):
    """NAPL BCs: Sn=0 at Dirichlet walls, mirror at impermeable, source (vectorised).
    Boundary cells whose mirror is a source are overridden to Sn=0 to prevent
    coarse-grid leakage through the impermeable walls.
    """
    Sn[_idx_left]  = 0.0
    Sn[_idx_right] = 0.0
    Sn[_idx_bot]   = Sn[_mirror_bot]
    Sn[_idx_top]   = Sn[_mirror_top]
    Sn[_idx_front] = Sn[_mirror_front]
    Sn[_idx_back]  = Sn[_mirror_back]
    # Source-conflict override: zero out impermeable-boundary cells whose
    # mirror is a source (otherwise they inherit SN_SOURCE through the mirror).
    Sn[_idx_bot_srcconf]   = 0.0
    Sn[_idx_top_srcconf]   = 0.0
    Sn[_idx_front_srcconf] = 0.0
    Sn[_idx_back_srcconf]  = 0.0
    Sn[_idx_src] = SN_SOURCE
    return Sn


# ======================================================================
# 9.  NUMBA-JITTED SPH OPERATORS  +  WRAPPER FUNCTIONS
# ======================================================================
#
# The inner particle loops are factored into Numba-compiled functions
# that operate on CSR neighbour arrays.  The Python wrappers handle
# the NumPy constitutive evaluations (k, H, Cs), then call the jitted
# inner loops.
#
# Both compute_dhdt and compute_dSndt use the SAME SPH operator:
#   div[k grad(H)] via the corrected Laplacian (Eq. 57).
# They differ only in: k-field, H-field, skip mask, storage divisor.

if HAS_NUMBA:
    from numba import prange

    @njit(cache=True, parallel=True)
    def _sph_div_k_gradH(H_f, k_h, C_store, ptype_arr, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                          L_inv, K_norm, err_x, err_y, err_z, Vp_val, N):
        """Corrected Laplacian (3D) with precomputed field arrays.
        Used by Darcy velocity (snapshot path) and NumPy fallback.
        """
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                Hi = H_f[i]
                ki = k_h[i]

                raw_gx = 0.0; raw_gy = 0.0; raw_gz = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    dH = H_f[j] - Hi
                    raw_gx += Vp_val * dH * nbr_gWx[k]
                    raw_gy += Vp_val * dH * nbr_gWy[k]
                    raw_gz += Vp_val * dH * nbr_gWz[k]

                L00 = L_inv[i,0,0]; L01 = L_inv[i,0,1]; L02 = L_inv[i,0,2]
                L10 = L_inv[i,1,0]; L11 = L_inv[i,1,1]; L12 = L_inv[i,1,2]
                L20 = L_inv[i,2,0]; L21 = L_inv[i,2,1]; L22 = L_inv[i,2,2]
                grad_Hx = L00*raw_gx + L01*raw_gy + L02*raw_gz
                grad_Hy = L10*raw_gx + L11*raw_gy + L12*raw_gz
                grad_Hz = L20*raw_gx + L21*raw_gy + L22*raw_gz

                lap_sum = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    km = 0.5 * (ki + k_h[j])
                    lap_sum += Vp_val * km * (H_f[j] - Hi) * nbr_Fhat[k]

                corr = ki * (grad_Hx*err_x[i] + grad_Hy*err_y[i] + grad_Hz*err_z[i])

                Kn = K_norm[i]
                if abs(Kn) >= 1e-30:
                    out[i] = (2.0 / Kn) * (lap_sum - corr) / C_store[i]

        return out

    # ── FULLY FUSED KERNELS: no intermediate field arrays ──────────────
    @njit(cache=True, parallel=True)
    def _sph_dhdt_fused(h_w, S_n, zp_arr, skip_mask,
                        nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                        L_inv, K_norm, err_x, err_y, err_z, Vp_val, N):
        """Water RHS dh_w/dt — fully fused 3D (k_w, H_w, C_tilde inline)."""
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i = h_w[i]
                sn_i = S_n[i]
                Hi = hw_i + zp_arr[i]
                ki = _s_kw(hw_i, sn_i)

                raw_gx = 0.0; raw_gy = 0.0; raw_gz = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    Hj = h_w[j] + zp_arr[j]
                    dH = Hj - Hi
                    raw_gx += Vp_val * dH * nbr_gWx[k]
                    raw_gy += Vp_val * dH * nbr_gWy[k]
                    raw_gz += Vp_val * dH * nbr_gWz[k]

                L00 = L_inv[i,0,0]; L01 = L_inv[i,0,1]; L02 = L_inv[i,0,2]
                L10 = L_inv[i,1,0]; L11 = L_inv[i,1,1]; L12 = L_inv[i,1,2]
                L20 = L_inv[i,2,0]; L21 = L_inv[i,2,1]; L22 = L_inv[i,2,2]
                grad_Hx = L00*raw_gx + L01*raw_gy + L02*raw_gz
                grad_Hy = L10*raw_gx + L11*raw_gy + L12*raw_gz
                grad_Hz = L20*raw_gx + L21*raw_gy + L22*raw_gz

                lap_sum = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    kj = _s_kw(h_w[j], S_n[j])
                    Hj = h_w[j] + zp_arr[j]
                    km = 0.5 * (ki + kj)
                    lap_sum += Vp_val * km * (Hj - Hi) * nbr_Fhat[k]

                corr = ki * (grad_Hx*err_x[i] + grad_Hy*err_y[i] + grad_Hz*err_z[i])

                Kn = K_norm[i]
                if abs(Kn) >= 1e-30:
                    Ct = _s_Ctilde(hw_i)
                    out[i] = (2.0 / Kn) * (lap_sum - corr) / Ct

        return out

    @njit(cache=True, parallel=True)
    def _sph_dSndt_fused(h_w, S_n, zp_arr, skip_mask,
                         nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                         L_inv, K_norm, err_x, err_y, err_z, Vp_val, N, phi_val):
        """NAPL RHS dS_n/dt — fully fused 3D (k_n, H_n inline)."""
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i = h_w[i]
                sn_i = S_n[i]
                Hi = _s_Hn(hw_i, sn_i, zp_arr[i])
                ki = _s_kn(hw_i, sn_i)

                raw_gx = 0.0; raw_gy = 0.0; raw_gz = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    Hj = _s_Hn(h_w[j], S_n[j], zp_arr[j])
                    dH = Hj - Hi
                    raw_gx += Vp_val * dH * nbr_gWx[k]
                    raw_gy += Vp_val * dH * nbr_gWy[k]
                    raw_gz += Vp_val * dH * nbr_gWz[k]

                L00 = L_inv[i,0,0]; L01 = L_inv[i,0,1]; L02 = L_inv[i,0,2]
                L10 = L_inv[i,1,0]; L11 = L_inv[i,1,1]; L12 = L_inv[i,1,2]
                L20 = L_inv[i,2,0]; L21 = L_inv[i,2,1]; L22 = L_inv[i,2,2]
                grad_Hx = L00*raw_gx + L01*raw_gy + L02*raw_gz
                grad_Hy = L10*raw_gx + L11*raw_gy + L12*raw_gz
                grad_Hz = L20*raw_gx + L21*raw_gy + L22*raw_gz

                lap_sum = 0.0
                for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                    j = nbr_j[k]
                    kj = _s_kn(h_w[j], S_n[j])
                    Hj = _s_Hn(h_w[j], S_n[j], zp_arr[j])
                    km = 0.5 * (ki + kj)
                    lap_sum += Vp_val * km * (Hj - Hi) * nbr_Fhat[k]

                corr = ki * (grad_Hx*err_x[i] + grad_Hy*err_y[i] + grad_Hz*err_z[i])

                Kn = K_norm[i]
                if abs(Kn) >= 1e-30:
                    out[i] = (2.0 / Kn) * (lap_sum - corr) / phi_val

        return out

    @njit(cache=True, parallel=True)
    def _sph_kvar_gradient(H_f, k_h,
                            nbr_ptr, nbr_j, nbr_gWx, nbr_gWy, nbr_gWz,
                            L_inv, Vp_val, N):
        """Corrected gradient (3D) with variable-k weighting.
        Returns (qx, qy, qz) = SPH approximation of k*grad(H).
        """
        qx = np.zeros(N); qy = np.zeros(N); qz = np.zeros(N)
        for i in prange(N):
            L00 = L_inv[i,0,0]; L01 = L_inv[i,0,1]; L02 = L_inv[i,0,2]
            L10 = L_inv[i,1,0]; L11 = L_inv[i,1,1]; L12 = L_inv[i,1,2]
            L20 = L_inv[i,2,0]; L21 = L_inv[i,2,1]; L22 = L_inv[i,2,2]
            ki = k_h[i]; Hi = H_f[i]
            sx = 0.0; sy = 0.0; sz = 0.0
            for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                j = nbr_j[k]
                km  = 0.5 * (ki + k_h[j])
                dH  = H_f[j] - Hi
                gx  = nbr_gWx[k]; gy = nbr_gWy[k]; gz = nbr_gWz[k]
                cWx = L00*gx + L01*gy + L02*gz
                cWy = L10*gx + L11*gy + L12*gz
                cWz = L20*gx + L21*gy + L22*gz
                val = Vp_val * km * dH
                sx += val * cWx
                sy += val * cWy
                sz += val * cWz
            qx[i] = sx; qy[i] = sy; qz[i] = sz
        return qx, qy, qz

# ---------- Fallback pure-Python implementations (3D) ----------

def _sph_div_k_gradH_py(H_f, k_h, C_store, ptype_arr, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                          L_inv, K_norm, err_x, err_y, err_z, Vp_val, N):
    out = np.zeros(N)
    for i in range(N):
        if skip_mask[i]:
            continue
        Hi = H_f[i]; ki = k_h[i]
        raw_gx = 0.0; raw_gy = 0.0; raw_gz = 0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_j[k]; dH = H_f[j] - Hi
            raw_gx += Vp_val * dH * nbr_gWx[k]
            raw_gy += Vp_val * dH * nbr_gWy[k]
            raw_gz += Vp_val * dH * nbr_gWz[k]
        L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L02=L_inv[i,0,2]
        L10=L_inv[i,1,0]; L11=L_inv[i,1,1]; L12=L_inv[i,1,2]
        L20=L_inv[i,2,0]; L21=L_inv[i,2,1]; L22=L_inv[i,2,2]
        grad_Hx = L00*raw_gx + L01*raw_gy + L02*raw_gz
        grad_Hy = L10*raw_gx + L11*raw_gy + L12*raw_gz
        grad_Hz = L20*raw_gx + L21*raw_gy + L22*raw_gz
        lap_sum = 0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_j[k]; km = 0.5*(ki+k_h[j])
            lap_sum += Vp_val * km * (H_f[j]-Hi) * nbr_Fhat[k]
        corr = ki * (grad_Hx*err_x[i] + grad_Hy*err_y[i] + grad_Hz*err_z[i])
        Kn = K_norm[i]
        if abs(Kn) < 1e-30: continue
        out[i] = (2.0/Kn)*(lap_sum-corr)/C_store[i]
    return out

def _sph_kvar_gradient_py(H_f, k_h, nbr_ptr, nbr_j, nbr_gWx, nbr_gWy, nbr_gWz,
                           L_inv, Vp_val, N):
    qx = np.zeros(N); qy = np.zeros(N); qz = np.zeros(N)
    for i in range(N):
        L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L02=L_inv[i,0,2]
        L10=L_inv[i,1,0]; L11=L_inv[i,1,1]; L12=L_inv[i,1,2]
        L20=L_inv[i,2,0]; L21=L_inv[i,2,1]; L22=L_inv[i,2,2]
        ki=k_h[i]; Hi=H_f[i]; sx=0.0; sy=0.0; sz=0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j=nbr_j[k]; km=0.5*(ki+k_h[j]); dH=H_f[j]-Hi
            gx=nbr_gWx[k]; gy=nbr_gWy[k]; gz=nbr_gWz[k]
            cWx=L00*gx+L01*gy+L02*gz
            cWy=L10*gx+L11*gy+L12*gz
            cWz=L20*gx+L21*gy+L22*gz
            val=Vp_val*km*dH; sx+=val*cWx; sy+=val*cWy; sz+=val*cWz
        qx[i]=sx; qy[i]=sy; qz[i]=sz
    return qx, qy, qz

# Choose implementation
_div_k_gradH  = _sph_div_k_gradH  if HAS_NUMBA else _sph_div_k_gradH_py
_kvar_gradient = _sph_kvar_gradient if HAS_NUMBA else _sph_kvar_gradient_py

# ---------- Wrapper functions (same API as before) ----------

# Precompute skip masks (boolean arrays)
_skip_dirichlet = (ptype == 1) | (ptype == 2)
_skip_napl      = _skip_dirichlet | is_source

# Pre-allocate reusable field arrays (avoids allocation every step)
_fld_kw = np.empty(N_part)
_fld_Hw = np.empty(N_part)
_fld_Ct = np.empty(N_part)
_fld_kn = np.empty(N_part)
_fld_Hn = np.empty(N_part)
_fld_Cn = np.full(N_part, phi_0)    # constant, never changes


def _precompute_step(h_field, Sn_field):
    """Compute all constitutive fields for both phases.
    Uses Numba prange (parallel) if available, else NumPy (serial).
    """
    if HAS_NUMBA:
        _precompute_fields(h_field, Sn_field, zp, N_part,
                           _fld_kw, _fld_Hw, _fld_Ct, _fld_kn, _fld_Hn)
    else:
        _fld_kw[:] = compute_kw_3ph(h_field, Sn_field)
        _fld_Hw[:] = h_field + zp
        Cs = compute_Cs(h_field)
        np.maximum(C_l + Cs, C_l, out=_fld_Ct)
        _fld_kn[:] = compute_kn_field(h_field, Sn_field)
        _fld_Hn[:] = compute_Hn_field(h_field, Sn_field)


def compute_dhdt(h_field, Sn_field):
    """RHS of the water seepage equation (three-phase k_w)."""
    if HAS_NUMBA:
        return _sph_dhdt_fused(h_field, Sn_field, zp, _skip_dirichlet,
                                nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                                L_inv, K_norm, err_x, err_y, err_z, Vp, N_part)
    else:
        _precompute_step(h_field, Sn_field)
        return _div_k_gradH(_fld_Hw, _fld_kw, _fld_Ct, ptype, _skip_dirichlet,
                             nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                             L_inv, K_norm, err_x, err_y, err_z, Vp, N_part)


def compute_dSndt(h_field, Sn_field):
    """NAPL transport equation RHS."""
    if HAS_NUMBA:
        return _sph_dSndt_fused(h_field, Sn_field, zp, _skip_napl,
                                 nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                                 L_inv, K_norm, err_x, err_y, err_z, Vp, N_part, phi_0)
    else:
        return _div_k_gradH(_fld_Hn, _fld_kn, _fld_Cn, ptype, _skip_napl,
                             nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                             L_inv, K_norm, err_x, err_y, err_z, Vp, N_part)


def compute_darcy_velocity(h_field, Sn_field):
    """Water Darcy velocity  q_w = -k_w grad(H_w).  Returns (qx, qy, qz)."""
    _precompute_step(h_field, Sn_field)
    qx, qy, qz = _kvar_gradient(_fld_Hw, _fld_kw, nbr_ptr, nbr_j,
                                  nbr_gWx, nbr_gWy, nbr_gWz, L_inv, Vp, N_part)
    return -qx, -qy, -qz


def compute_darcy_velocity_napl(h_field, Sn_field):
    """NAPL Darcy velocity  q_n = -k_n grad(H_n).  Returns (qx, qy, qz)."""
    qx, qy, qz = _kvar_gradient(_fld_Hn, _fld_kn, nbr_ptr, nbr_j,
                                  nbr_gWx, nbr_gWy, nbr_gWz, L_inv, Vp, N_part)
    return -qx, -qy, -qz


# ======================================================================
# 11.  TIME STEP   [Paper Eq. 66 / Appendix A.17]
# ======================================================================
CFL = args.cfl

def stable_dt(h_field, Sn_field):
    Cs = compute_Cs(h_field)
    C_tilde = C_l + Cs
    C_min = max(np.min(C_tilde), C_l)
    dt_w = CFL * C_min * h_sml**2 / k_sat
    # NAPL CFL
    dt_n = CFL * phi_0 * h_sml**2 / k_sat_n
    return min(dt_w, dt_n)


# ======================================================================
# 11b. GHOST EXCHANGE  (MPI-aware; no-op in single-process)
# ======================================================================
# Per-step ghost exchange uses a two-pass scheme:
#   Pass 1 (X): each rank sends its leftmost/rightmost GHOST_W owned x-layers
#               to its left/right neighbour; receives x-ghost regions in turn.
#   Pass 2 (Y): each rank sends front/back y-layers from the EXTENDED x-range
#               (owned + just-received x-ghosts); receives y-ghost regions.
#               Because the y-pass send buffer includes the just-received
#               x-ghost rows, corner data is propagated correctly without
#               explicit diagonal MPI calls.
#
# Sendrecv uses MPI.PROC_NULL for ranks on the global boundary, which makes
# the call a no-op for that direction — no special-casing needed.

def _build_ghost_exchange_indices():
    """Precompute send-buffer indices (into the local array) for each
    direction. Recv buffers go directly into the contiguous ghost blocks.
    Returns None in single-process mode.
    """
    if not USE_MPI:
        return None

    def _idx_for(i_range, j_range, k_range):
        """Local indices for the cartesian product, in C-order (i, j, k).
        Order matches what the neighbour will unpack into its ghost block."""
        out = []
        for ii in i_range:
            for jj in j_range:
                for kk in k_range:
                    li = ijk_to_local.get((int(ii), int(jj), int(kk)), -1)
                    if li < 0:
                        raise RuntimeError(
                            f"Ghost index lookup failed at "
                            f"({ii},{jj},{kk}) on rank {MPI_RANK}")
                    out.append(li)
        return np.array(out, dtype=np.int64)

    k_full = np.arange(0, Nz)

    # Pass 1: x-direction send buffers (owned edge layers)
    if NB_L != MPI.PROC_NULL:
        send_L = _idx_for(np.arange(IX_LO, IX_LO + GHOST_W),
                          np.arange(IY_LO, IY_HI), k_full)
    else:
        send_L = np.empty(0, dtype=np.int64)
    if NB_R != MPI.PROC_NULL:
        send_R = _idx_for(np.arange(IX_HI - GHOST_W, IX_HI),
                          np.arange(IY_LO, IY_HI), k_full)
    else:
        send_R = np.empty(0, dtype=np.int64)

    # Pass 2: y-direction send buffers (extended x-range to cover corners)
    if NB_F != MPI.PROC_NULL:
        send_F = _idx_for(np.arange(IX_GL, IX_GR),
                          np.arange(IY_LO, IY_LO + GHOST_W), k_full)
    else:
        send_F = np.empty(0, dtype=np.int64)
    if NB_B != MPI.PROC_NULL:
        send_B = _idx_for(np.arange(IX_GL, IX_GR),
                          np.arange(IY_HI - GHOST_W, IY_HI), k_full)
    else:
        send_B = np.empty(0, dtype=np.int64)

    return {
        "send_L": send_L, "send_R": send_R,
        "send_F": send_F, "send_B": send_B,
        # Recv targets are the contiguous ghost blocks in the local array:
        "recv_XL": np.arange(OFF_XL, OFF_XL + N_GHOST_XL, dtype=np.int64),
        "recv_XR": np.arange(OFF_XR, OFF_XR + N_GHOST_XR, dtype=np.int64),
        "recv_YF": np.arange(OFF_YF, OFF_YF + N_GHOST_YF, dtype=np.int64),
        "recv_YB": np.arange(OFF_YB, OFF_YB + N_GHOST_YB, dtype=np.int64),
    }


_GHOST = _build_ghost_exchange_indices()


def exchange_ghosts(*fields):
    """Exchange ghost-layer values for the given 1D local arrays.
    Each `fld` has length N_LOCAL; ghost regions get updated in place.
    No-op in single-process mode.
    """
    if not USE_MPI or _GHOST is None:
        return

    # Pass 1: X exchange
    for fld in fields:
        send_L_buf = np.ascontiguousarray(fld[_GHOST["send_L"]])
        send_R_buf = np.ascontiguousarray(fld[_GHOST["send_R"]])
        recv_L_buf = np.empty(N_GHOST_XL, dtype=fld.dtype)
        recv_R_buf = np.empty(N_GHOST_XR, dtype=fld.dtype)

        # Send to L, receive from R
        _comm.Sendrecv(sendbuf=send_L_buf, dest=NB_L,
                       recvbuf=recv_R_buf, source=NB_R)
        # Send to R, receive from L
        _comm.Sendrecv(sendbuf=send_R_buf, dest=NB_R,
                       recvbuf=recv_L_buf, source=NB_L)

        if NB_L != MPI.PROC_NULL:
            fld[_GHOST["recv_XL"]] = recv_L_buf
        if NB_R != MPI.PROC_NULL:
            fld[_GHOST["recv_XR"]] = recv_R_buf

    # Pass 2: Y exchange (send buffers include the just-received x-ghosts)
    for fld in fields:
        send_F_buf = np.ascontiguousarray(fld[_GHOST["send_F"]])
        send_B_buf = np.ascontiguousarray(fld[_GHOST["send_B"]])
        recv_F_buf = np.empty(N_GHOST_YF, dtype=fld.dtype)
        recv_B_buf = np.empty(N_GHOST_YB, dtype=fld.dtype)

        _comm.Sendrecv(sendbuf=send_F_buf, dest=NB_F,
                       recvbuf=recv_B_buf, source=NB_B)
        _comm.Sendrecv(sendbuf=send_B_buf, dest=NB_B,
                       recvbuf=recv_F_buf, source=NB_F)

        if NB_F != MPI.PROC_NULL:
            fld[_GHOST["recv_YF"]] = recv_F_buf
        if NB_B != MPI.PROC_NULL:
            fld[_GHOST["recv_YB"]] = recv_B_buf


# Pre-gather per-rank metadata for snapshot/checkpoint assembly.
# We gather the GLOBAL particle index of each rank's owned particles
# onto rank 0 once at setup. Subsequent gathers of field data are
# placed at the correct global positions using fancy indexing.
if USE_MPI:
    _counts_owned = _comm.allgather(int(N_OWNED))
    if MPI_RANK == 0:
        _gather_perm = np.empty(N_part_global, dtype=np.int64)
        _displs_owned = np.cumsum([0] + _counts_owned[:-1]).tolist()
        _gidx_recvbuf = (_gather_perm, _counts_owned, _displs_owned,
                         MPI._typedict[np.dtype(np.int64).char])
    else:
        _gather_perm = None
        _gidx_recvbuf = None
    _comm.Gatherv(sendbuf=np.ascontiguousarray(global_idx_owned),
                  recvbuf=_gidx_recvbuf, root=0)
else:
    _counts_owned = [N_OWNED]
    _gather_perm = None


def _gather_to_global_array(local_owned):
    """Gather one field's owned-only data to rank 0 in global particle order.
    Returns: full-length array (length N_part_global) on rank 0, None elsewhere.
    Identity in single-process mode.
    """
    if not USE_MPI:
        return local_owned

    owned = np.ascontiguousarray(local_owned)
    if MPI_RANK == 0:
        gathered = np.empty(N_part_global, dtype=owned.dtype)
        displs = np.cumsum([0] + _counts_owned[:-1]).tolist()
        recvbuf = (gathered, _counts_owned, displs,
                   MPI._typedict[owned.dtype.char])
    else:
        recvbuf = None
    _comm.Gatherv(sendbuf=owned, recvbuf=recvbuf, root=0)

    if MPI_RANK != 0:
        return None

    # Reorder from rank-concatenated to true global particle order
    global_arr = np.empty(N_part_global, dtype=owned.dtype)
    global_arr[_gather_perm] = gathered
    return global_arr


# Initial ghost population (so the first SPH operator call sees correct data).
# h_w and S_n at this point should already match across ranks for the shared
# ghost cells (each rank computed them from the same global IC formula), but
# the exchange is cheap and guarantees consistency.
if USE_MPI:
    log_root("Initial ghost exchange ...")
    exchange_ghosts(h_w, S_n)
    if args.mpi_validate:
        # Roundtrip sanity: exchange positions and verify they didn't change
        xp_save = xp.copy()
        yp_save = yp.copy()
        zp_save = zp.copy()
        exchange_ghosts(xp, yp, zp)
        max_diff = max(np.max(np.abs(xp - xp_save)),
                       np.max(np.abs(yp - yp_save)),
                       np.max(np.abs(zp - zp_save)))
        max_diff_global = comm_allreduce_max(max_diff)
        log_root(f"  Position-consistency check (across-rank max diff): "
                 f"{max_diff_global:.3e}")
        if max_diff_global > 1e-10:
            log_root("  WARNING: ghost positions differ across ranks!")


# ======================================================================
# 12.  HDF5 SNAPSHOT I/O
# ======================================================================
try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False
    print("WARNING: h5py not available - HDF5 snapshots disabled.")

hdf5_path = os.path.join(OUTPUT_DIR, args.snapshot_file)  # kept for compat (unused)

SNAP_DIR = os.path.join(OUTPUT_DIR, "snapshots")

def _save_snapshot_global(step, t, h_field, Sn_field, dhdt, dSndt,
                          qx, qy, qz, qxn, qyn, qzn,
                          xp_g, yp_g, zp_g, ptype_g, is_source_g, kw_g, kn_g):
    """Write one snapshot to its OWN file: snapshots/step_NNNNNNNNNNNN.{h5,npz}.

    One file per snapshot avoids the bus-error / metadata-cache problems
    encountered with single multi-GB HDF5 files on parallel filesystems
    (Lustre/GPFS). Each file is self-contained and independently readable.

    All arrays must already be in global particle order, length N_part_global.
    Called only on rank 0 in MPI mode.
    """
    Sw = compute_Sw_3ph(h_field, Sn_field)
    os.makedirs(SNAP_DIR, exist_ok=True)

    if HAS_H5:
        path = os.path.join(SNAP_DIR, f"step_{step:012d}.h5")
        # Write to a tmp file then atomic-rename so partial writes don't
        # leave a half-written file under the canonical name.
        tmp_path = path + ".tmp"
        with h5py.File(tmp_path, "w") as f:
            # Run + snapshot metadata at ROOT level (no nesting)
            f.attrs["step"]   = step
            f.attrs["time_s"] = t
            f.attrs["Nx"]     = Nx
            f.attrs["Ny"]     = Ny
            f.attrs["Nz"]     = Nz
            f.attrs["Lx"]     = Lx
            f.attrs["Ly"]     = Ly
            f.attrs["Lz"]     = Lz
            f.attrs["k_sat"]  = k_sat
            f.attrs["H_u"]    = H_u
            f.attrs["H_d"]    = H_d
            if USE_MPI:
                f.attrs["mpi_nrank"] = MPI_NRANK
                f.attrs["mpi_Px"]    = Px
                f.attrs["mpi_Py"]    = Py

            # Per-particle datasets at ROOT level (no group nesting)
            f.create_dataset("x",     data=xp_g)
            f.create_dataset("y",     data=yp_g)
            f.create_dataset("z",     data=zp_g)
            f.create_dataset("h",     data=h_field)
            f.create_dataset("H",     data=h_field + zp_g)
            f.create_dataset("Sn",    data=Sn_field)
            f.create_dataset("Sw",    data=Sw)
            f.create_dataset("kw",    data=kw_g)
            f.create_dataset("kn",    data=kn_g)
            f.create_dataset("dhdt",  data=dhdt)
            f.create_dataset("dSndt", data=dSndt)
            f.create_dataset("qx",    data=qx)
            f.create_dataset("qy",    data=qy)
            f.create_dataset("qz",    data=qz)
            f.create_dataset("qxn",   data=qxn)
            f.create_dataset("qyn",   data=qyn)
            f.create_dataset("qzn",   data=qzn)
            f.create_dataset("ptype", data=ptype_g)
            f.create_dataset("is_source", data=is_source_g.astype(np.int8))
        os.replace(tmp_path, path)
    else:
        path = os.path.join(SNAP_DIR, f"step_{step:012d}.npz")
        np.savez_compressed(path,
            step=step, time_s=t, Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
            k_sat=k_sat, H_u=H_u, H_d=H_d,
            x=xp_g, y=yp_g, z=zp_g, h=h_field, H=h_field + zp_g,
            Sn=Sn_field, Sw=Sw, kw=kw_g, kn=kn_g,
            dhdt=dhdt, dSndt=dSndt,
            qx=qx, qy=qy, qz=qz,
            qxn=qxn, qyn=qyn, qzn=qzn,
            ptype=ptype_g, is_source=is_source_g.astype(np.int8))


# Pre-gather constant fields (positions, ptype, is_source) ONCE — they
# don't change between snapshots. Only rank 0 holds the gathered copies.
if USE_MPI:
    _xp_global        = _gather_to_global_array(xp[:N_OWNED])
    _yp_global        = _gather_to_global_array(yp[:N_OWNED])
    _zp_global        = _gather_to_global_array(zp[:N_OWNED])
    _ptype_global     = _gather_to_global_array(ptype[:N_OWNED])
    _is_source_global = _gather_to_global_array(is_source[:N_OWNED].astype(np.int8))
    if MPI_RANK == 0:
        _is_source_global = _is_source_global.astype(bool)
else:
    _xp_global = xp
    _yp_global = yp
    _zp_global = zp
    _ptype_global = ptype
    _is_source_global = is_source


def save_snapshot(fname, step, t, h_field, Sn_field, dhdt, dSndt,
                  qx, qy, qz, qxn, qyn, qzn):
    """MPI-aware snapshot saver.

    Writes one self-contained file per snapshot to SNAP_DIR/step_NNNN.{h5,npz}.
    The `fname` argument is retained for legacy interface compatibility but
    is ignored — the actual filename is derived from the step number.

    In MPI mode: all ranks pass their LOCAL per-particle arrays (length
    N_LOCAL); this routine gathers the OWNED slices onto rank 0 in global
    particle order, then rank 0 writes the per-snapshot file.  Constant
    fields (positions, ptype, is_source) were gathered ONCE at setup
    and are reused.
    """
    # Per-call computed fields (kw, kn) — gather alongside
    kw_local = compute_kw_3ph(h_field, Sn_field)
    kn_local = compute_kn_field(h_field, Sn_field)

    if not USE_MPI:
        _save_snapshot_global(step, t,
                              h_field, Sn_field, dhdt, dSndt,
                              qx, qy, qz, qxn, qyn, qzn,
                              _xp_global, _yp_global, _zp_global,
                              _ptype_global, _is_source_global,
                              kw_local, kn_local)
        return

    # Gather all variable-per-step fields
    h_g     = _gather_to_global_array(h_field[:N_OWNED])
    Sn_g    = _gather_to_global_array(Sn_field[:N_OWNED])
    dhdt_g  = _gather_to_global_array(dhdt[:N_OWNED])
    dSndt_g = _gather_to_global_array(dSndt[:N_OWNED])
    qx_g    = _gather_to_global_array(qx[:N_OWNED])
    qy_g    = _gather_to_global_array(qy[:N_OWNED])
    qz_g    = _gather_to_global_array(qz[:N_OWNED])
    qxn_g   = _gather_to_global_array(qxn[:N_OWNED])
    qyn_g   = _gather_to_global_array(qyn[:N_OWNED])
    qzn_g   = _gather_to_global_array(qzn[:N_OWNED])
    kw_g    = _gather_to_global_array(kw_local[:N_OWNED])
    kn_g    = _gather_to_global_array(kn_local[:N_OWNED])

    if MPI_RANK == 0:
        _save_snapshot_global(step, t,
                              h_g, Sn_g, dhdt_g, dSndt_g,
                              qx_g, qy_g, qz_g, qxn_g, qyn_g, qzn_g,
                              _xp_global, _yp_global, _zp_global,
                              _ptype_global, _is_source_global,
                              kw_g, kn_g)

# NOTE: each snapshot is now an INDEPENDENT FILE in SNAP_DIR. Restart logic
# removes future-of-checkpoint files via filesystem unlink — no h5repack needed.


# ======================================================================
# 12b. CHECKPOINT SAVE / LOAD  (individual HDF5 files — HPC/SLURM)
# ======================================================================
#
# Each checkpoint is a separate HDF5 file:  ckpt_00001000.h5
# This avoids corruption from interrupted writes to a single file
# (e.g. SLURM wall-time kill), and allows easy cleanup of old checkpoints.
# Each file is self-contained: fields, metadata, and full history arrays.

CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

def _save_checkpoint_global(step, t, hw, Sn, time_log, l2_log, l2n_log,
                            Sn_max_log, napl_mass_log, napl_mass_nosrc_log):
    """Underlying checkpoint writer. Operates on already-gathered global arrays.
    Called only on rank 0 in MPI mode.
    """
    if not HAS_H5:
        path = os.path.join(CKPT_DIR, f"ckpt_{step:012d}.npz")
        np.savez_compressed(path, step=step, t_phys=t, h_w=hw, S_n=Sn,
            Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
            time_log=np.array(time_log), l2_log=np.array(l2_log),
            l2n_log=np.array(l2n_log), Sn_max_log=np.array(Sn_max_log),
            napl_mass_log=np.array(napl_mass_log),
            napl_mass_nosrc_log=np.array(napl_mass_nosrc_log))
        print(f"    checkpoint saved (NPZ fallback): {path}")
        return

    path = os.path.join(CKPT_DIR, f"ckpt_{step:012d}.h5")
    with h5py.File(path, "w") as f:
        f.attrs["step"]    = step
        f.attrs["t_phys"]  = t
        f.attrs["Nx"]      = Nx
        f.attrs["Ny"]      = Ny
        f.attrs["Nz"]      = Nz
        f.attrs["Lx"]      = Lx
        f.attrs["Ly"]      = Ly
        f.attrs["Lz"]      = Lz
        f.attrs["k_sat"]   = k_sat
        f.attrs["k_sat_n"] = k_sat_n
        f.attrs["H_u"]     = H_u
        f.attrs["H_d"]     = H_d
        f.attrs["src_x0"]  = SRC_X0; f.attrs["src_x1"] = SRC_X1
        f.attrs["src_y0"]  = SRC_Y0; f.attrs["src_y1"] = SRC_Y1
        f.attrs["src_z0"]  = SRC_Z0; f.attrs["src_z1"] = SRC_Z1
        f.attrs["sn_source"] = SN_SOURCE

        ckw = dict(compression="gzip", compression_opts=4)
        f.create_dataset("h_w", data=hw, **ckw)
        f.create_dataset("S_n", data=Sn, **ckw)

        f.create_dataset("time_log",            data=np.array(time_log))
        f.create_dataset("l2_log",              data=np.array(l2_log))
        f.create_dataset("l2n_log",             data=np.array(l2n_log))
        f.create_dataset("Sn_max_log",          data=np.array(Sn_max_log))
        f.create_dataset("napl_mass_log",       data=np.array(napl_mass_log))
        f.create_dataset("napl_mass_nosrc_log", data=np.array(napl_mass_nosrc_log))

    print(f"    checkpoint saved: {path}")


def save_checkpoint(step, t, hw, Sn, time_log, l2_log, l2n_log,
                    Sn_max_log, napl_mass_log, napl_mass_nosrc_log):
    """MPI-aware checkpoint save.

    In MPI mode: gather OWNED slices to rank 0, then write a single
    HDF5 checkpoint identical in format to the single-process case.
    Restart on a different rank count is supported because we don't
    record rank-specific info in the checkpoint.
    """
    if not USE_MPI:
        _save_checkpoint_global(step, t, hw, Sn, time_log, l2_log, l2n_log,
                                Sn_max_log, napl_mass_log, napl_mass_nosrc_log)
        return

    hw_g = _gather_to_global_array(hw[:N_OWNED])
    Sn_g = _gather_to_global_array(Sn[:N_OWNED])

    if MPI_RANK == 0:
        _save_checkpoint_global(step, t, hw_g, Sn_g,
                                time_log, l2_log, l2n_log,
                                Sn_max_log, napl_mass_log, napl_mass_nosrc_log)


def load_latest_checkpoint():
    """Scan checkpoint directory for the most recent file.
    Tries HDF5 first, then NPZ fallback.
    Returns dict or None.
    """
    import glob

    # Helper to extract step number from a checkpoint path.
    # Robust to any zero-padding width (legacy files mixed with current).
    def step_from_path(p):
        base = os.path.basename(p)       # ckpt_00001000.h5
        num  = base.split("_")[1].split(".")[0]  # 00001000
        return int(num)

    # Scan for HDF5 checkpoints, sort numerically by step
    h5_files = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.h5")),
                       key=step_from_path)
    # Scan for NPZ checkpoints
    npz_files = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.npz")),
                        key=step_from_path)

    # Pick the latest across both formats
    latest_h5  = h5_files[-1]  if h5_files  else None
    latest_npz = npz_files[-1] if npz_files else None

    # Determine which is more recent
    path = None
    fmt  = None
    if latest_h5 and latest_npz:
        if step_from_path(latest_h5) >= step_from_path(latest_npz):
            path, fmt = latest_h5, "h5"
        else:
            path, fmt = latest_npz, "npz"
    elif latest_h5:
        path, fmt = latest_h5, "h5"
    elif latest_npz:
        path, fmt = latest_npz, "npz"
    else:
        return None

    if fmt == "h5" and HAS_H5:
        with h5py.File(path, "r") as f:
            # Geometry compatibility check
            ck_Nx = int(f.attrs.get("Nx", -1))
            ck_Ny = int(f.attrs.get("Ny", -1))
            ck_Nz = int(f.attrs.get("Nz", -1))
            ck_Lx = float(f.attrs.get("Lx", -1.0))
            ck_Ly = float(f.attrs.get("Ly", -1.0))
            ck_Lz = float(f.attrs.get("Lz", -1.0))
            mismatches = []
            if ck_Nx != Nx: mismatches.append(f"Nx (ckpt={ck_Nx} vs run={Nx})")
            if ck_Ny != Ny: mismatches.append(f"Ny (ckpt={ck_Ny} vs run={Ny})")
            if ck_Nz != Nz: mismatches.append(f"Nz (ckpt={ck_Nz} vs run={Nz})")
            if abs(ck_Lx - Lx) > 1e-9: mismatches.append(f"Lx (ckpt={ck_Lx} vs run={Lx})")
            if abs(ck_Ly - Ly) > 1e-9: mismatches.append(f"Ly (ckpt={ck_Ly} vs run={Ly})")
            if abs(ck_Lz - Lz) > 1e-9: mismatches.append(f"Lz (ckpt={ck_Lz} vs run={Lz})")
            if mismatches:
                sys.exit(f"ERROR: checkpoint geometry mismatch ({path}):\n  "
                         + "\n  ".join(mismatches)
                         + "\n  Either match these settings on the command line, "
                         + "delete the checkpoint, or use a different --outdir.")

            d = {
                "step":    int(f.attrs["step"]),
                "t_phys":  float(f.attrs["t_phys"]),
                "h_w":     f["h_w"][:],
                "S_n":     f["S_n"][:],
                "time_log":            f["time_log"][:],
                "l2_log":              f["l2_log"][:],
                "l2n_log":             f["l2n_log"][:],
                "Sn_max_log":          f["Sn_max_log"][:],
                "napl_mass_log":       f["napl_mass_log"][:],
                "napl_mass_nosrc_log": f["napl_mass_nosrc_log"][:],
            }
        print(f"  Restarting from checkpoint: {path}")
        print(f"    step = {d['step']},  t = {d['t_phys']:.4e} s")
        return d

    elif fmt == "npz":
        d_raw = np.load(path, allow_pickle=True)
        d = {k: d_raw[k] for k in d_raw.files}
        d["step"]   = int(d["step"])
        d["t_phys"] = float(d["t_phys"])

        # Geometry compatibility check (only if checkpoint has geometry attrs)
        if "Nx" in d_raw.files:
            ck_Nx = int(d_raw["Nx"]); ck_Ny = int(d_raw["Ny"]); ck_Nz = int(d_raw["Nz"])
            ck_Lx = float(d_raw["Lx"]); ck_Ly = float(d_raw["Ly"]); ck_Lz = float(d_raw["Lz"])
            mismatches = []
            if ck_Nx != Nx: mismatches.append(f"Nx (ckpt={ck_Nx} vs run={Nx})")
            if ck_Ny != Ny: mismatches.append(f"Ny (ckpt={ck_Ny} vs run={Ny})")
            if ck_Nz != Nz: mismatches.append(f"Nz (ckpt={ck_Nz} vs run={Nz})")
            if abs(ck_Lx - Lx) > 1e-9: mismatches.append(f"Lx (ckpt={ck_Lx} vs run={Lx})")
            if abs(ck_Ly - Ly) > 1e-9: mismatches.append(f"Ly (ckpt={ck_Ly} vs run={Ly})")
            if abs(ck_Lz - Lz) > 1e-9: mismatches.append(f"Lz (ckpt={ck_Lz} vs run={Lz})")
            if mismatches:
                sys.exit(f"ERROR: checkpoint geometry mismatch ({path}):\n  "
                         + "\n  ".join(mismatches)
                         + "\n  Either match these settings on the command line, "
                         + "delete the checkpoint, or use a different --outdir.")

        print(f"  Restarting from NPZ checkpoint: {path}")
        print(f"    step = {d['step']},  t = {d['t_phys']:.4e} s")
        return d

    return None


# ======================================================================
# 13.  MAIN SIMULATION LOOP
# ======================================================================

N_steps_max    = args.n_steps
snapshot_every = args.snapshot_every
print_every    = args.print_every
ckpt_every     = args.ckpt_every
ss_tol         = 1e-14

# --- Attempt restart from checkpoint ---
# Only rank 0 reads the checkpoint file; the data is then scattered.
if MPI_RANK == 0:
    ckpt = load_latest_checkpoint()
else:
    ckpt = None

# Broadcast existence-and-metadata to all ranks (so they agree on start_step)
if USE_MPI:
    ckpt_exists = (ckpt is not None) if MPI_RANK == 0 else None
    ckpt_exists = _comm.bcast(ckpt_exists, root=0)
else:
    ckpt_exists = (ckpt is not None)

if ckpt_exists:
    # Scalar fields: broadcast
    if USE_MPI:
        if MPI_RANK == 0:
            scalars = (int(ckpt["step"]), float(ckpt["t_phys"]),
                       np.array(ckpt["time_log"]),
                       np.array(ckpt["l2_log"]),
                       np.array(ckpt["l2n_log"]),
                       np.array(ckpt["Sn_max_log"]),
                       np.array(ckpt["napl_mass_log"]),
                       np.array(ckpt["napl_mass_nosrc_log"]))
        else:
            scalars = None
        scalars = _comm.bcast(scalars, root=0)
        (start_step_m1, t_phys, _tl, _l2, _l2n, _snm, _nm, _nmn) = scalars
        start_step = start_step_m1 + 1
        time_log            = list(_tl)
        l2_log              = list(_l2)
        l2n_log             = list(_l2n)
        Sn_max_log          = list(_snm)
        napl_mass_log       = list(_nm)
        napl_mass_nosrc_log = list(_nmn)
    else:
        start_step = int(ckpt["step"]) + 1
        t_phys     = float(ckpt["t_phys"])
        time_log            = list(ckpt["time_log"])
        l2_log              = list(ckpt["l2_log"])
        l2n_log             = list(ckpt["l2n_log"])
        Sn_max_log          = list(ckpt["Sn_max_log"])
        napl_mass_log       = list(ckpt["napl_mass_log"])
        napl_mass_nosrc_log = list(ckpt["napl_mass_nosrc_log"])

    # Per-particle fields: scatter from rank 0 → each rank's owned slice
    if USE_MPI:
        # Rank 0 holds the full global h_w, S_n arrays; scatter the
        # OWNED slices indexed by each rank's global_idx_owned.
        if MPI_RANK == 0:
            hw_global = np.asarray(ckpt["h_w"], dtype=np.float64)
            Sn_global = np.asarray(ckpt["S_n"], dtype=np.float64)
            # Rearrange via _gather_perm: the gather collected rank-
            # concatenated data with _gather_perm[k] = global index of
            # the k-th rank-concatenated owned particle. So to scatter
            # back into per-rank-owned slices, we take hw_global[_gather_perm]
            # which gives the rank-concatenated layout, then split by counts.
            hw_concat = hw_global[_gather_perm]
            Sn_concat = Sn_global[_gather_perm]
            displs = np.cumsum([0] + _counts_owned[:-1]).tolist()
        else:
            hw_concat = None
            Sn_concat = None
            displs = None

        # Scatterv requires careful buffer setup
        hw_owned_local = np.empty(N_OWNED, dtype=np.float64)
        Sn_owned_local = np.empty(N_OWNED, dtype=np.float64)
        if MPI_RANK == 0:
            sendbuf_hw = (hw_concat, _counts_owned, displs,
                          MPI._typedict[np.dtype(np.float64).char])
            sendbuf_Sn = (Sn_concat, _counts_owned, displs,
                          MPI._typedict[np.dtype(np.float64).char])
        else:
            sendbuf_hw = None
            sendbuf_Sn = None
        _comm.Scatterv(sendbuf=sendbuf_hw, recvbuf=hw_owned_local, root=0)
        _comm.Scatterv(sendbuf=sendbuf_Sn, recvbuf=Sn_owned_local, root=0)

        h_w[:N_OWNED] = hw_owned_local
        S_n[:N_OWNED] = Sn_owned_local
        # Refresh ghost layers to match
        exchange_ghosts(h_w, S_n)
    else:
        h_w[:] = ckpt["h_w"]
        S_n[:] = ckpt["S_n"]

    log_root(f"  Restarting from checkpoint at step {start_step - 1}, "
             f"t = {t_phys:.4e} s")

    # --- Purge stale snapshots/checkpoints past the restart step ---
    # With one-file-per-snapshot, this is a simple filesystem unlink:
    # delete any step_NNN.{h5,npz} file or ckpt_NNN.{h5,npz} file whose
    # step number is greater than the checkpoint we just restored from.
    # Only rank 0 touches files; other ranks wait at the barrier below.
    if MPI_RANK == 0:
        ckpt_step = start_step - 1
        import glob

        # Stale snapshot files (per-file format: snapshots/step_NNN.{h5,npz})
        n_snap = 0
        for pat in ["step_*.h5", "step_*.npz"]:
            for s_path in glob.glob(os.path.join(SNAP_DIR, pat)):
                base = os.path.basename(s_path)
                s_step = int(base.split("_")[1].split(".")[0])
                if s_step > ckpt_step:
                    os.remove(s_path)
                    n_snap += 1
        if n_snap > 0:
            print(f"  Purged {n_snap} stale snapshot file(s) (step > {ckpt_step})")

        # Stale checkpoint files
        n_ckpt = 0
        for pat in ["ckpt_*.h5", "ckpt_*.npz"]:
            for cp_path in glob.glob(os.path.join(CKPT_DIR, pat)):
                base = os.path.basename(cp_path)
                cp_step = int(base.split("_")[1].split(".")[0])
                if cp_step > ckpt_step:
                    os.remove(cp_path)
                    n_ckpt += 1
        if n_ckpt > 0:
            print(f"  Removed {n_ckpt} stale checkpoint file(s) (step > {ckpt_step})")

    if USE_MPI:
        _comm.Barrier()
else:
    start_step = 1
    t_phys     = 0.0
    time_log       = []
    l2_log         = []
    l2n_log        = []
    Sn_max_log     = []
    napl_mass_log      = []
    napl_mass_nosrc_log = []

# History of Sn snapshots (only rank 0 keeps; gathered from all ranks)
Sn_history = []       # list of (step, t, S_n_global_array)
Sn_snap_every = (args.sn_snap_every
                 if args.sn_snap_every > 0
                 else max(1, N_steps_max // 40))   # ~40 frames for animation

log_root(f"\n{'='*70}")
log_root(f"  THREE-PHASE SPH: Water + LNAPL + Air")
log_root(f"  Steps [{start_step}, {N_steps_max}],  NAPL source at "
         f"[{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}] x [{SRC_Z0},{SRC_Z1}]")
log_root(f"{'='*70}")
t_wall0 = wall_time.time()

mask_conv = (ptype == 0) | (ptype == 3) | (ptype == 4)
mask_not_src = ~is_source & mask_conv

# Save initial Sn snapshot (gather to rank 0)
if start_step == 1:
    if USE_MPI:
        S_n_global0 = _gather_to_global_array(S_n[:N_OWNED])
        if MPI_RANK == 0:
            Sn_history.append((0, 0.0, S_n_global0))
    else:
        Sn_history.append((0, 0.0, S_n.copy()))

step = start_step - 1   # default if loop is skipped (restart at end)
for step in range(start_step, N_steps_max + 1):
    # CFL timestep: each rank computes its local minimum; take global min
    dt_local = stable_dt(h_w, S_n)
    dt = comm_allreduce_min(dt_local)

    # SPH RHS over the whole local array. Ghost particles get bad values
    # (their full neighbour stencils aren't available locally), but those
    # values are discarded — we only use dhdt[:N_OWNED] / dSndt[:N_OWNED].
    dhdt  = compute_dhdt(h_w, S_n)
    dSndt = compute_dSndt(h_w, S_n)

    # Forward-Euler: update OWNED only. Ghost values stay at the previous
    # time level until the exchange below refreshes them.
    h_w[:N_OWNED] += dt * dhdt[:N_OWNED]
    S_n[:N_OWNED] += dt * dSndt[:N_OWNED]
    S_n[:N_OWNED] = np.clip(S_n[:N_OWNED], 0.0, 1.0 - S_res)

    # BC enforcement: each rank handles its own slice of boundary particles.
    # Edge ranks have non-empty Dirichlet index arrays; internal ranks are
    # no-ops there.
    enforce_dirichlet_bc(h_w)
    enforce_impermeable_bc(h_w)
    enforce_napl_bc(S_n)

    # Refresh ghost layer so next step's SPH operators see consistent data
    if USE_MPI:
        exchange_ghosts(h_w, S_n)

    t_phys += dt

    # Convergence metrics — local sum, then allreduce to global.
    # Restrict masks to owned-only via slicing.
    mc_owned  = mask_conv[:N_OWNED]
    mns_owned = mask_not_src[:N_OWNED]

    l2_sq_local   = float(np.sum(dhdt[:N_OWNED][mc_owned]**2))
    l2_cnt_local  = int(np.sum(mc_owned))
    l2n_sq_local  = float(np.sum(dSndt[:N_OWNED][mns_owned]**2))
    l2n_cnt_local = int(np.sum(mns_owned))
    sn_max_local  = (float(np.max(S_n[:N_OWNED][~is_source[:N_OWNED]]))
                     if np.any(~is_source[:N_OWNED]) else 0.0)
    napl_mass_total_local = float(np.sum(S_n[:N_OWNED] * Vp * phi_0))
    napl_mass_nosrc_local = float(np.sum(S_n[:N_OWNED][~is_source[:N_OWNED]]
                                          * Vp * phi_0))

    l2_sq           = comm_allreduce_sum(l2_sq_local)
    l2_cnt          = comm_allreduce_sum(l2_cnt_local)
    l2n_sq          = comm_allreduce_sum(l2n_sq_local)
    l2n_cnt         = comm_allreduce_sum(l2n_cnt_local)
    sn_max          = comm_allreduce_max(sn_max_local)
    napl_mass_total = comm_allreduce_sum(napl_mass_total_local)
    napl_mass_nosrc = comm_allreduce_sum(napl_mass_nosrc_local)

    l2  = np.sqrt(l2_sq / max(l2_cnt, 1))
    l2n = np.sqrt(l2n_sq / max(l2n_cnt, 1)) if l2n_cnt > 0 else 0.0

    time_log.append(t_phys)
    l2_log.append(l2)
    l2n_log.append(l2n)
    Sn_max_log.append(sn_max)
    napl_mass_log.append(napl_mass_total)
    napl_mass_nosrc_log.append(napl_mass_nosrc)

    # Print (rank 0 only in MPI mode)
    if step % print_every == 0 or step == start_step:
        log_root(f"  step {step:5d}  t = {t_phys:.4e} s  dt = {dt:.4e} s  "
                 f"L2w = {l2:.4e}  L2n = {l2n:.4e}  Sn_max = {sn_max:.4f}  "
                 f"NAPL_out = {napl_mass_nosrc:.4e}")

    # HDF5 snapshot — uses save_snapshot which is MPI-aware (see Section 12).
    if step % snapshot_every == 0 or step == start_step:
        qx_s,  qy_s,  qz_s  = compute_darcy_velocity(h_w, S_n)
        qxn_s, qyn_s, qzn_s = compute_darcy_velocity_napl(h_w, S_n)
        save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt, dSndt,
                       qx_s, qy_s, qz_s, qxn_s, qyn_s, qzn_s)

    # Sn snapshot for animation: gather to rank 0, only rank 0 stores
    if step % Sn_snap_every == 0 or step == start_step:
        if USE_MPI:
            S_n_global = _gather_to_global_array(S_n[:N_OWNED])
            if MPI_RANK == 0:
                Sn_history.append((step, t_phys, S_n_global))
        else:
            Sn_history.append((step, t_phys, S_n.copy()))

    # Checkpoint (MPI-aware)
    if step % ckpt_every == 0:
        save_checkpoint(step, t_phys, h_w, S_n,
                        time_log, l2_log, l2n_log,
                        Sn_max_log, napl_mass_log, napl_mass_nosrc_log)

    if l2 < ss_tol and step > 10:
        log_root(f"\n  *** Converged at step {step}:  L2 = {l2:.4e}  < {ss_tol:.1e}")
        break

t_wall = wall_time.time() - t_wall0
log_root(f"\n  Wall time: {t_wall:.1f} s    Physical time: {t_phys:.4e} s")

ran_steps = (step >= start_step)

if ran_steps:
    # Final Sn snapshot — gather to rank 0 in MPI mode
    if USE_MPI:
        S_n_global_final = _gather_to_global_array(S_n[:N_OWNED])
        if MPI_RANK == 0:
            Sn_history.append((step, t_phys, S_n_global_final))
    else:
        Sn_history.append((step, t_phys, S_n.copy()))

    # Final checkpoint
    save_checkpoint(step, t_phys, h_w, S_n,
                    time_log, l2_log, l2n_log,
                    Sn_max_log, napl_mass_log, napl_mass_nosrc_log)

# Final velocities
log_root("Computing final Darcy velocity ...", flush=True)
qx_final,  qy_final,  qz_final  = compute_darcy_velocity(h_w, S_n)
qxn_final, qyn_final, qzn_final = compute_darcy_velocity_napl(h_w, S_n)
dhdt_f = compute_dhdt(h_w, S_n)
dSndt_f = compute_dSndt(h_w, S_n)
save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt_f, dSndt_f,
               qx_final, qy_final, qz_final, qxn_final, qyn_final, qzn_final)
log_root("  done.")


# ======================================================================
# 14.  POST-PROCESSING  —  minimal final figures on rank 0 only
# ======================================================================
#
# Heavy slice plots and animations are handled by the standalone
# postprocess_napl.py script reading the HDF5 snapshot file.
# Here we only produce three small diagnostic plots from data that
# rank 0 already has (or can cheaply gather):
#
#   Fig 4 - NAPL mass balance     (uses napl_mass_log time series)
#   Fig 5 - Convergence           (uses l2_log / l2n_log / Sn_max_log)
#   Fig 6 - Saturation vs z       (vertical column at source center)
#
# Plus a SUMMARY block printed on rank 0 with key final-state metrics.

# Slice indices: at NAPL source center (needed for Fig 6 and the summary)
ix_src = int(round(0.5 * (SRC_X0 + SRC_X1) / dx))
iy_src = int(round(0.5 * (SRC_Y0 + SRC_Y1) / dy))
ix_src = max(0, min(ix_src, Nx - 1))
iy_src = max(0, min(iy_src, Ny - 1))

# Gather the final state to rank 0 for Fig 6.
# In single-process mode this is a no-op identity copy.
if USE_MPI:
    h_w_global  = _gather_to_global_array(h_w [:N_OWNED])
    S_n_global  = _gather_to_global_array(S_n [:N_OWNED])
    qx_g        = _gather_to_global_array(qx_final[:N_OWNED])
    qy_g        = _gather_to_global_array(qy_final[:N_OWNED])
    qz_g        = _gather_to_global_array(qz_final[:N_OWNED])
else:
    h_w_global, S_n_global = h_w, S_n
    qx_g, qy_g, qz_g = qx_final, qy_final, qz_final

# Local max(|q_w|) → global via allreduce (used in summary)
qmag_local_max = float(np.max(np.sqrt(qx_final[:N_OWNED]**2
                                       + qy_final[:N_OWNED]**2
                                       + qz_final[:N_OWNED]**2)))
qmag_global_max = comm_allreduce_max(qmag_local_max)

# Local max(S_n outside source) → global
sn_max_out_local = (float(np.max(S_n[:N_OWNED][~is_source[:N_OWNED]]))
                    if np.any(~is_source[:N_OWNED]) else 0.0)
sn_max_out = comm_allreduce_max(sn_max_out_local)

# Local |h_final - h_initial| → global
# (h_init was set BEFORE any time integration, in the initial-condition block,
# but it has length N_LOCAL too — including ghosts, which never change since
# ghosts are overwritten by exchange of consistent values. Safe to slice owned.)
dh_max_local = float(np.max(np.abs(h_w[:N_OWNED] - h_init[:N_OWNED])))
dh_max = comm_allreduce_max(dh_max_local)

# Local NAPL mass → global (sum)
napl_mass_total_local = float(np.sum(S_n[:N_OWNED] * Vp * phi_0))
napl_mass_total = comm_allreduce_sum(napl_mass_total_local)

# Final plots — only on rank 0
if MPI_RANK == 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Fig 4: NAPL mass balance ─────────────────────────────────────
    if len(time_log) > 0:
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
        ax4a.plot(time_log, napl_mass_log,       'k-', lw=1.5,
                  label="Total (incl. source)")
        ax4a.plot(time_log, napl_mass_nosrc_log, 'r-', lw=1.5,
                  label="Outside source")
        ax4a.set_xlabel("Time [s]")
        ax4a.set_ylabel(r"NAPL volume $\phi S_n V$  [m$^3$]")
        ax4a.set_title("NAPL Mass Balance")
        ax4a.legend()
        ax4a.grid(True, alpha=0.3)

        if len(napl_mass_nosrc_log) > 2:
            t_arr = np.array(time_log)
            m_arr = np.array(napl_mass_nosrc_log)
            rate = np.diff(m_arr) / np.maximum(np.diff(t_arr), 1e-30)
            ax4b.plot(t_arr[1:], rate, 'r-', lw=1)
            ax4b.set_xlabel("Time [s]")
            ax4b.set_ylabel(r"d(NAPL)/dt [m$^3$/s]")
            ax4b.set_title("Source Discharge Rate")
            ax4b.grid(True, alpha=0.3)

        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_napl_mass_balance.png"),
                     dpi=150, bbox_inches="tight")
        print("  Saved fig4_napl_mass_balance.png")
        plt.close(fig4)

    # ── Fig 5: Convergence + NAPL front ──────────────────────────────
    if len(time_log) > 0:
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
        ax5a.semilogy(time_log, l2_log,  "b-", lw=1, label=r"L2($dh_w/dt$)")
        ax5a.semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
        ax5a.set_xlabel("Time [s]")
        ax5a.set_ylabel("Residual")
        ax5a.set_title("Convergence")
        ax5a.legend()
        ax5a.grid(True, which="both", alpha=0.3)

        ax5b.plot(time_log, Sn_max_log, "r-", lw=1.5)
        ax5b.set_xlabel("Time [s]")
        ax5b.set_ylabel(r"max $S_n$ (outside source)")
        ax5b.set_title("NAPL Front")
        ax5b.grid(True, alpha=0.3)

        fig5.tight_layout()
        fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_convergence.png"),
                     dpi=150, bbox_inches="tight")
        print("  Saved fig5_convergence.png")
        plt.close(fig5)

    # ── Fig 6: Saturation profiles at source column ──────────────────
    # Vertical line at (ix_src, iy_src) — column in z.
    # h_w_global / S_n_global are in global particle order:
    #   global_index = i * Ny * Nz + j * Nz + k
    # So the column at (ix_src, iy_src) is contiguous:
    col_start = ix_src * (Ny * Nz) + iy_src * Nz
    col_end   = col_start + Nz
    h_col  = h_w_global[col_start:col_end]
    Sn_col = S_n_global[col_start:col_end]
    Sw_col = compute_Sw_3ph(h_col, Sn_col)
    Sa_col = np.clip(1.0 - Sw_col - Sn_col, 0, 1)
    z_col = np.arange(Nz) * dz

    fig6, ax6 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax6.plot(Sw_col, z_col, "b-",  lw=2,   label=r"$S_w$")
    ax6.plot(Sn_col, z_col, "r-",  lw=2,   label=r"$S_n$")
    ax6.plot(Sa_col, z_col, "g--", lw=1.5, label=r"$S_a$")
    ax6.set_xlabel("Saturation [-]")
    ax6.set_ylabel("z [m]")
    ax6.set_title(f"Saturation profiles at "
                  f"(x={ix_src*dx:.2f}, y={iy_src*dy:.2f}) m")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.05, 1.05)
    fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_saturation_profiles.png"),
                 dpi=150, bbox_inches="tight")
    print("  Saved fig6_saturation_profiles.png")
    plt.close(fig6)


# ── Summary (rank 0 only) ────────────────────────────────────────────
q_expected = k_sat * (H_u - H_d) / Lx

log_root(f"\n{'='*70}")
log_root(f"  SUMMARY  (3D Three-phase: Water + LNAPL + Air)")
log_root(f"{'='*70}")
log_root(f"  Domain                        = {Lx} x {Ly} x {Lz} m")
log_root(f"  Grid                          = {Nx} x {Ny} x {Nz} = "
         f"{N_part_global} particles")
if USE_MPI:
    log_root(f"  MPI ranks                     = {MPI_NRANK} ({Px}×{Py} pencil)")
log_root(f"  Max |h_final - h_initial|     = {dh_max:.6e} m")
log_root(f"  Final L2(dh_w/dt)             = {l2_log[-1] if l2_log else 0:.6e}")
log_root(f"  Final L2(dS_n/dt)             = {l2n_log[-1] if l2n_log else 0:.6e}")
log_root(f"  Max Darcy |q_w|               = {qmag_global_max:.6e} m/s")
log_root(f"  Max S_n (outside source)      = {sn_max_out:.6f}")
log_root(f"  Total NAPL volume (phi*Sn*V)  = {napl_mass_total:.6e} m^3")
log_root(f"  k_sat (water)                 = {k_sat:.6e} m/s")
log_root(f"  k_sat (NAPL)                  = {k_sat_n:.6e} m/s")
log_root(f"  Expected q_w ~ k_sat*dH/Lx    = {q_expected:.6e} m/s")
if HAS_H5:
    log_root(f"  HDF5 snapshots                = {SNAP_DIR}/  (one file per step)")
else:
    log_root(f"  NPZ snapshots                 = {SNAP_DIR}/  (one file per step)")
log_root(f"  Checkpoints                   = {CKPT_DIR}/")
log_root(f"{'='*70}")
log_root("\nDone.")
