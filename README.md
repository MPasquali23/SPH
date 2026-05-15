# SPH 3D Three-Phase Seepage

A particle-based simulator for transient seepage of water, NAPL (a
non-aqueous phase liquid, e.g. oil contaminant), and air through an
unsaturated porous medium. Built around the SPH framework of
[Lian et al. (2021), *CMAME* **387**, 114169](https://doi.org/10.1016/j.cma.2021.114169).

## What it solves

A coupled three-phase Darcy system on a uniform particle lattice:

* **Water** pressure head `h_w(x, t)` evolves by the SPH form of Darcy's
  equation, with hydraulic conductivity that depends on water and NAPL
  saturations through Van Genuchten constitutive relations.
* **NAPL** saturation `S_n(x, t)` evolves by an advective transport
  equation driven by the NAPL phase pressure (water + capillary +
  buoyancy terms).
* **Air** is the residual phase: `S_a = 1 - S_w - S_n`.
* Boundary conditions: Dirichlet water heads on x=0 and x=Lx (regional
  flow), impermeable mirror walls on the other four faces, prescribed
  NAPL source bbox at the top of the domain.

Output is a per-snapshot HDF5 file with positions, both phase
saturations, hydraulic head, Darcy velocities (water + NAPL), and
boundary metadata. Use the postprocess script for animations and figures.

## Components

| File                             | What it is                                                                                                                                                                          | When to use |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `sph_seepage_napl_3D.py`         | CPU simulator (Numba JIT). Single-process when run directly; Use `NUMBA_NUM_THREADS=XX` to run on multiple threads OMP-like.                                                          | All non-GPU runs. The same file covers both single-node Numba and multi-node MPI; no switch needed beyond the launch command. |
| `sph_seepage_napl_MPI.py`        | (not tested!) CPU simulator (Numba JIT). Single-process when run directly; MPI-parallel when launched with `mpirun -np N`.                                                          | All non-GPU runs. The same file covers both single-node Numba and multi-node MPI; no switch needed beyond the launch command. |
| `sph_seepage_napl_gpu_zorder.py` | GPU simulator (Numba CUDA). Z-order particle reordering for memory coalescence.                                                                                                     | Single-GPU runs on A100, V100, P100, RTX-class cards. |
| `postprocess_napl.py`            | Reads the per-snapshot output directory, produces two animations (slice-scatter + saturation-profile) and two 4-panel static figures. Streaming I/O (one file at a time per worker). | After any simulation run, regardless of which simulator produced it. |
run_scaling_3D.py and benchmark_3d.py all used to test CPU performances.

The four bullets below describe each more fully.

### CPU — Numba single-process or multi-process (OMP-like)

`sph_seepage_napl_MPI.py`, run without `mpirun` (NOT TESTED!) or `sph_seepage_napl_3D.py`. 
The script auto-detects that it's a single process 
(or multiple processes if NUMBA_NUM_THREADS is set) and uses the 
Numba JIT loops directly. Good for:

* Small grids (~50³) on a workstation or one HPC node
* Validation runs before scaling up
* Anywhere `mpi4py` isn't installed

Numba JIT compiles the per-particle SPH operator kernels on first call;
expect ~30 s of compile time at the start.

### CPU simulator — MPI multi-node

(NOT TESTED!!!!)
Same script, launched with `mpirun -np N` (N > 1). Activates a 2D pencil
decomposition along (x, y) keeping z whole per rank — picked to match
the physics, where NAPL migrates downward through a vertically-coupled
column and laterally across the regional flow. Each rank owns a pencil
plus a 3-cell ghost layer on its 4 horizontal neighbours; two-pass
sendrecv exchanges keep the ghost data current each step (corners
propagate via the second pass, so no explicit diagonal communication).

Activate with the conventional launch:

```bash
mpirun -np 16 python sph_seepage_napl_MPI.py --nx 101 --ny 101 --nz 101 ...
```

The script picks `(Px, Py)` automatically to match domain aspect ratio;
override with `--mpi-px` / `--mpi-py` if you want a specific layout.
First-run sanity: add `--mpi-validate` to enable a position-consistency
check after the initial ghost exchange. Bit-identical agreement with
single-process is expected to within floating-point reduction noise
(~1e-13 relative on the final L2 metrics).

### GPU simulator — Numba CUDA

`sph_seepage_napl_gpu_zorder.py`. Same physics, kernels written in
`@cuda.jit`. Particles are reordered into Z-order (Morton curve) at
setup so that neighbour reads hit nearby cache lines — typically ~1.5-2×
speedup on memory-bandwidth-bound data-center GPUs.

Tunables: `--gpu-block-size`, `--gpu-dt-every` (recompute the CFL dt
only every N steps to save a kernel launch), `--gpu-metrics-every`
(host-side L2 norms). `--gpu-verify-cpu` runs a CPU comparison every
snapshot — very slow but invaluable on first run. `--no-gpu-zorder`
disables Z-order if you want to compare timings.

GPU code is single-device. No multi-GPU support in this version.

### Postprocess

`postprocess_napl.py`. Reads `OUTDIR/snapshots/step_*.h5` (or `.npz`)
files one at a time per worker, renders each to its own PNG, then
ffmpeg-stitches the PNGs into MP4.

Produces four outputs in `--outdir`:

* `anim_slices_rgb.mp4` — animated YZ + XZ slice scatter through the
  source column. RGB encoding: R = NAPL, G = air, B = water. Streamlines
  for the water (blue) and NAPL (red) Darcy velocity fields. Time
  evolves frame-by-frame.
* `anim_profile_xy.mp4` — animated saturation profiles (S_w, S_n, S_a)
  versus z at the source column. Water-table elevation marked as a
  horizontal dashed line.
* `static_slices_4panels.png` — same slice view as the animation, but
  4 stacked rows at 4 time points (first, 1/3, 2/3, last).
* `static_profiles_4panels.png` — 2×2 grid of saturation profiles at
  the same 4 time points.

## Quick start

Single-process CPU, 51³ default grid, 2000 steps:

```bash
python sph_seepage_napl_3D.py --outdir runs/test01
# or (NOT TESTED!!)
python sph_seepage_napl_MPI.py --outdir runs/test01
python postprocess_napl.py --input runs/test01 --outdir runs/test01/post
```

The postprocess accepts either `--input runs/test01` (auto-finds the
`snapshots/` subdirectory inside) or `--input runs/test01/snapshots`
directly.

## Running on HPC

### Conventions

* The simulation defaults write one HDF5 file per snapshot to
  `OUTDIR/snapshots/step_NNNNNNNNNNNN.h5`.

### CPU single-node (Numba)

```bash
#!/bin/bash
#SBATCH --job-name=sph_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

module load gnu8/8.3.0
module load hdf5/1.10.5
module load miniconda3

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env_name

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python sph_seepage_napl_3D.py \
    --nx 101 --ny 101 --nz 101 \
    --n-steps 100000 \
    --snapshot-every 500 \
    --ckpt-every 2000 \
    --outdir $SCRATCH/sph_runs/run_${SLURM_JOB_ID}
```

Numba uses the OpenMP threading layer by default; `NUMBA_NUM_THREADS`
caps it to your allocated cores. Without this, Numba will try to use all
physical cores on the node and either oversubscribe or get throttled
by the scheduler.

### CPU multi-node (MPI) --> NOT TESTED!!!

```bash
#!/bin/bash
#SBATCH --job-name=sph_mpi
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

# Numba threading inside each MPI rank — keep small or disable, since
# threads compete with MPI ranks for cores.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gnu8/8.3.0
module load hdf5/1.10.5
module load miniconda3

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env_name_MPI


mpirun -np $SLURM_NTASKS \
    python sph_seepage_napl_MPI.py \
        --nx 201 --ny 201 --nz 201 \
        --n-steps 500000 \
        --snapshot-every 1000 \
        --ckpt-every 5000 \
        --mpi-validate \
        --outdir $SCRATCH/sph_runs/run_${SLURM_JOB_ID}
```

`--mpi-validate` is cheap and worth keeping for first runs; remove for
production. The pencil layout auto-picks `(Px, Py)` to be near-square
matching the domain aspect ratio; on 96 ranks with a square domain you
get a 12×8 or 16×6 grid. Override with `--mpi-px` / `--mpi-py` to lock
the layout.

Restart on a different rank count is supported — the checkpoint stores
no rank-specific info. You can interrupt a 48-rank run and resume on 96
ranks just by changing `--ntasks-per-node` in the SLURM script.

### GPU

```bash
#!/bin/bash
#SBATCH --job-name=sph_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

module load gnu8/8.3.0
module load hdf5/1.10.5
module load cuda/12.4.1
module load miniconda3

# needed on UNIPR HPC
echo "workaround for broken cuda module: export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate cuda_env

python sph_seepage_napl_gpu_zorder.py \
    --nx 201 --ny 201 --nz 201 \
    --n-steps 500000 \
    --snapshot-every 1000 \
    --ckpt-every 5000 \
    --gpu-block-size 128 \
    --gpu-dt-every 50 \
    --outdir $SCRATCH/sph_runs/run_${SLURM_JOB_ID}
```

### Postprocess

```bash
#!/bin/bash
#SBATCH --job-name=sph_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

module load gnu8/8.3.0
module load hdf5/1.10.5
module load miniconda3

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env_name_post_process

RUN=$SCRATCH/sph_runs/run_001

python postprocess_napl.py \
    --input $RUN \
    --outdir $RUN/post \
    --workers $SLURM_CPUS_PER_TASK \
    --fps 10 --dpi 200 --quality high \
    --mp-start-method spawn
```

Memory is the resource that matters here, not cores. The script prints
an estimate at startup: `Streaming workflow: each worker holds 1
snapshot at a time (~XXX MB / worker)`. Aim for at least 4 GB of `--mem`
per worker; less and you'll trip the bus errors described below.

## Output format

A run with `--outdir runs/myrun` produces:

```
runs/myrun/
├── snapshots/
│   ├── step_000000000001.h5      ← first snapshot
│   ├── step_000000001000.h5
│   ├── step_000000002000.h5
│   └── ...
└── checkpoints/
    ├── ckpt_000000005000.h5
    └── ckpt_000000010000.h5      ← latest; used for restart
```

Each snapshot file is a flat HDF5 with root-level attrs and datasets,
inspectable with `h5dump -H` or HDFView. The schema:

```
/                         (attributes)
  step                    int64    step number
  time_s                  float64  physical time, s
  Nx, Ny, Nz              int64    grid dimensions
  Lx, Ly, Lz              float64  domain extents, m
  k_sat, H_u, H_d         float64  saturated K and Dirichlet BC heads
  mpi_nrank, mpi_Px, mpi_Py   (only present on MPI runs)

/x  /y  /z                float64  particle positions, length N_part
/h                        float64  water pressure head
/H                        float64  hydraulic head (h + z)
/Sn /Sw                   float64  NAPL and water saturations
/kw /kn                   float64  relative permeabilities (water, NAPL)
/dhdt /dSndt              float64  RHS values at this snapshot
/qx /qy /qz               float64  water Darcy velocity components
/qxn /qyn /qzn            float64  NAPL Darcy velocity components
/ptype                    int64    boundary classification (0..6)
/is_source                int8     mask of NAPL source particles
```

Checkpoints `ckpt_NNN.h5` contain only what's needed to restart:
`h_w`, `S_n`, the time-series logs (L2 convergence, NAPL mass), and
metadata.

### Restart

If a checkpoint exists in `OUTDIR/checkpoints/`, the simulation
auto-resumes from the latest one. No flag needed. To start fresh, delete
the `checkpoints/` directory (or point `--outdir` somewhere else).

If a run wrote snapshots past the restart step before being killed,
those are removed automatically on the next start — they're from a
trajectory that's being overwritten. No `h5repack` needed.

## Troubleshooting

### Numba JIT compile blocking step 0

On first run, Numba spends 20-60 s compiling kernels before the first
time step. This is normal. On the cluster the compilation cache lives
in `~/.numba_cache` by default; subsequent runs of the same script
version skip the compile.

## Dependencies

* Python 3.9+
* `numpy`, `scipy`, `matplotlib`, `numba`
* `h5py` (with libhdf5; for HDF5 snapshots — script falls back to NPZ
  per-file if missing)
* `mpi4py` (only for MPI runs)
* `numba-cuda` with CUDA toolkit (only for GPU runs)
* `ffmpeg` (for postprocess MP4 output; falls back to GIF via Pillow if
  absent)
* `Pillow` (always recommended; GIF fallback)

A conda environment that covers everything:

```bash
srun --nodes=1 --ntasks-per-node=2 --partition=cpu --qos=cpu --mem=4G --pty bash
module load gnu8/8.3.0
module load hdf5/1.10.5
module load cuda/12.4.1
module load miniconda3
conda create -n env_name numpy scipy matplotlib numba h5py \
    numba-cuda mpi4py pillow ffmpeg -c conda-forge
conda activate env_name
# CUDA support: install cudatoolkit matching your driver
```
