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
* HDF5 snapshots at regular intervals.
* Matplotlib diagnostic figures.
"""

import numpy as np
import os
import sys
import time as wall_time

# ======================================================================
# 0.  OUTPUT DIRECTORY
# ======================================================================
OUTPUT_DIR = "/home/michele/Scaricati"
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
# 2.  DOMAIN  &  SPH DISCRETISATION
# ======================================================================
Lx, Ly = 10.0, 10.0

Nx = 51       # particles in x
Ny = 51       # particles in y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

print(f"\nDomain  {Lx} x {Ly} m    grid  {Nx} x {Ny}    "
      f"dx = {dx:.4f} m    dy = {dy:.4f} m")

# Regular lattice positions
xp = np.zeros(Nx * Ny)
yp = np.zeros(Nx * Ny)
idx_2d = np.zeros((Nx, Ny), dtype=int)

pid = 0
for i in range(Nx):
    for j in range(Ny):
        xp[pid] = i * dx
        yp[pid] = j * dy
        idx_2d[i, j] = pid
        pid += 1

N_part = Nx * Ny
Vp     = dx * dy        # particle area (2-D)
h_sml  = 1.3 * dx       # smoothing length

print(f"Particles: {N_part}    h_sml = {h_sml:.4f} m")

# ======================================================================
# 3.  CUBIC SPLINE KERNEL  (2-D)  [Paper Eq. 34]
# ======================================================================
alpha_d = 15.0 / (7.0 * np.pi * h_sml**2)

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

print("Building neighbour lists ...", flush=True)

# Each entry: (j, r, x_ji, y_ji, F_hat, gradWx, gradWy)
neighbours = [[] for _ in range(N_part)]

for i in range(N_part):
    xi, yi = xp[i], yp[i]
    ix = int(round(xi / dx))
    iy = int(round(yi / dy))
    d_max = int(np.ceil(support_r / dx)) + 1
    for ii in range(max(0, ix - d_max), min(Nx, ix + d_max + 1)):
        for jj in range(max(0, iy - d_max), min(Ny, iy + d_max + 1)):
            j = idx_2d[ii, jj]
            if j == i:
                continue
            xji = xp[j] - xi       # r_ji components
            yji = yp[j] - yi
            r   = np.sqrt(xji**2 + yji**2)
            if r >= support_r or r < 1e-30:
                continue

            dWr = dW_dr(r)
            # grad_i W  =  dW/dr * (x_i - x_j)/r  =  dW/dr * (-r_ji)/r
            gWx = dWr * (-xji) / r
            gWy = dWr * (-yji) / r

            # F_hat = (r_ji . grad_i W) / |r_ji|^2
            dot = xji * gWx + yji * gWy
            Fhat = dot / (r * r)

            neighbours[i].append((j, r, xji, yji, Fhat, gWx, gWy))

print("  done.")

# ======================================================================
# 5.  PRECOMPUTE SPH CORRECTION QUANTITIES
# ======================================================================
#
# Corrected gradient  (Eq. 36):
#   L_ij  =  [ sum_j V_j (x_j-x_i)^m  (grad_i W)^n ]^{-1}
#
# Corrected Laplacian  (Eq. 38):
#   K^norm  = 0.5 * sum_j V_j (x_ji^2 + y_ji^2) F_hat_ij
#   err^m   = sum_j V_j r^m_ji F_hat_ij

print("Precomputing correction matrices ...", flush=True)

K_norm = np.zeros(N_part)
err_x  = np.zeros(N_part)
err_y  = np.zeros(N_part)
L_inv  = np.zeros((N_part, 2, 2))

for i in range(N_part):
    Kn = 0.0; ex = 0.0; ey = 0.0
    M  = np.zeros((2, 2))
    for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
        Kn += Vp * (xji**2 + yji**2) * Fhat
        ex += Vp * xji * Fhat
        ey += Vp * yji * Fhat
        M[0, 0] += Vp * xji * gWx
        M[0, 1] += Vp * xji * gWy
        M[1, 0] += Vp * yji * gWx
        M[1, 1] += Vp * yji * gWy

    K_norm[i] = 0.5 * Kn
    err_x[i]  = ex
    err_y[i]  = ey

    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-30:
        L_inv[i, 0, 0] =  M[1, 1] / det
        L_inv[i, 0, 1] = -M[0, 1] / det
        L_inv[i, 1, 0] = -M[1, 0] / det
        L_inv[i, 1, 1] =  M[0, 0] / det
    else:
        L_inv[i] = np.eye(2)

print("  done.")

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
# 7.  PARTICLE CLASSIFICATION  &  BOUNDARY CONDITIONS
# ======================================================================
#
#   LEFT   x = 0  :  Dirichlet  H = H_u  = 8 m   [Eq. 67]
#   RIGHT  x = Lx :  Dirichlet  H = H_d  = 6 m   [Eq. 67]
#   BOTTOM y = 0  :  Impermeable  q.n = 0          [Eq. 73]
#   TOP    y = Ly :  Impermeable  q.n = 0          [Eq. 73]
#
# ptype:  0 = interior,  1 = left BC,  2 = right BC,  3 = bottom,  4 = top

H_u = 8.0
H_d = 6.0

ptype = np.zeros(N_part, dtype=int)
tol_bc = 0.5 * dx

for i in range(N_part):
    if   xp[i] < tol_bc:         ptype[i] = 1   # left
    elif xp[i] > Lx - tol_bc:    ptype[i] = 2   # right
    elif yp[i] < tol_bc:         ptype[i] = 3   # bottom
    elif yp[i] > Ly - tol_bc:    ptype[i] = 4   # top

n_left  = np.sum(ptype == 1)
n_right = np.sum(ptype == 2)
n_bot   = np.sum(ptype == 3)
n_top   = np.sum(ptype == 4)
n_int   = np.sum(ptype == 0)
print(f"\nParticle types:  interior={n_int}  left={n_left}  "
      f"right={n_right}  bottom={n_bot}  top={n_top}")

# -- Initial condition --
#
# The TRUE steady state satisfies  div[k(h) * grad(H)] = 0  with  H = h + z.
# Because  k  depends non-linearly on h (unsaturated zone), the true
# steady-state H(x,y) is NOT simply linear in x.  We initialise with
# the simple linear head  H(x) = H_u + (H_d - H_u) x / Lx  as an
# *approximate* initial guess, and let the solver converge to the true
# steady state.  The L2(dh/dt) will decrease during the transient
# adjustment, then *plateau* at a small numerical residual once the true
# steady state is reached - that plateau is the signal of convergence.

h_w = np.zeros(N_part)
for i in range(N_part):
    H_guess = H_u + (H_d - H_u) * xp[i] / Lx
    h_w[i]  = H_guess - yp[i]

# Enforce Dirichlet BCs on left/right
for i in range(N_part):
    if ptype[i] == 1:  h_w[i] = H_u - yp[i]
    if ptype[i] == 2:  h_w[i] = H_d - yp[i]

h_init = h_w.copy()

# ======================================================================
# 8.  IMPERMEABLE BOUNDARY ENFORCEMENT   [Paper Eq. 73-74]
# ======================================================================
#
# For top/bottom impermeable boundaries the paper (Eq. 74) prescribes that
# boundary-particle hydraulic properties mirror their nearest interior
# neighbour so that  dH/dn = 0.
#
#   H_boundary  =  H_interior_neighbour
#   =>  h_boundary  =  h_interior + (z_interior - z_boundary)

def enforce_impermeable_bc(h_field):
    """Mirror H from interior to top/bottom boundary rows."""
    for i in range(N_part):
        if ptype[i] == 3:      # bottom  y = 0
            ix = int(round(xp[i] / dx))
            ix = min(ix, Nx - 1)
            j_int = idx_2d[ix, 1]
            # H_bnd = H_int  =>  h_bnd = h_int + z_int - z_bnd
            h_field[i] = h_field[j_int] + yp[j_int] - yp[i]
        elif ptype[i] == 4:    # top  y = Ly
            ix = int(round(xp[i] / dx))
            ix = min(ix, Nx - 1)
            j_int = idx_2d[ix, Ny - 2]
            h_field[i] = h_field[j_int] + yp[j_int] - yp[i]
    return h_field


# ======================================================================
# 9.  RHS:  dh/dt  =  (1/C_tilde) div[ k grad(H) ]    [Paper Eq. 57]
# ======================================================================
#
# SPH form  (Eq. 57, isotropic  k^mn = k delta^mn):
#
#   dh_i/dt = (1/C_tilde) * (2/K_norm) *
#       [ sum_j V_j k_ij_mean (H_j - H_i) F_hat
#         - k_i * (grad_H_i . err_vec_i) ]
#
# where  k_ij_mean = (k_i + k_j)/2   (arithmetic mean)

def compute_dhdt(h_field):
    """RHS of the seepage equation for all particles."""
    k_h   = compute_k(h_field)
    Cs    = compute_Cs(h_field)
    H_f   = h_field + yp            # hydraulic head

    C_tilde = np.maximum(C_l + Cs, C_l)

    dhdt = np.zeros(N_part)

    for i in range(N_part):
        # Dirichlet particles - fixed
        if ptype[i] == 1 or ptype[i] == 2:
            continue

        Hi = H_f[i]
        ki = k_h[i]

        # -- corrected gradient of H (for error-correction term) --
        raw_gx = 0.0; raw_gy = 0.0
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            dH = H_f[j] - Hi
            raw_gx += Vp * dH * gWx
            raw_gy += Vp * dH * gWy
        Li = L_inv[i]
        grad_Hx = Li[0, 0] * raw_gx + Li[0, 1] * raw_gy
        grad_Hy = Li[1, 0] * raw_gx + Li[1, 1] * raw_gy

        # -- variable-k Laplacian term (Eq. 57 first sum) --
        lap_sum = 0.0
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            k_mean = 0.5 * (ki + k_h[j])
            lap_sum += Vp * k_mean * (H_f[j] - Hi) * Fhat

        # -- error-correction term (Eq. 57 second sum, isotropic) --
        corr = ki * (grad_Hx * err_x[i] + grad_Hy * err_y[i])

        Kn = K_norm[i]
        if abs(Kn) < 1e-30:
            continue

        rhs = (2.0 / Kn) * (lap_sum - corr)
        dhdt[i] = rhs / C_tilde[i]

    return dhdt


# ======================================================================
# 10.  DARCY VELOCITY   q = -k grad(H)    [Paper Eq. 60 + sign]
# ======================================================================
#
# Paper Eq. 60 computes  sum V_j k_ij (H_j-H_i) grad_W_ij  which is
# the SPH-consistent approximation of  k * grad(H).
# Physical Darcy velocity:  q = -k grad(H),  so we negate.

def compute_darcy_velocity(h_field):
    """Darcy velocity  q = -k grad(H)  using Eq. 60 (with sign correction)."""
    k_h = compute_k(h_field)
    H_f = h_field + yp

    qx = np.zeros(N_part)
    qy = np.zeros(N_part)

    for i in range(N_part):
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            k_mean = 0.5 * (k_h[i] + k_h[j])
            dH = H_f[j] - H_f[i]
            # Corrected kernel gradient
            L = L_inv[i]
            cWx = L[0, 0] * gWx + L[0, 1] * gWy
            cWy = L[1, 0] * gWx + L[1, 1] * gWy
            qx[i] += Vp * k_mean * dH * cWx
            qy[i] += Vp * k_mean * dH * cWy

    # q = -k grad(H)  =>  negate the SPH approximation of k*grad(H)
    return -qx, -qy


# ======================================================================
# 11.  TIME STEP   [Paper Eq. 66 / Appendix A.17]
# ======================================================================
CFL = 0.1

def stable_dt(h_field):
    Cs = compute_Cs(h_field)
    C_tilde = C_l + Cs
    C_min = max(np.min(C_tilde), C_l)
    return CFL * C_min * h_sml**2 / k_sat


# ======================================================================
# 12.  HDF5 SNAPSHOT I/O
# ======================================================================
try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False
    print("WARNING: h5py not available - HDF5 snapshots disabled.")

hdf5_path = os.path.join(OUTPUT_DIR, "sph_seepage_snapshots.h5")

def save_snapshot(fname, step, t, h_field, dhdt, qx, qy):
    """Append a snapshot group to the HDF5 file."""
    if not HAS_H5:
        return
    mode = "a" if os.path.exists(fname) else "w"
    with h5py.File(fname, mode) as f:
        grp_name = f"step_{step:06d}"
        if grp_name in f:
            del f[grp_name]
        grp = f.create_group(grp_name)
        grp.attrs["step"]   = step
        grp.attrs["time_s"] = t
        grp.attrs["Nx"]     = Nx
        grp.attrs["Ny"]     = Ny
        grp.attrs["Lx"]     = Lx
        grp.attrs["Ly"]     = Ly
        grp.attrs["k_sat"]  = k_sat
        grp.attrs["H_u"]    = H_u
        grp.attrs["H_d"]    = H_d

        grp.create_dataset("x",    data=xp)
        grp.create_dataset("y",    data=yp)
        grp.create_dataset("h",    data=h_field)
        grp.create_dataset("H",    data=h_field + yp)
        grp.create_dataset("Sr",   data=compute_Sr(h_field))
        grp.create_dataset("k",    data=compute_k(h_field))
        grp.create_dataset("dhdt", data=dhdt)
        grp.create_dataset("qx",   data=qx)
        grp.create_dataset("qy",   data=qy)
        grp.create_dataset("ptype", data=ptype)

# Remove stale file if present
if HAS_H5 and os.path.exists(hdf5_path):
    os.remove(hdf5_path)


# ======================================================================
# 13.  MAIN SIMULATION LOOP
# ======================================================================
#
# Strategy:
#   * Integrate with forward Euler (consistent with paper's Leap-Frog
#     for first-order time accuracy) until L2(dh/dt) plateaus.
#   * L2(dh/dt) should decrease during the transient adjustment from the
#     approximate initial condition toward the true steady state, then
#     plateau at a small residual.  That plateau indicates the true
#     steady state has been reached.
#   * Save HDF5 snapshots at regular intervals for diagnostics.

N_steps_max    = 500000        # safety cap
snapshot_every = 1000          # HDF5 snapshot interval
print_every    = 500           # console print interval
ss_tol         = 1e-14         # steady-state tolerance on L2(dh/dt)

time_log = []
l2_log   = []
l2_max_log = []
t_phys   = 0.0

print(f"\n{'='*70}")
print(f"  STARTING TIME INTEGRATION   (max {N_steps_max} steps, tol = {ss_tol:.1e})")
print(f"{'='*70}")
t_wall0 = wall_time.time()

# Mask for convergence metric: interior + impermeable boundaries
# (excludes Dirichlet boundaries where dhdt=0 trivially)
mask_conv = (ptype == 0) | (ptype == 3) | (ptype == 4)

for step in range(1, N_steps_max + 1):
    dt = stable_dt(h_w)

    # RHS
    dhdt = compute_dhdt(h_w)

    # Update
    h_w += dt * dhdt

    # Re-impose BCs
    for i in range(N_part):
        if ptype[i] == 1:  h_w[i] = H_u - yp[i]
        if ptype[i] == 2:  h_w[i] = H_d - yp[i]
    h_w = enforce_impermeable_bc(h_w)

    t_phys += dt

    # Convergence metrics
    l2     = np.sqrt(np.mean(dhdt[mask_conv]**2))
    l2_max = np.max(np.abs(dhdt[mask_conv]))
    time_log.append(t_phys)
    l2_log.append(l2)
    l2_max_log.append(l2_max)

    # Print
    if step % print_every == 0 or step == 1:
        print(f"  step {step:5d}  t = {t_phys:.4e} s  dt = {dt:.4e} s  "
              f"L2(dh/dt) = {l2:.6e}  Linf = {l2_max:.6e}")

    # HDF5 snapshot
    if step % snapshot_every == 0 or step == 1:
        qx_s, qy_s = compute_darcy_velocity(h_w)
        save_snapshot(hdf5_path, step, t_phys, h_w, dhdt, qx_s, qy_s)

    # Early exit on convergence
    if l2 < ss_tol and step > 10:
        print(f"\n  *** Converged at step {step}:  L2 = {l2:.4e}  < {ss_tol:.1e}")
        break

t_wall = wall_time.time() - t_wall0
print(f"\n  Wall time: {t_wall:.1f} s    Physical time: {t_phys:.4e} s")

# Final snapshot
print("Computing final Darcy velocity ...", flush=True)
qx_final, qy_final = compute_darcy_velocity(h_w)
save_snapshot(hdf5_path, step, t_phys, h_w, dhdt, qx_final, qy_final)
print("  done.")


# ======================================================================
# 14.  POST-PROCESSING  &  FIGURES
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X2d = xp.reshape(Nx, Ny)
Y2d = yp.reshape(Nx, Ny)

h_i2d  = h_init.reshape(Nx, Ny)
h_f2d  = h_w.reshape(Nx, Ny)
H_f2d  = (h_w + yp).reshape(Nx, Ny)
Sr_2d  = compute_Sr(h_w).reshape(Nx, Ny)
k_2d   = compute_k(h_w).reshape(Nx, Ny)
qx2d   = qx_final.reshape(Nx, Ny)
qy2d   = qy_final.reshape(Nx, Ny)
qm2d   = np.sqrt(qx2d**2 + qy2d**2)

# -- Fig 1: initial vs final pressure head --
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
levels_h = np.linspace(min(h_init.min(), h_w.min()),
                       max(h_init.max(), h_w.max()), 30)
im0 = axes[0].contourf(X2d, Y2d, h_i2d, levels=levels_h, cmap="RdYlBu")
fig.colorbar(im0, ax=axes[0], shrink=0.85, label="h [m]")
axes[0].contour(X2d, Y2d, h_i2d, levels=[0.0], colors="k", linewidths=1.5)
axes[0].set_title("Initial  h  [m]")
axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]"); axes[0].set_aspect("equal")

im1 = axes[1].contourf(X2d, Y2d, h_f2d, levels=levels_h, cmap="RdYlBu")
fig.colorbar(im1, ax=axes[1], shrink=0.85, label="h [m]")
axes[1].contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=1.5)
axes[1].set_title(f"Final  h  [m]   (t = {t_phys:.2e} s)")
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]"); axes[1].set_aspect("equal")

fig.suptitle("SPH Seepage - Pressure Head Field", fontsize=14, y=1.02)
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_pressure_head.png"),
            dpi=150, bbox_inches="tight")
print("  Saved fig1_pressure_head.png")

# -- Fig 2: hydraulic head + velocity quiver --
fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf = ax2.contourf(X2d, Y2d, H_f2d, levels=30, cmap="coolwarm")
fig2.colorbar(cf, ax=ax2, shrink=0.85, label="H [m]")

skip = max(1, Nx // 15)
q_max = np.max(qm2d)
q_scale = q_max * 15 if q_max > 0 else 1.0
ax2.quiver(X2d[::skip, ::skip], Y2d[::skip, ::skip],
           qx2d[::skip, ::skip], qy2d[::skip, ::skip],
           color="k", scale=q_scale, width=0.003)
ax2.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=1.5,
            linestyles="--")
ax2.set_title("Hydraulic Head H [m]  &  Darcy Velocity q", fontsize=13)
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.set_aspect("equal")
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_hydraulic_head_velocity.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig2_hydraulic_head_velocity.png")

# -- Fig 3: saturation --
fig3, ax3 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf3 = ax3.contourf(X2d, Y2d, Sr_2d, levels=np.linspace(0, 1, 21), cmap="Blues")
fig3.colorbar(cf3, ax=ax3, shrink=0.85, label=r"$S_r$")
ax3.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=1.5,
            linestyles="--")
ax3.set_title("Degree of Saturation  $S_r$", fontsize=13)
ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_aspect("equal")
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_saturation.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig3_saturation.png")

# -- Fig 4: velocity magnitude field --
fig4, ax4 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf4 = ax4.contourf(X2d, Y2d, qm2d, levels=30, cmap="viridis")
fig4.colorbar(cf4, ax=ax4, shrink=0.85, label="|q| [m/s]")
ax4.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="w", linewidths=1.5,
            linestyles="--")
skip2 = max(1, Nx // 12)
ax4.quiver(X2d[::skip2, ::skip2], Y2d[::skip2, ::skip2],
           qx2d[::skip2, ::skip2], qy2d[::skip2, ::skip2],
           color="w", scale=q_scale, width=0.003, alpha=0.7)
ax4.set_title("Darcy Velocity Magnitude |q| [m/s]", fontsize=13)
ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]"); ax4.set_aspect("equal")
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_velocity_magnitude.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig4_velocity_magnitude.png")

# -- Fig 5: h and |q| profiles along horizontal lines --
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
for jrow, label in [(Ny//4, f"y={Y2d[0,Ny//4]:.1f} m (below WT)"),
                     (Ny//2, f"y={Y2d[0,Ny//2]:.1f} m (near WT)"),
                     (3*Ny//4, f"y={Y2d[0,3*Ny//4]:.1f} m (above WT)")]:
    ax5a.plot(X2d[:, jrow], h_f2d[:, jrow], label=label)
    ax5b.plot(X2d[:, jrow], qm2d[:, jrow], label=label)

ax5a.set_xlabel("x [m]"); ax5a.set_ylabel("h [m]")
ax5a.set_title("Pressure-head profiles"); ax5a.legend(); ax5a.grid(True, alpha=0.3)
ax5b.set_xlabel("x [m]"); ax5b.set_ylabel("|q| [m/s]")
ax5b.set_title("Velocity-magnitude profiles"); ax5b.legend(); ax5b.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_profiles.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig5_profiles.png")

# -- Fig 6: velocity components along mid-height --
j_mid = Ny // 2
fig6a, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))
ax6a.plot(X2d[:, j_mid], qx2d[:, j_mid], 'b-', label=r"$q_x$")
ax6a.plot(X2d[:, j_mid], qy2d[:, j_mid], 'r--', label=r"$q_y$")
ax6a.axhline(0, color='k', lw=0.5)
ax6a.set_xlabel("x [m]"); ax6a.set_ylabel("q [m/s]")
ax6a.set_title(f"Velocity components at y = {Y2d[0,j_mid]:.1f} m")
ax6a.legend(); ax6a.grid(True, alpha=0.3)

ax6b.plot(X2d[:, j_mid], qm2d[:, j_mid], 'k-')
ax6b.set_xlabel("x [m]"); ax6b.set_ylabel("|q| [m/s]")
ax6b.set_title(f"|q| at y = {Y2d[0,j_mid]:.1f} m"); ax6b.grid(True, alpha=0.3)

# Show expected uniform |q| for reference
q_expected = k_sat * (H_u - H_d) / Lx
ax6b.axhline(q_expected, color='r', ls='--', lw=1, label=f"$k_{{sat}} \\Delta H / L$ = {q_expected:.4e}")
ax6b.legend()
fig6a.tight_layout()
fig6a.savefig(os.path.join(OUTPUT_DIR, "fig6_velocity_components.png"),
              dpi=150, bbox_inches="tight")
print("  Saved fig6_velocity_components.png")

# -- Fig 7: convergence history --
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
ax7a.semilogy(time_log, l2_log, "b-", lw=1, label="L2(dh/dt)")
ax7a.semilogy(time_log, l2_max_log, "r--", lw=1, label="Linf(dh/dt)")
ax7a.set_xlabel("Time [s]"); ax7a.set_ylabel("Residual")
ax7a.set_title("Convergence History"); ax7a.legend(); ax7a.grid(True, which="both", alpha=0.3)

ax7b.semilogy(range(1, len(l2_log)+1), l2_log, "b-", lw=1, label="L2")
ax7b.semilogy(range(1, len(l2_max_log)+1), l2_max_log, "r--", lw=1, label="Linf")
ax7b.set_xlabel("Step"); ax7b.set_ylabel("Residual")
ax7b.set_title("Convergence vs Step"); ax7b.legend(); ax7b.grid(True, which="both", alpha=0.3)
fig7.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_convergence.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig7_convergence.png")

# -- Summary --
dh_max = np.max(np.abs(h_w - h_init))
q_expected = k_sat * (H_u - H_d) / Lx

# Velocity uniformity check in the saturated zone
sat_mask = (Sr_2d > 0.99).flatten()
int_mask = (ptype == 0)
sat_int  = sat_mask & int_mask
if np.any(sat_int):
    q_sat = qm2d.flatten()[sat_int]
    q_sat_mean = np.mean(q_sat)
    q_sat_std  = np.std(q_sat)
    q_sat_cv   = q_sat_std / q_sat_mean if q_sat_mean > 0 else 0
else:
    q_sat_mean = 0; q_sat_std = 0; q_sat_cv = 0

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Max |h_final - h_initial|     = {dh_max:.6e} m")
print(f"  Final L2(dh/dt)               = {l2_log[-1]:.6e}")
print(f"  Final Linf(dh/dt)             = {l2_max_log[-1]:.6e}")
print(f"  Max Darcy |q|                 = {np.max(qm2d):.6e} m/s")
print(f"  Saturated zone |q|: mean      = {q_sat_mean:.6e} m/s")
print(f"                      std       = {q_sat_std:.6e} m/s")
print(f"                      CV        = {q_sat_cv:.4f}")
print(f"  k_sat                         = {k_sat:.6e} m/s")
print(f"  Expected q ~ k_sat*dH/Lx     = {q_expected:.6e} m/s")
if HAS_H5:
    print(f"  HDF5 snapshots                = {hdf5_path}")
print(f"{'='*70}")

plt.close("all")
print("\nDone.")
