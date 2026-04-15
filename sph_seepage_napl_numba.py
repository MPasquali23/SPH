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
from scipy.spatial import cKDTree


try:
    import numba
    from numba import njit
    HAS_NUMBA = True
    print("Numba available — JIT-accelerated particle loops enabled.")
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not available — falling back to pure Python loops.")

# ======================================================================
# 0.  OUTPUT DIRECTORY
# ======================================================================
OUTPUT_DIR = "../data_sph"
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
# 2.  DOMAIN  &  SPH DISCRETISATION
# ======================================================================
# debugging using different grid sizes
import sys
if "--lx" in sys.argv:
    idx = sys.argv.index("--lx")
    Lx = Ly = float(sys.argv[idx + 1])
if "--nx" in sys.argv:
    idx = sys.argv.index("--nx")
    Nx = Ny = int(sys.argv[idx + 1])
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
else:
    Lx = 10.0
    Ly = 10.0
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

print("Building neighbour lists (cKDTree) ...", flush=True)

tree = cKDTree(np.column_stack([xp, yp]))
nbl  = tree.query_ball_tree(tree, r=support_r)      # list of lists

# Each entry: (j, r, x_ji, y_ji, F_hat, gradWx, gradWy)
neighbours = [[] for _ in range(N_part)]

for i in range(N_part):
    xi, yi = xp[i], yp[i]
    for j in nbl[i]:
        if j == i:
            continue
        xji = xp[j] - xi
        yji = yp[j] - yi
        r   = np.sqrt(xji**2 + yji**2)
        if r < 1e-30:
            continue

        dWr = dW_dr(r)
        gWx = dWr * (-xji) / r
        gWy = dWr * (-yji) / r

        dot  = xji * gWx + yji * gWy
        Fhat = dot / (r * r)

        neighbours[i].append((j, r, xji, yji, Fhat, gWx, gWy))

print("  done.  Converting to CSR arrays ...", flush=True)

# --- Convert neighbour lists to CSR arrays for Numba ---
_total_nbrs = sum(len(nb) for nb in neighbours)
nbr_j    = np.empty(_total_nbrs, dtype=np.int64)
nbr_Fhat = np.empty(_total_nbrs, dtype=np.float64)
nbr_gWx  = np.empty(_total_nbrs, dtype=np.float64)
nbr_gWy  = np.empty(_total_nbrs, dtype=np.float64)
nbr_ptr  = np.empty(N_part + 1, dtype=np.int64)

_offset = 0
for i in range(N_part):
    nbr_ptr[i] = _offset
    for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
        nbr_j[_offset]    = j
        nbr_Fhat[_offset] = Fhat
        nbr_gWx[_offset]  = gWx
        nbr_gWy[_offset]  = gWy
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

    return (gamma_w / gamma_n) * (h_w + h_cnw) + yp


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

# -- NAPL source region --
# 1 m x 1 m square centred at (5, 9) -- in the unsaturated zone
# (water table at x=5 is at z_wt ~ 7 m, so source is well above).
# Represents a leaking underground storage tank.
SRC_X0, SRC_X1 = 4.5, 5.5
SRC_Y0, SRC_Y1 = 8.5, 9.5
SN_SOURCE       = 0.80       # fixed NAPL saturation at source

is_source = np.zeros(N_part, dtype=bool)
for i in range(N_part):
    if SRC_X0 <= xp[i] <= SRC_X1 and SRC_Y0 <= yp[i] <= SRC_Y1:
        is_source[i] = True

n_src = np.sum(is_source)
print(f"NAPL source particles: {n_src}  "
      f"([{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}],  Sn = {SN_SOURCE})")

# NAPL saturation field
S_n = np.zeros(N_part)
S_n[is_source] = SN_SOURCE

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


def enforce_napl_bc(Sn):
    """NAPL BCs: Sn=0 at Dirichlet walls, mirror at impermeable walls,
    maintain source."""
    for i in range(N_part):
        if ptype[i] == 1 or ptype[i] == 2:     # left/right: clean water
            Sn[i] = 0.0
        elif ptype[i] == 3:                     # bottom: mirror
            ix = min(int(round(xp[i] / dx)), Nx - 1)
            Sn[i] = Sn[idx_2d[ix, 1]]
        elif ptype[i] == 4:                     # top: mirror
            ix = min(int(round(xp[i] / dx)), Nx - 1)
            Sn[i] = Sn[idx_2d[ix, Ny - 2]]
    Sn[is_source] = SN_SOURCE
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
    @njit(cache=True)
    def _sph_div_k_gradH(H_f, k_h, C_store, ptype_arr, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                          L_inv, K_norm, err_x, err_y, Vp_val, N):
        """Corrected Laplacian: div[k grad(H)] / C_store  for each particle.

        skip_mask[i] = True means particle i is skipped (dhdt = 0).
        """
        out = np.zeros(N)
        for i in range(N):
            if skip_mask[i]:
                continue

            Hi = H_f[i]
            ki = k_h[i]

            # -- corrected gradient of H --
            raw_gx = 0.0
            raw_gy = 0.0
            for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                j = nbr_j[k]
                dH = H_f[j] - Hi
                raw_gx += Vp_val * dH * nbr_gWx[k]
                raw_gy += Vp_val * dH * nbr_gWy[k]

            L00 = L_inv[i, 0, 0]; L01 = L_inv[i, 0, 1]
            L10 = L_inv[i, 1, 0]; L11 = L_inv[i, 1, 1]
            grad_Hx = L00 * raw_gx + L01 * raw_gy
            grad_Hy = L10 * raw_gx + L11 * raw_gy

            # -- variable-k Laplacian sum --
            lap_sum = 0.0
            for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                j = nbr_j[k]
                km = 0.5 * (ki + k_h[j])
                lap_sum += Vp_val * km * (H_f[j] - Hi) * nbr_Fhat[k]

            # -- error correction --
            corr = ki * (grad_Hx * err_x[i] + grad_Hy * err_y[i])

            Kn = K_norm[i]
            if abs(Kn) < 1e-30:
                continue

            out[i] = (2.0 / Kn) * (lap_sum - corr) / C_store[i]

        return out

    @njit(cache=True)
    def _sph_kvar_gradient(H_f, k_h,
                            nbr_ptr, nbr_j, nbr_gWx, nbr_gWy,
                            L_inv, Vp_val, N):
        """Corrected gradient with variable-k weighting:
           sum_j V_j k_mean (H_j - H_i) * L^{-1} grad_W
        Returns (qx, qy) = the SPH approximation of k*grad(H).
        Caller negates for Darcy velocity.
        """
        qx = np.zeros(N)
        qy = np.zeros(N)
        for i in range(N):
            L00 = L_inv[i, 0, 0]; L01 = L_inv[i, 0, 1]
            L10 = L_inv[i, 1, 0]; L11 = L_inv[i, 1, 1]
            ki = k_h[i]
            Hi = H_f[i]
            sx = 0.0
            sy = 0.0
            for k in range(nbr_ptr[i], nbr_ptr[i+1]):
                j = nbr_j[k]
                km  = 0.5 * (ki + k_h[j])
                dH  = H_f[j] - Hi
                gx  = nbr_gWx[k]
                gy  = nbr_gWy[k]
                cWx = L00 * gx + L01 * gy
                cWy = L10 * gx + L11 * gy
                val = Vp_val * km * dH
                sx += val * cWx
                sy += val * cWy
            qx[i] = sx
            qy[i] = sy
        return qx, qy

# ---------- Fallback pure-Python implementations (same logic) ----------

def _sph_div_k_gradH_py(H_f, k_h, C_store, ptype_arr, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                          L_inv, K_norm, err_x, err_y, Vp_val, N):
    out = np.zeros(N)
    for i in range(N):
        if skip_mask[i]:
            continue
        Hi = H_f[i]; ki = k_h[i]
        raw_gx = 0.0; raw_gy = 0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_j[k]; dH = H_f[j] - Hi
            raw_gx += Vp_val * dH * nbr_gWx[k]
            raw_gy += Vp_val * dH * nbr_gWy[k]
        L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L10=L_inv[i,1,0]; L11=L_inv[i,1,1]
        grad_Hx = L00*raw_gx + L01*raw_gy
        grad_Hy = L10*raw_gx + L11*raw_gy
        lap_sum = 0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_j[k]; km = 0.5*(ki+k_h[j])
            lap_sum += Vp_val * km * (H_f[j]-Hi) * nbr_Fhat[k]
        corr = ki * (grad_Hx*err_x[i] + grad_Hy*err_y[i])
        Kn = K_norm[i]
        if abs(Kn) < 1e-30: continue
        out[i] = (2.0/Kn)*(lap_sum-corr)/C_store[i]
    return out

def _sph_kvar_gradient_py(H_f, k_h, nbr_ptr, nbr_j, nbr_gWx, nbr_gWy,
                           L_inv, Vp_val, N):
    qx = np.zeros(N); qy = np.zeros(N)
    for i in range(N):
        L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L10=L_inv[i,1,0]; L11=L_inv[i,1,1]
        ki=k_h[i]; Hi=H_f[i]; sx=0.0; sy=0.0
        for k in range(nbr_ptr[i], nbr_ptr[i+1]):
            j=nbr_j[k]; km=0.5*(ki+k_h[j]); dH=H_f[j]-Hi
            gx=nbr_gWx[k]; gy=nbr_gWy[k]
            cWx=L00*gx+L01*gy; cWy=L10*gx+L11*gy
            val=Vp_val*km*dH; sx+=val*cWx; sy+=val*cWy
        qx[i]=sx; qy[i]=sy
    return qx, qy

# Choose implementation
_div_k_gradH  = _sph_div_k_gradH  if HAS_NUMBA else _sph_div_k_gradH_py
_kvar_gradient = _sph_kvar_gradient if HAS_NUMBA else _sph_kvar_gradient_py

# ---------- Wrapper functions (same API as before) ----------

# Precompute skip masks (boolean arrays)
_skip_dirichlet = (ptype == 1) | (ptype == 2)
_skip_napl      = _skip_dirichlet | is_source

def compute_dhdt(h_field, Sn_field):
    """RHS of the water seepage equation (three-phase k_w)."""
    k_h     = compute_kw_3ph(h_field, Sn_field)
    Cs      = compute_Cs(h_field)
    H_f     = h_field + yp
    C_tilde = np.maximum(C_l + Cs, C_l)

    return _div_k_gradH(H_f, k_h, C_tilde, ptype, _skip_dirichlet,
                         nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                         L_inv, K_norm, err_x, err_y, Vp, N_part)


def compute_dSndt(h_field, Sn_field):
    """NAPL transport equation RHS."""
    k_n     = compute_kn_field(h_field, Sn_field)
    H_n     = compute_Hn_field(h_field, Sn_field)
    C_store = np.full(N_part, phi_0)

    return _div_k_gradH(H_n, k_n, C_store, ptype, _skip_napl,
                         nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                         L_inv, K_norm, err_x, err_y, Vp, N_part)


def compute_darcy_velocity(h_field, Sn_field):
    """Water Darcy velocity  q_w = -k_w grad(H_w)."""
    k_h = compute_kw_3ph(h_field, Sn_field)
    H_f = h_field + yp
    qx, qy = _kvar_gradient(H_f, k_h, nbr_ptr, nbr_j,
                              nbr_gWx, nbr_gWy, L_inv, Vp, N_part)
    return -qx, -qy


def compute_darcy_velocity_napl(h_field, Sn_field):
    """NAPL Darcy velocity  q_n = -k_n grad(H_n)."""
    k_n = compute_kn_field(h_field, Sn_field)
    H_n = compute_Hn_field(h_field, Sn_field)
    qx, qy = _kvar_gradient(H_n, k_n, nbr_ptr, nbr_j,
                              nbr_gWx, nbr_gWy, L_inv, Vp, N_part)
    return -qx, -qy


# ======================================================================
# 11.  TIME STEP   [Paper Eq. 66 / Appendix A.17]
# ======================================================================
CFL = 0.1

def stable_dt(h_field, Sn_field):
    Cs = compute_Cs(h_field)
    C_tilde = C_l + Cs
    C_min = max(np.min(C_tilde), C_l)
    dt_w = CFL * C_min * h_sml**2 / k_sat
    # NAPL CFL
    dt_n = CFL * phi_0 * h_sml**2 / k_sat_n
    return min(dt_w, dt_n)


# ======================================================================
# 12.  HDF5 SNAPSHOT I/O
# ======================================================================
try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False
    print("WARNING: h5py not available - HDF5 snapshots disabled.")

hdf5_path = os.path.join(OUTPUT_DIR, "sph_napl_snapshots.h5")

SNAP_DIR = os.path.join(OUTPUT_DIR, "snapshots")

def save_snapshot(fname, step, t, h_field, Sn_field, dhdt, dSndt, qx, qy):
    """Save snapshot — HDF5 if available, else individual NPZ files."""
    Sw = compute_Sw_3ph(h_field, Sn_field)

    if HAS_H5:
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
            grp.create_dataset("Sn",   data=Sn_field)
            grp.create_dataset("Sw",   data=Sw)
            grp.create_dataset("kw",   data=compute_kw_3ph(h_field, Sn_field))
            grp.create_dataset("kn",   data=compute_kn_field(h_field, Sn_field))
            grp.create_dataset("dhdt", data=dhdt)
            grp.create_dataset("dSndt", data=dSndt)
            grp.create_dataset("qx",   data=qx)
            grp.create_dataset("qy",   data=qy)
            grp.create_dataset("ptype", data=ptype)
            grp.create_dataset("is_source", data=is_source.astype(np.int8))
    else:
        # NPZ fallback: one file per snapshot in snapshots/ directory
        os.makedirs(SNAP_DIR, exist_ok=True)
        path = os.path.join(SNAP_DIR, f"step_{step:06d}.npz")
        np.savez_compressed(path,
            step=step, time_s=t, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
            k_sat=k_sat, H_u=H_u, H_d=H_d,
            x=xp, y=yp, h=h_field, H=h_field + yp,
            Sn=Sn_field, Sw=Sw,
            kw=compute_kw_3ph(h_field, Sn_field),
            kn=compute_kn_field(h_field, Sn_field),
            dhdt=dhdt, dSndt=dSndt, qx=qx, qy=qy,
            ptype=ptype, is_source=is_source.astype(np.int8))

# NOTE: snapshot file is NOT deleted on restart — new groups are appended
# so the full time history is preserved across SLURM job submissions.


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

def save_checkpoint(step, t, hw, Sn, time_log, l2_log, l2n_log,
                    Sn_max_log, napl_mass_log, napl_mass_nosrc_log):
    """Save full simulation state to an individual HDF5 checkpoint file."""
    if not HAS_H5:
        # Fallback: NPZ if h5py unavailable
        path = os.path.join(CKPT_DIR, f"ckpt_{step:08d}.npz")
        np.savez_compressed(path, step=step, t_phys=t, h_w=hw, S_n=Sn,
            time_log=np.array(time_log), l2_log=np.array(l2_log),
            l2n_log=np.array(l2n_log), Sn_max_log=np.array(Sn_max_log),
            napl_mass_log=np.array(napl_mass_log),
            napl_mass_nosrc_log=np.array(napl_mass_nosrc_log))
        print(f"    checkpoint saved (NPZ fallback): {path}")
        return

    path = os.path.join(CKPT_DIR, f"ckpt_{step:08d}.h5")
    with h5py.File(path, "w") as f:
        # Scalar metadata as attributes on root
        f.attrs["step"]    = step
        f.attrs["t_phys"]  = t
        f.attrs["Nx"]      = Nx
        f.attrs["Ny"]      = Ny
        f.attrs["Lx"]      = Lx
        f.attrs["Ly"]      = Ly
        f.attrs["k_sat"]   = k_sat
        f.attrs["k_sat_n"] = k_sat_n
        f.attrs["H_u"]     = H_u
        f.attrs["H_d"]     = H_d

        # Field arrays (compressed)
        ckw = dict(compression="gzip", compression_opts=4)
        f.create_dataset("h_w", data=hw, **ckw)
        f.create_dataset("S_n", data=Sn, **ckw)

        # Convergence history
        f.create_dataset("time_log",            data=np.array(time_log))
        f.create_dataset("l2_log",              data=np.array(l2_log))
        f.create_dataset("l2n_log",             data=np.array(l2n_log))
        f.create_dataset("Sn_max_log",          data=np.array(Sn_max_log))
        f.create_dataset("napl_mass_log",       data=np.array(napl_mass_log))
        f.create_dataset("napl_mass_nosrc_log", data=np.array(napl_mass_nosrc_log))

    print(f"    checkpoint saved: {path}")


def load_latest_checkpoint():
    """Scan checkpoint directory for the most recent file.
    Tries HDF5 first, then NPZ fallback.
    Returns dict or None.
    """
    import glob

    # Scan for HDF5 checkpoints
    h5_files = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.h5")))
    # Scan for NPZ checkpoints
    npz_files = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.npz")))

    # Pick the latest across both formats
    latest_h5  = h5_files[-1]  if h5_files  else None
    latest_npz = npz_files[-1] if npz_files else None

    # Extract step numbers to compare
    def step_from_path(p):
        base = os.path.basename(p)       # ckpt_00001000.h5
        num  = base.split("_")[1].split(".")[0]  # 00001000
        return int(num)

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
        print(f"  Restarting from NPZ checkpoint: {path}")
        print(f"    step = {d['step']},  t = {d['t_phys']:.4e} s")
        return d

    return None


# ======================================================================
# 13.  MAIN SIMULATION LOOP
# ======================================================================

N_steps_max    = 5000          # adjust as needed
snapshot_every = 10
print_every    = 10
ckpt_every     = 10000
ss_tol         = 1e-14

# --- Attempt restart from checkpoint ---
ckpt = load_latest_checkpoint()
if ckpt is not None:
    start_step = int(ckpt["step"]) + 1
    t_phys     = float(ckpt["t_phys"])
    h_w[:]     = ckpt["h_w"]
    S_n[:]     = ckpt["S_n"]
    time_log       = list(ckpt["time_log"])
    l2_log         = list(ckpt["l2_log"])
    l2n_log        = list(ckpt["l2n_log"])
    Sn_max_log     = list(ckpt["Sn_max_log"])
    napl_mass_log      = list(ckpt["napl_mass_log"])
    napl_mass_nosrc_log = list(ckpt["napl_mass_nosrc_log"])

    # --- Purge stale data from future of the checkpoint ---
    # If a previous run wrote snapshots/checkpoints past the checkpoint
    # step before being killed, those are from a trajectory that will be
    # overwritten.  Remove them now to avoid an inconsistent history.
    ckpt_step = int(ckpt["step"])

    # Purge stale HDF5 snapshot groups
    if HAS_H5 and os.path.exists(hdf5_path):
        n_purged = 0
        with h5py.File(hdf5_path, "a") as f:
            stale = [k for k in f.keys()
                     if k.startswith("step_")
                     and int(k.split("_")[1]) > ckpt_step]
            for k in stale:
                del f[k]
                n_purged += 1
        if n_purged > 0:
            print(f"  Purged {n_purged} stale snapshot(s) "
                  f"(step > {ckpt_step}) from {hdf5_path}")
            # Repack to reclaim dead space left by deleted groups
            import shutil, subprocess
            tmp_path = hdf5_path + ".repack"
            try:
                result = subprocess.run(
                    ["h5repack", hdf5_path, tmp_path],
                    capture_output=True, timeout=120)
                if result.returncode == 0:
                    shutil.move(tmp_path, hdf5_path)
                    print(f"  Repacked {hdf5_path} (dead space reclaimed)")
                else:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Purge stale checkpoint files past the restart point
    import glob
    for pat in ["ckpt_*.h5", "ckpt_*.npz"]:
        for cp_path in glob.glob(os.path.join(CKPT_DIR, pat)):
            base = os.path.basename(cp_path)
            cp_step = int(base.split("_")[1].split(".")[0])
            if cp_step > ckpt_step:
                os.remove(cp_path)
                print(f"  Removed stale checkpoint: {base}")

else:
    start_step = 1
    t_phys     = 0.0
    time_log       = []
    l2_log         = []
    l2n_log        = []
    Sn_max_log     = []
    napl_mass_log      = []
    napl_mass_nosrc_log = []

# History of Sn snapshots (for animation / evolution panel)
Sn_history = []       # list of (step, t, Sn_copy)
Sn_snap_every = max(1, N_steps_max // 40)   # ~40 frames for animation

print(f"\n{'='*70}")
print(f"  THREE-PHASE SPH: Water + LNAPL + Air")
print(f"  Steps [{start_step}, {N_steps_max}],  NAPL source at "
      f"[{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}]")
print(f"{'='*70}")
t_wall0 = wall_time.time()

mask_conv = (ptype == 0) | (ptype == 3) | (ptype == 4)
mask_not_src = ~is_source & mask_conv

# Save initial Sn snapshot
if start_step == 1:
    Sn_history.append((0, 0.0, S_n.copy()))

step = start_step - 1   # default if loop is skipped (restart at end)
for step in range(start_step, N_steps_max + 1):
    dt = stable_dt(h_w, S_n)

    dhdt  = compute_dhdt(h_w, S_n)
    dSndt = compute_dSndt(h_w, S_n)

    h_w += dt * dhdt
    S_n += dt * dSndt

    S_n = np.clip(S_n, 0.0, 1.0 - S_res)

    for i in range(N_part):
        if ptype[i] == 1:  h_w[i] = H_u - yp[i]
        if ptype[i] == 2:  h_w[i] = H_d - yp[i]
    h_w = enforce_impermeable_bc(h_w)
    S_n = enforce_napl_bc(S_n)

    t_phys += dt

    # Convergence metrics
    l2     = np.sqrt(np.mean(dhdt[mask_conv]**2))
    l2n    = np.sqrt(np.mean(dSndt[mask_not_src]**2)) if np.any(mask_not_src) else 0.0
    sn_max = np.max(S_n[~is_source]) if np.any(~is_source) else 0.0

    # NAPL mass tracking
    napl_mass_total  = np.sum(S_n * Vp * phi_0)
    napl_mass_nosrc  = np.sum(S_n[~is_source] * Vp * phi_0)

    time_log.append(t_phys)
    l2_log.append(l2)
    l2n_log.append(l2n)
    Sn_max_log.append(sn_max)
    napl_mass_log.append(napl_mass_total)
    napl_mass_nosrc_log.append(napl_mass_nosrc)

    # Print
    if step % print_every == 0 or step == start_step:
        print(f"  step {step:5d}  t = {t_phys:.4e} s  dt = {dt:.4e} s  "
              f"L2w = {l2:.4e}  L2n = {l2n:.4e}  Sn_max = {sn_max:.4f}  "
              f"NAPL_out = {napl_mass_nosrc:.4e}")

    # HDF5 snapshot
    if step % snapshot_every == 0 or step == start_step:
        qx_s, qy_s = compute_darcy_velocity(h_w, S_n)
        save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt, dSndt, qx_s, qy_s)

    # Sn snapshot for animation
    if step % Sn_snap_every == 0 or step == start_step:
        Sn_history.append((step, t_phys, S_n.copy()))

    # Checkpoint
    if step % ckpt_every == 0:
        save_checkpoint(step, t_phys, h_w, S_n,
                        time_log, l2_log, l2n_log,
                        Sn_max_log, napl_mass_log, napl_mass_nosrc_log)

    if l2 < ss_tol and step > 10:
        print(f"\n  *** Converged at step {step}:  L2 = {l2:.4e}  < {ss_tol:.1e}")
        break

t_wall = wall_time.time() - t_wall0
print(f"\n  Wall time: {t_wall:.1f} s    Physical time: {t_phys:.4e} s")

ran_steps = (step >= start_step)

if ran_steps:
    # Final Sn snapshot
    Sn_history.append((step, t_phys, S_n.copy()))

    # Final checkpoint
    save_checkpoint(step, t_phys, h_w, S_n,
                    time_log, l2_log, l2n_log,
                    Sn_max_log, napl_mass_log, napl_mass_nosrc_log)

# Final velocities
print("Computing final Darcy velocity ...", flush=True)
qx_final, qy_final = compute_darcy_velocity(h_w, S_n)
dhdt_f = compute_dhdt(h_w, S_n)
dSndt_f = compute_dSndt(h_w, S_n)
save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt_f, dSndt_f, qx_final, qy_final)
print("  done.")


# ======================================================================
# 14.  POST-PROCESSING  &  FIGURES
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.animation as animation

X2d = xp.reshape(Nx, Ny)
Y2d = yp.reshape(Nx, Ny)

h_i2d  = h_init.reshape(Nx, Ny)
h_f2d  = h_w.reshape(Nx, Ny)
H_f2d  = (h_w + yp).reshape(Nx, Ny)
Sn_2d  = S_n.reshape(Nx, Ny)
Sw_2d  = compute_Sw_3ph(h_w, S_n).reshape(Nx, Ny)
Sa_2d  = np.clip(1.0 - Sw_2d - Sn_2d, 0, 1)
qx2d   = qx_final.reshape(Nx, Ny)
qy2d   = qy_final.reshape(Nx, Ny)
qm2d   = np.sqrt(qx2d**2 + qy2d**2)

qxn_f, qyn_f = compute_darcy_velocity_napl(h_w, S_n)
qxn2d  = qxn_f.reshape(Nx, Ny)
qyn2d  = qyn_f.reshape(Nx, Ny)
qmn2d  = np.sqrt(qxn2d**2 + qyn2d**2)

src_rect_kw = dict(xy=(SRC_X0, SRC_Y0), width=SRC_X1-SRC_X0,
                   height=SRC_Y1-SRC_Y0, lw=2, ec='red', fc='none', ls='--')


# ── Fig 1: NAPL saturation + NAPL streamlines ──────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 8), constrained_layout=True)
levels_sn = np.linspace(0, SN_SOURCE, 21)
cf1 = ax1.contourf(X2d, Y2d, Sn_2d, levels=levels_sn, cmap="YlOrRd", extend="max")
fig1.colorbar(cf1, ax=ax1, shrink=0.85, label=r"$S_n$ [-]")
ax1.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="blue", linewidths=2,
            linestyles="--", zorder=5)
ax1.add_patch(Rectangle(**src_rect_kw))
# Streamlines for NAPL velocity (only where Sn > threshold)
speed_n = np.maximum(qmn2d, 1e-30)
if np.max(qmn2d) > 1e-25:
    ax1.streamplot(X2d[:, 0], Y2d[0, :], qxn2d.T, qyn2d.T,
                   color="k", linewidth=0.8, density=1.2, arrowsize=1.2)
ax1.set_title(f"NAPL Saturation $S_n$ + NAPL Streamlines  (t = {t_phys:.1f} s)",
              fontsize=13)
ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_aspect("equal")
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_napl_saturation.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig1_napl_saturation.png")

# ── Fig 2: Composite three-phase RGB map ────────────────────────────────
#   R = S_n (NAPL),  G = S_a (air),  B = S_w (water)
fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
rgb = np.zeros((Nx, Ny, 3))
rgb[:, :, 0] = np.clip(Sn_2d / max(Sn_2d.max(), 0.01), 0, 1)   # Red = NAPL
rgb[:, :, 1] = np.clip(Sa_2d, 0, 1)                               # Green = Air
rgb[:, :, 2] = np.clip(Sw_2d, 0, 1)                               # Blue = Water
# Transpose for imshow: (Ny, Nx, 3), origin lower
ax2.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
           extent=[0, Lx, 0, Ly], aspect="equal", interpolation="bilinear")
ax2.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="white", linewidths=1.5,
            linestyles="--")
ax2.add_patch(Rectangle(**{**src_rect_kw, 'ec': 'white'}))
# Legend patches
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(fc='blue', label='Water $S_w$'),
                    Patch(fc='red', label='NAPL $S_n$'),
                    Patch(fc='green', label='Air $S_a$')],
           loc='upper right', fontsize=10, framealpha=0.8)
ax2.set_title("Three-Phase Saturation (RGB Composite)", fontsize=13)
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_three_phase_rgb.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig2_three_phase_rgb.png")

# ── Fig 3: Water head + water streamlines ───────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf3 = ax3.contourf(X2d, Y2d, H_f2d, levels=30, cmap="coolwarm")
fig3.colorbar(cf3, ax=ax3, shrink=0.85, label="$H_w$ [m]")
ax3.streamplot(X2d[:, 0], Y2d[0, :], qx2d.T, qy2d.T,
               color="k", linewidth=0.8, density=1.5, arrowsize=1.2)
ax3.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=2,
            linestyles="--")
ax3.contour(X2d, Y2d, Sn_2d, levels=[0.01], colors="red", linewidths=2)
ax3.set_title("Water Hydraulic Head  $H_w$  &  Water Streamlines", fontsize=13)
ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_aspect("equal")
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_water_head_streamlines.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig3_water_head_streamlines.png")

# ── Fig 4: Vertical cross-section evolution of Sn at x = 5 m ───────────
ix_mid = Nx // 2
fig4, ax4 = plt.subplots(figsize=(8, 8), constrained_layout=True)
cmap_evo = plt.cm.inferno
n_snaps = len(Sn_history)
for idx, (s, t_s, sn_snap) in enumerate(Sn_history):
    sn_2d_snap = sn_snap.reshape(Nx, Ny)
    col = cmap_evo(idx / max(n_snaps - 1, 1))
    ax4.plot(sn_2d_snap[ix_mid, :], Y2d[ix_mid, :], color=col, lw=1.5,
             label=f"t={t_s:.1f} s" if idx % max(1, n_snaps//6) == 0
                   or idx == n_snaps - 1 else None)
# Mark water table range
zwt_left  = H_u + (H_d - H_u) * X2d[ix_mid, 0] / Lx
ax4.axhline(zwt_left, color='blue', ls='--', lw=1, alpha=0.5, label='initial WT')
ax4.set_xlabel(r"$S_n$ [-]", fontsize=12)
ax4.set_ylabel("y [m]", fontsize=12)
ax4.set_title(f"NAPL Saturation Evolution at x = {X2d[ix_mid,0]:.1f} m", fontsize=13)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.02, SN_SOURCE + 0.05)
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_Sn_evolution_vertical.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig4_Sn_evolution_vertical.png")

# ── Fig 5: NAPL mass balance ────────────────────────────────────────────
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
ax5a.plot(time_log, napl_mass_log, 'k-', lw=1.5, label="Total (incl. source)")
ax5a.plot(time_log, napl_mass_nosrc_log, 'r-', lw=1.5, label="Outside source")
ax5a.set_xlabel("Time [s]"); ax5a.set_ylabel(r"NAPL volume $\phi S_n V$  [m$^3$/m]")
ax5a.set_title("NAPL Mass Balance"); ax5a.legend(); ax5a.grid(True, alpha=0.3)

# Source discharge rate (finite difference)
if len(napl_mass_nosrc_log) > 2:
    t_arr = np.array(time_log)
    m_arr = np.array(napl_mass_nosrc_log)
    dt_arr = np.diff(t_arr)
    dm_arr = np.diff(m_arr)
    rate = dm_arr / np.maximum(dt_arr, 1e-30)
    ax5b.plot(t_arr[1:], rate, 'r-', lw=1)
    ax5b.set_xlabel("Time [s]"); ax5b.set_ylabel("d(NAPL vol)/dt  [m³/m/s]")
    ax5b.set_title("NAPL Source Discharge Rate"); ax5b.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_napl_mass_balance.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig5_napl_mass_balance.png")

# ── Fig 6: Water table perturbation ─────────────────────────────────────
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))

# Find h=0 contour for initial and final
# Approximate: interpolate y where h crosses zero at each x-column
def find_wt(h_2d, X, Y):
    """Find water table elevation (h=0 contour) per x-column."""
    nx = h_2d.shape[0]
    xwt = np.zeros(nx)
    ywt = np.zeros(nx)
    for ix in range(nx):
        col = h_2d[ix, :]
        yvals = Y[ix, :]
        # Find crossing from + to - going upward
        crossings = np.where(np.diff(np.sign(col)))[0]
        if len(crossings) > 0:
            j = crossings[0]
            # Linear interpolation
            f = col[j] / (col[j] - col[j+1]) if col[j] != col[j+1] else 0.5
            ywt[ix] = yvals[j] + f * (yvals[j+1] - yvals[j])
        else:
            ywt[ix] = yvals[0] if col[0] < 0 else yvals[-1]
        xwt[ix] = X[ix, 0]
    return xwt, ywt

xwt_i, ywt_i = find_wt(h_i2d, X2d, Y2d)
xwt_f, ywt_f = find_wt(h_f2d, X2d, Y2d)

ax6a.plot(xwt_i, ywt_i, 'b--', lw=1.5, label="Initial WT")
ax6a.plot(xwt_f, ywt_f, 'r-',  lw=2,   label="Final WT")
ax6a.fill_between(xwt_f, ywt_i, ywt_f, alpha=0.2, color='orange',
                  label="Perturbation")
ax6a.set_xlabel("x [m]"); ax6a.set_ylabel("z [m]")
ax6a.set_title("Water Table Position"); ax6a.legend(); ax6a.grid(True, alpha=0.3)

ax6b.plot(xwt_f, ywt_f - ywt_i, 'k-', lw=2)
ax6b.axhline(0, color='gray', ls=':', lw=0.8)
ax6b.set_xlabel("x [m]"); ax6b.set_ylabel("ΔWT [m]")
ax6b.set_title("Water Table Displacement (final − initial)")
ax6b.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_water_table_perturbation.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig6_water_table_perturbation.png")

# ── Fig 7: Convergence history + NAPL front ─────────────────────────────
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
ax7a.semilogy(time_log, l2_log, "b-", lw=1, label=r"L2($dh_w/dt$)")
ax7a.semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
ax7a.set_xlabel("Time [s]"); ax7a.set_ylabel("Residual")
ax7a.set_title("Convergence History"); ax7a.legend()
ax7a.grid(True, which="both", alpha=0.3)

ax7b.plot(time_log, Sn_max_log, "r-", lw=1.5)
ax7b.set_xlabel("Time [s]"); ax7b.set_ylabel(r"max $S_n$ (outside source)")
ax7b.set_title("NAPL Front Propagation"); ax7b.grid(True, alpha=0.3)
fig7.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_convergence.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig7_convergence.png")

# ── Fig 8: Three-phase saturation profiles (vertical at x=5) ───────────
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 6))
ax8a.plot(Sw_2d[ix_mid, :], Y2d[ix_mid, :], "b-", lw=2, label=r"$S_w$")
ax8a.plot(Sn_2d[ix_mid, :], Y2d[ix_mid, :], "r-", lw=2, label=r"$S_n$")
ax8a.plot(Sa_2d[ix_mid, :], Y2d[ix_mid, :], "g--", lw=1.5, label=r"$S_a$")
ax8a.set_xlabel("Saturation [-]"); ax8a.set_ylabel("y [m]")
ax8a.set_title(f"Saturation profiles at x = {X2d[ix_mid,0]:.1f} m")
ax8a.legend(); ax8a.grid(True, alpha=0.3); ax8a.set_xlim(-0.05, 1.05)

for ix_val in [Nx//5, 2*Nx//5, Nx//2, 3*Nx//5, 4*Nx//5]:
    ax8b.plot(Sn_2d[ix_val, :], Y2d[ix_val, :], label=f"x={X2d[ix_val,0]:.1f}")
ax8b.set_xlabel(r"$S_n$ [-]"); ax8b.set_ylabel("y [m]")
ax8b.set_title("NAPL saturation at various x"); ax8b.legend()
ax8b.grid(True, alpha=0.3)
fig8.tight_layout()
fig8.savefig(os.path.join(OUTPUT_DIR, "fig8_saturation_profiles.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig8_saturation_profiles.png")

# ── GIF Animation: NAPL migration ──────────────────────────────────────
if len(Sn_history) > 2:
    print("  Generating NAPL migration animation ...", flush=True)
    fig_a, ax_a = plt.subplots(figsize=(10, 8))

    sn_max_all = max(SN_SOURCE, max(sn.max() for _, _, sn in Sn_history))
    levels_anim = np.linspace(0, sn_max_all, 21)

    def animate_frame(frame_idx):
        ax_a.clear()
        s, t_s, sn_snap = Sn_history[frame_idx]
        sn_2d = sn_snap.reshape(Nx, Ny)
        h_2d_snap = h_f2d   # approximate: use final h for WT overlay
        ax_a.contourf(X2d, Y2d, sn_2d, levels=levels_anim, cmap="YlOrRd",
                      extend="max")
        ax_a.contour(X2d, Y2d, h_2d_snap, levels=[0.0], colors="blue",
                     linewidths=2, linestyles="--")
        ax_a.add_patch(Rectangle((SRC_X0, SRC_Y0), SRC_X1-SRC_X0,
                                 SRC_Y1-SRC_Y0, lw=2, ec='red', fc='none',
                                 ls='--'))
        ax_a.set_title(f"NAPL Saturation $S_n$  —  step {s},  t = {t_s:.1f} s",
                       fontsize=13)
        ax_a.set_xlabel("x [m]"); ax_a.set_ylabel("y [m]")
        ax_a.set_aspect("equal")

    anim = animation.FuncAnimation(fig_a, animate_frame,
                                   frames=len(Sn_history), interval=200)
    gif_path = os.path.join(OUTPUT_DIR, "napl_migration.gif")
    anim.save(gif_path, writer="pillow", fps=5, dpi=100)
    plt.close(fig_a)
    print(f"  Saved {gif_path}  ({len(Sn_history)} frames)")
else:
    print("  Skipping animation (< 3 snapshots)")

# ── Summary ─────────────────────────────────────────────────────────────
dh_max = np.max(np.abs(h_w - h_init))
q_expected = k_sat * (H_u - H_d) / Lx
sn_total = np.sum(S_n * Vp * phi_0)
sn_max_out = np.max(S_n[~is_source]) if np.any(~is_source) else 0.0

print(f"\n{'='*70}")
print(f"  SUMMARY  (Three-phase: Water + LNAPL + Air)")
print(f"{'='*70}")
print(f"  Max |h_final - h_initial|     = {dh_max:.6e} m")
print(f"  Final L2(dh_w/dt)             = {l2_log[-1]:.6e}")
print(f"  Final L2(dS_n/dt)             = {l2n_log[-1]:.6e}")
print(f"  Max Darcy |q_w|               = {np.max(qm2d):.6e} m/s")
print(f"  Max S_n (outside source)      = {sn_max_out:.6f}")
print(f"  Total NAPL volume (phi*Sn*V)  = {sn_total:.6e} m^3/m")
print(f"  k_sat (water)                 = {k_sat:.6e} m/s")
print(f"  k_sat (NAPL)                  = {k_sat_n:.6e} m/s")
print(f"  Expected q_w ~ k_sat*dH/Lx   = {q_expected:.6e} m/s")
if HAS_H5:
    print(f"  HDF5 snapshots                = {hdf5_path}")
print(f"  Checkpoints                   = {CKPT_DIR}/")
print(f"{'='*70}")

plt.close("all")
print("\nDone.")
