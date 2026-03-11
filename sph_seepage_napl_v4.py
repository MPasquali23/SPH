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

# ======================================================================
# 0.  OUTPUT DIRECTORY
# ======================================================================
OUTPUT_DIR = "../data_sph/v4"
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

def compute_dhdt(h_field, Sn_field):
    """RHS of the water seepage equation (three-phase k_w)."""
    k_h   = compute_kw_3ph(h_field, Sn_field)
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
# 9b. NAPL RHS:  dS_n/dt  =  (1/phi) div[ k_n grad(H_n) ]
# ======================================================================

def compute_dSndt(h_field, Sn_field):
    """NAPL transport equation RHS.  Same SPH Laplacian structure."""
    k_n = compute_kn_field(h_field, Sn_field)
    H_n = compute_Hn_field(h_field, Sn_field)

    dSndt = np.zeros(N_part)

    for i in range(N_part):
        if ptype[i] == 1 or ptype[i] == 2:
            continue
        if is_source[i]:
            continue

        Hni = H_n[i]
        kni = k_n[i]

        # corrected gradient of H_n
        raw_gx = 0.0; raw_gy = 0.0
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            dHn = H_n[j] - Hni
            raw_gx += Vp * dHn * gWx
            raw_gy += Vp * dHn * gWy
        Li = L_inv[i]
        grad_Hx = Li[0, 0] * raw_gx + Li[0, 1] * raw_gy
        grad_Hy = Li[1, 0] * raw_gx + Li[1, 1] * raw_gy

        # variable-kn Laplacian
        lap_sum = 0.0
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            kn_mean = 0.5 * (kni + k_n[j])
            lap_sum += Vp * kn_mean * (H_n[j] - Hni) * Fhat

        corr = kni * (grad_Hx * err_x[i] + grad_Hy * err_y[i])

        Kn_val = K_norm[i]
        if abs(Kn_val) < 1e-30:
            continue

        rhs = (2.0 / Kn_val) * (lap_sum - corr)
        dSndt[i] = rhs / phi_0

    return dSndt


# ======================================================================
# 10.  DARCY VELOCITY   q = -k grad(H)    [Paper Eq. 60 + sign]
# ======================================================================
#
# Paper Eq. 60 computes  sum V_j k_ij (H_j-H_i) grad_W_ij  which is
# the SPH-consistent approximation of  k * grad(H).
# Physical Darcy velocity:  q = -k grad(H),  so we negate.

def compute_darcy_velocity(h_field, Sn_field):
    """Water Darcy velocity  q_w = -k_w grad(H_w)  (three-phase)."""
    k_h = compute_kw_3ph(h_field, Sn_field)
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


def compute_darcy_velocity_napl(h_field, Sn_field):
    """NAPL Darcy velocity  q_n = -k_n grad(H_n)."""
    k_n = compute_kn_field(h_field, Sn_field)
    H_n = compute_Hn_field(h_field, Sn_field)

    qx = np.zeros(N_part)
    qy = np.zeros(N_part)

    for i in range(N_part):
        for (j, rij, xji, yji, Fhat, gWx, gWy) in neighbours[i]:
            kn_mean = 0.5 * (k_n[i] + k_n[j])
            dHn = H_n[j] - H_n[i]
            L = L_inv[i]
            cWx = L[0, 0] * gWx + L[0, 1] * gWy
            cWy = L[1, 0] * gWx + L[1, 1] * gWy
            qx[i] += Vp * kn_mean * dHn * cWx
            qy[i] += Vp * kn_mean * dHn * cWy

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

def save_snapshot(fname, step, t, h_field, Sn_field, dhdt, dSndt, qx, qy):
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

        Sw = compute_Sw_3ph(h_field, Sn_field)
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

N_steps_max    = 100000          # adjust as needed
snapshot_every = 1000
print_every    = 500
ss_tol         = 1e-14         # steady-state tolerance on L2(dh/dt)

time_log = []
l2_log   = []
l2_max_log = []
l2n_log    = []
Sn_max_log = []
t_phys   = 0.0

print(f"\n{'='*70}")
print(f"  THREE-PHASE SPH: Water + LNAPL + Air")
print(f"  Max steps = {N_steps_max},  NAPL source at "
      f"[{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}]")
print(f"{'='*70}")
t_wall0 = wall_time.time()

# Mask for convergence metric: interior + impermeable boundaries
# (excludes Dirichlet boundaries where dhdt=0 trivially)
mask_conv = (ptype == 0) | (ptype == 3) | (ptype == 4)
mask_not_src = ~is_source & mask_conv

for step in range(1, N_steps_max + 1):
    dt = stable_dt(h_w, S_n)

    # RHS for both phases
    dhdt  = compute_dhdt(h_w, S_n)
    dSndt = compute_dSndt(h_w, S_n)

    # Update
    h_w += dt * dhdt
    S_n += dt * dSndt

    # Clamp NAPL saturation
    S_n = np.clip(S_n, 0.0, 1.0 - S_res)

    # Re-impose water BCs
    for i in range(N_part):
        if ptype[i] == 1:  h_w[i] = H_u - yp[i]
        if ptype[i] == 2:  h_w[i] = H_d - yp[i]
    h_w = enforce_impermeable_bc(h_w)

    # Re-impose NAPL BCs
    S_n = enforce_napl_bc(S_n)

    t_phys += dt

    # Convergence metrics
    l2     = np.sqrt(np.mean(dhdt[mask_conv]**2))
    l2_max = np.max(np.abs(dhdt[mask_conv]))
    l2n    = np.sqrt(np.mean(dSndt[mask_not_src]**2)) if np.any(mask_not_src) else 0.0
    sn_max = np.max(S_n[~is_source]) if np.any(~is_source) else 0.0

    time_log.append(t_phys)
    l2_log.append(l2)
    l2_max_log.append(l2_max)
    l2n_log.append(l2n)
    Sn_max_log.append(sn_max)

    # Print
    if step % print_every == 0 or step == 1:
        print(f"  step {step:5d}  t = {t_phys:.4e} s  dt = {dt:.4e} s  "
              f"L2w = {l2:.4e}  L2n = {l2n:.4e}  Sn_max = {sn_max:.4f}")

    # HDF5 snapshot
    if step % snapshot_every == 0 or step == 1:
        qx_s, qy_s = compute_darcy_velocity(h_w, S_n)
        save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt, dSndt, qx_s, qy_s)

    # Early exit on convergence
    if l2 < ss_tol and step > 10:
        print(f"\n  *** Converged at step {step}:  L2 = {l2:.4e}  < {ss_tol:.1e}")
        break

t_wall = wall_time.time() - t_wall0
print(f"\n  Wall time: {t_wall:.1f} s    Physical time: {t_phys:.4e} s")

# Final snapshot
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

X2d = xp.reshape(Nx, Ny)
Y2d = yp.reshape(Nx, Ny)

h_i2d  = h_init.reshape(Nx, Ny)
h_f2d  = h_w.reshape(Nx, Ny)
H_f2d  = (h_w + yp).reshape(Nx, Ny)
kw_2d  = compute_kw_3ph(h_w, S_n).reshape(Nx, Ny)
Sn_2d  = S_n.reshape(Nx, Ny)
Sw_2d  = compute_Sw_3ph(h_w, S_n).reshape(Nx, Ny)
qx2d   = qx_final.reshape(Nx, Ny)
qy2d   = qy_final.reshape(Nx, Ny)
qm2d   = np.sqrt(qx2d**2 + qy2d**2)

# Also compute NAPL velocity for plots
qxn_f, qyn_f = compute_darcy_velocity_napl(h_w, S_n)
qxn2d  = qxn_f.reshape(Nx, Ny)
qyn2d  = qyn_f.reshape(Nx, Ny)
qmn2d  = np.sqrt(qxn2d**2 + qyn2d**2)

from matplotlib.patches import Rectangle

# -- Fig 1: NAPL saturation field --
fig1, ax1 = plt.subplots(figsize=(10, 8), constrained_layout=True)
levels_sn = np.linspace(0, SN_SOURCE, 21)
cf1 = ax1.contourf(X2d, Y2d, Sn_2d, levels=levels_sn, cmap="YlOrRd", extend="max")
fig1.colorbar(cf1, ax=ax1, shrink=0.85, label=r"$S_n$ [-]")
ax1.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="blue", linewidths=2, linestyles="--")
ax1.add_patch(Rectangle((SRC_X0, SRC_Y0), 1, 1, lw=2, ec='red', fc='none', ls='--'))
skip = max(1, Nx // 15)
qn_scale = max(np.max(qmn2d) * 15, 1e-20)
ax1.quiver(X2d[::skip, ::skip], Y2d[::skip, ::skip],
           qxn2d[::skip, ::skip], qyn2d[::skip, ::skip],
           color="k", scale=qn_scale, width=0.003, alpha=0.7)
ax1.set_title(f"NAPL Saturation $S_n$ (t = {t_phys:.1f} s)", fontsize=13)
ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_aspect("equal")
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_napl_saturation.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig1_napl_saturation.png")

# -- Fig 2: Water saturation (showing NAPL displacement) --
fig2, ax2 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf2 = ax2.contourf(X2d, Y2d, Sw_2d, levels=np.linspace(0, 1, 21), cmap="Blues")
fig2.colorbar(cf2, ax=ax2, shrink=0.85, label=r"$S_w$ [-]")
ax2.contour(X2d, Y2d, Sn_2d, levels=[0.01], colors="red", linewidths=1.5)
ax2.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=1.5, linestyles="--")
ax2.set_title("Water Saturation $S_w$", fontsize=13)
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.set_aspect("equal")
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_water_saturation.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig2_water_saturation.png")

# -- Fig 3: Water hydraulic head + velocity --
fig3, ax3 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf3 = ax3.contourf(X2d, Y2d, H_f2d, levels=30, cmap="coolwarm")
fig3.colorbar(cf3, ax=ax3, shrink=0.85, label="$H_w$ [m]")
qw_scale = max(np.max(qm2d) * 15, 1e-20)
ax3.quiver(X2d[::skip, ::skip], Y2d[::skip, ::skip],
           qx2d[::skip, ::skip], qy2d[::skip, ::skip],
           color="k", scale=qw_scale, width=0.003)
ax3.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="k", linewidths=1.5, linestyles="--")
ax3.contour(X2d, Y2d, Sn_2d, levels=[0.01], colors="red", linewidths=2)
ax3.set_title("Water Head $H_w$ & Darcy Velocity", fontsize=13)
ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_aspect("equal")
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_water_head_velocity.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig3_water_head_velocity.png")

# -- Fig 4: Three-phase saturation profiles (vertical at x=5) --
ix_mid = Nx // 2
Sa_2d = np.clip(1.0 - Sw_2d - Sn_2d, 0, 1)
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
ax4a.plot(Sw_2d[ix_mid, :], Y2d[ix_mid, :], "b-", lw=2, label=r"$S_w$")
ax4a.plot(Sn_2d[ix_mid, :], Y2d[ix_mid, :], "r-", lw=2, label=r"$S_n$")
ax4a.plot(Sa_2d[ix_mid, :], Y2d[ix_mid, :], "g--", lw=1.5, label=r"$S_a$")
ax4a.set_xlabel("Saturation [-]"); ax4a.set_ylabel("y [m]")
ax4a.set_title(f"Saturation profiles at x = {X2d[ix_mid,0]:.1f} m")
ax4a.legend(); ax4a.grid(True, alpha=0.3); ax4a.set_xlim(-0.05, 1.05)

for ix_val in [Nx//5, 2*Nx//5, Nx//2, 3*Nx//5, 4*Nx//5]:
    ax4b.plot(Sn_2d[ix_val, :], Y2d[ix_val, :], label=f"x={X2d[ix_val,0]:.1f}")
ax4b.set_xlabel(r"$S_n$ [-]"); ax4b.set_ylabel("y [m]")
ax4b.set_title("NAPL saturation at various x"); ax4b.legend(); ax4b.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_saturation_profiles.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig4_saturation_profiles.png")

# -- Fig 5: velocity magnitude (water) --
fig5, ax5 = plt.subplots(figsize=(10, 8), constrained_layout=True)
cf5 = ax5.contourf(X2d, Y2d, qm2d, levels=30, cmap="viridis")
fig5.colorbar(cf5, ax=ax5, shrink=0.85, label="|$q_w$| [m/s]")
ax5.contour(X2d, Y2d, h_f2d, levels=[0.0], colors="w", linewidths=1.5, linestyles="--")
ax5.contour(X2d, Y2d, Sn_2d, levels=[0.01], colors="red", linewidths=2)
ax5.set_title("Water Darcy Velocity Magnitude", fontsize=13)
ax5.set_xlabel("x [m]"); ax5.set_ylabel("y [m]"); ax5.set_aspect("equal")
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_water_velocity.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig5_water_velocity.png")

# -- Fig 6: Convergence + NAPL front --
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))
ax6a.semilogy(time_log, l2_log, "b-", lw=1, label=r"L2($dh_w/dt$)")
ax6a.semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
ax6a.set_xlabel("Time [s]"); ax6a.set_ylabel("Residual")
ax6a.set_title("Convergence History"); ax6a.legend(); ax6a.grid(True, which="both", alpha=0.3)

ax6b.plot(time_log, Sn_max_log, "r-", lw=1)
ax6b.set_xlabel("Time [s]"); ax6b.set_ylabel(r"max $S_n$ (outside source)")
ax6b.set_title("NAPL Front Propagation"); ax6b.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_convergence_napl.png"),
             dpi=150, bbox_inches="tight")
print("  Saved fig6_convergence_napl.png")

# -- Summary --
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
print(f"{'='*70}")

plt.close("all")
print("\nDone.")
