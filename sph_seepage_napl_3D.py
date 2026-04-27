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
OUTPUT_DIR = "../data_sph/"
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
Lx, Ly, Lz = 10.0, 10.0, 10.0

Nx = 51       # particles in x  (horizontal flow direction)
Ny = 51       # particles in y  (horizontal, perpendicular to flow)
Nz = 51       # particles in z  (vertical, gravity direction)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

print(f"\nDomain  {Lx} x {Ly} x {Lz} m    grid  {Nx} x {Ny} x {Nz}    "
      f"dx = {dx:.4f}  dy = {dy:.4f}  dz = {dz:.4f} m")

# Regular lattice positions
N_part = Nx * Ny * Nz
xp = np.zeros(N_part)
yp = np.zeros(N_part)
zp = np.zeros(N_part)
idx_3d = np.zeros((Nx, Ny, Nz), dtype=int)

pid = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            xp[pid] = i * dx
            yp[pid] = j * dy
            zp[pid] = k * dz
            idx_3d[i, j, k] = pid
            pid += 1

Vp     = dx * dy * dz       # particle volume (3-D)
h_sml  = 1.3 * dx           # smoothing length

print(f"Particles: {N_part}    h_sml = {h_sml:.4f} m")

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

H_u = 8.0
H_d = 6.0

ptype = np.zeros(N_part, dtype=int)
tol_bc = 0.5 * dx

for i in range(N_part):
    if   xp[i] < tol_bc:         ptype[i] = 1   # left  (x=0)
    elif xp[i] > Lx - tol_bc:    ptype[i] = 2   # right (x=Lx)
    elif zp[i] < tol_bc:         ptype[i] = 3   # bottom (z=0)
    elif zp[i] > Lz - tol_bc:    ptype[i] = 4   # top    (z=Lz)
    elif yp[i] < tol_bc:         ptype[i] = 5   # front  (y=0)
    elif yp[i] > Ly - tol_bc:    ptype[i] = 6   # back   (y=Ly)

n_left  = np.sum(ptype == 1)
n_right = np.sum(ptype == 2)
n_bot   = np.sum(ptype == 3)
n_top   = np.sum(ptype == 4)
n_front = np.sum(ptype == 5)
n_back  = np.sum(ptype == 6)
n_int   = np.sum(ptype == 0)
print(f"\nParticle types:  interior={n_int}  left={n_left}  right={n_right}  "
      f"bot={n_bot}  top={n_top}  front={n_front}  back={n_back}")

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
SRC_X0, SRC_X1 = 4.5, 5.5
SRC_Y0, SRC_Y1 = 4.5, 5.5
SRC_Z0, SRC_Z1 = 8.5, 9.5
SN_SOURCE       = 0.80       # fixed NAPL saturation at source

is_source = ((xp >= SRC_X0) & (xp <= SRC_X1) &
             (yp >= SRC_Y0) & (yp <= SRC_Y1) &
             (zp >= SRC_Z0) & (zp <= SRC_Z1))

n_src = np.sum(is_source)
print(f"NAPL source particles: {n_src}  "
      f"([{SRC_X0},{SRC_X1}] x [{SRC_Y0},{SRC_Y1}] x [{SRC_Z0},{SRC_Z1}],  Sn = {SN_SOURCE})")

# NAPL saturation field
S_n = np.zeros(N_part)
S_n[is_source] = SN_SOURCE

# ======================================================================
# 8.  BOUNDARY ENFORCEMENT  (vectorised with precomputed index arrays)
# ======================================================================
#
# 6 boundary face groups + source.  All loops eliminated via fancy indexing.

# Precompute boundary particle indices
_idx_left  = np.where(ptype == 1)[0]
_idx_right = np.where(ptype == 2)[0]
_idx_bot   = np.where(ptype == 3)[0]
_idx_top   = np.where(ptype == 4)[0]
_idx_front = np.where(ptype == 5)[0]
_idx_back  = np.where(ptype == 6)[0]
_idx_src   = np.where(is_source)[0]

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
CFL = 0.25

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

def save_snapshot(fname, step, t, h_field, Sn_field, dhdt, dSndt, qx, qy, qz):
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
            grp.attrs["Nz"]     = Nz
            grp.attrs["Lx"]     = Lx
            grp.attrs["Ly"]     = Ly
            grp.attrs["Lz"]     = Lz
            grp.attrs["k_sat"]  = k_sat
            grp.attrs["H_u"]    = H_u
            grp.attrs["H_d"]    = H_d

            grp.create_dataset("x",    data=xp)
            grp.create_dataset("y",    data=yp)
            grp.create_dataset("z",    data=zp)
            grp.create_dataset("h",    data=h_field)
            grp.create_dataset("H",    data=h_field + zp)
            grp.create_dataset("Sn",   data=Sn_field)
            grp.create_dataset("Sw",   data=Sw)
            grp.create_dataset("kw",   data=compute_kw_3ph(h_field, Sn_field))
            grp.create_dataset("kn",   data=compute_kn_field(h_field, Sn_field))
            grp.create_dataset("dhdt", data=dhdt)
            grp.create_dataset("dSndt", data=dSndt)
            grp.create_dataset("qx",   data=qx)
            grp.create_dataset("qy",   data=qy)
            grp.create_dataset("qz",   data=qz)
            grp.create_dataset("ptype", data=ptype)
            grp.create_dataset("is_source", data=is_source.astype(np.int8))
    else:
        # NPZ fallback: one file per snapshot in snapshots/ directory
        os.makedirs(SNAP_DIR, exist_ok=True)
        path = os.path.join(SNAP_DIR, f"step_{step:06d}.npz")
        np.savez_compressed(path,
            step=step, time_s=t, Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
            k_sat=k_sat, H_u=H_u, H_d=H_d,
            x=xp, y=yp, z=zp, h=h_field, H=h_field + zp,
            Sn=Sn_field, Sw=Sw,
            kw=compute_kw_3ph(h_field, Sn_field),
            kn=compute_kn_field(h_field, Sn_field),
            dhdt=dhdt, dSndt=dSndt, qx=qx, qy=qy, qz=qz,
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

N_steps_max    = 2000          # adjust as needed
snapshot_every = 200
print_every    = 100
ckpt_every     = 500
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

    enforce_dirichlet_bc(h_w)
    enforce_impermeable_bc(h_w)
    enforce_napl_bc(S_n)

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
        qx_s, qy_s, qz_s = compute_darcy_velocity(h_w, S_n)
        save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt, dSndt, qx_s, qy_s, qz_s)

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
qx_final, qy_final, qz_final = compute_darcy_velocity(h_w, S_n)
dhdt_f = compute_dhdt(h_w, S_n)
dSndt_f = compute_dSndt(h_w, S_n)
save_snapshot(hdf5_path, step, t_phys, h_w, S_n, dhdt_f, dSndt_f, qx_final, qy_final, qz_final)
print("  done.")


# ======================================================================
# 14.  POST-PROCESSING  &  FIGURES   (3D)
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Reshape all 1D fields into 3D arrays (Nx, Ny, Nz)
def to3d(arr):  return arr.reshape(Nx, Ny, Nz)

X3 = to3d(xp); Y3 = to3d(yp); Z3 = to3d(zp)
h_3   = to3d(h_w)
H_3   = to3d(h_w + zp)
Sn_3  = to3d(S_n)
Sw_3  = to3d(compute_Sw_3ph(h_w, S_n))
Sa_3  = np.clip(1.0 - Sw_3 - Sn_3, 0, 1)
qx3   = to3d(qx_final); qy3 = to3d(qy_final); qz3 = to3d(qz_final)
qm3   = np.sqrt(qx3**2 + qy3**2 + qz3**2)

qxn_f, qyn_f, qzn_f = compute_darcy_velocity_napl(h_w, S_n)
qxn3 = to3d(qxn_f); qyn3 = to3d(qyn_f); qzn3 = to3d(qzn_f)
qmn3 = np.sqrt(qxn3**2 + qyn3**2 + qzn3**2)

# Slice indices: at NAPL source center
ix_src = int(round(0.5*(SRC_X0+SRC_X1)/dx))
iy_src = int(round(0.5*(SRC_Y0+SRC_Y1)/dy))
ix_src = max(0, min(ix_src, Nx-1))
iy_src = max(0, min(iy_src, Ny-1))
print(f"Slice indices: ix_src={ix_src} (x={X3[ix_src,0,0]:.2f}m), "
      f"iy_src={iy_src} (y={Y3[0,iy_src,0]:.2f}m)")

# 1D coordinate vectors for streamplot/contour
x_axis = X3[:, 0, 0]   # length Nx
y_axis = Y3[0, :, 0]   # length Ny
z_axis = Z3[0, 0, :]   # length Nz

# 2D meshgrids for slice plots
YY_yz, ZZ_yz = np.meshgrid(y_axis, z_axis, indexing="ij")   # (Ny, Nz)
XX_xz, ZZ_xz = np.meshgrid(x_axis, z_axis, indexing="ij")   # (Nx, Nz)

# ── Helper: extract 2D slice fields ──────────────────────────────────
def slice_yz(field3, ix):  return field3[ix, :, :]   # (Ny, Nz)
def slice_xz(field3, iy):  return field3[:, iy, :]   # (Nx, Nz)


# ── Fig 1: NAPL Sn — YZ slice + XZ slice with streamlines ──────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
levels_sn = np.linspace(0, SN_SOURCE, 21)

# YZ slice at x = ix_src
Sn_yz = slice_yz(Sn_3, ix_src); h_yz = slice_yz(h_3, ix_src)
qy_yz = slice_yz(qyn3, ix_src); qz_yz = slice_yz(qzn3, ix_src)
cf1a = ax1a.contourf(YY_yz, ZZ_yz, Sn_yz, levels=levels_sn, cmap="YlOrRd", extend="max")
fig1.colorbar(cf1a, ax=ax1a, shrink=0.8, label=r"$S_n$")
ax1a.contour(YY_yz, ZZ_yz, h_yz, levels=[0.0], colors="blue", linewidths=2, linestyles="--")
if np.max(np.abs(qy_yz) + np.abs(qz_yz)) > 1e-25:
    ax1a.streamplot(y_axis, z_axis, qy_yz.T, qz_yz.T, color="k", linewidth=0.7, density=1.2, arrowsize=1.0)
ax1a.add_patch(Rectangle((SRC_Y0, SRC_Z0), SRC_Y1-SRC_Y0, SRC_Z1-SRC_Z0, lw=2, ec='red', fc='none', ls='--'))
ax1a.set_title(f"YZ slice at x = {x_axis[ix_src]:.2f} m  (NAPL streamlines)")
ax1a.set_xlabel("y [m]"); ax1a.set_ylabel("z [m]"); ax1a.set_aspect("equal")

# XZ slice at y = iy_src
Sn_xz = slice_xz(Sn_3, iy_src); h_xz = slice_xz(h_3, iy_src)
qx_xz = slice_xz(qxn3, iy_src); qz_xz = slice_xz(qzn3, iy_src)
cf1b = ax1b.contourf(XX_xz, ZZ_xz, Sn_xz, levels=levels_sn, cmap="YlOrRd", extend="max")
fig1.colorbar(cf1b, ax=ax1b, shrink=0.8, label=r"$S_n$")
ax1b.contour(XX_xz, ZZ_xz, h_xz, levels=[0.0], colors="blue", linewidths=2, linestyles="--")
if np.max(np.abs(qx_xz) + np.abs(qz_xz)) > 1e-25:
    ax1b.streamplot(x_axis, z_axis, qx_xz.T, qz_xz.T, color="k", linewidth=0.7, density=1.2, arrowsize=1.0)
ax1b.add_patch(Rectangle((SRC_X0, SRC_Z0), SRC_X1-SRC_X0, SRC_Z1-SRC_Z0, lw=2, ec='red', fc='none', ls='--'))
ax1b.set_title(f"XZ slice at y = {y_axis[iy_src]:.2f} m  (NAPL streamlines)")
ax1b.set_xlabel("x [m]"); ax1b.set_ylabel("z [m]"); ax1b.set_aspect("equal")

fig1.suptitle(f"NAPL Saturation $S_n$  (t = {t_phys:.1f} s)", fontsize=14)
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_napl_slices.png"), dpi=150, bbox_inches="tight")
print("  Saved fig1_napl_slices.png")
plt.close(fig1)


# ── Fig 2: Water head + water streamlines on both slices ──────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

H_yz = slice_yz(H_3, ix_src)
qy_yz_w = slice_yz(qy3, ix_src); qz_yz_w = slice_yz(qz3, ix_src)
cf2a = ax2a.contourf(YY_yz, ZZ_yz, H_yz, levels=30, cmap="coolwarm")
fig2.colorbar(cf2a, ax=ax2a, shrink=0.8, label="$H_w$ [m]")
ax2a.streamplot(y_axis, z_axis, qy_yz_w.T, qz_yz_w.T, color="k", linewidth=0.7, density=1.2)
ax2a.contour(YY_yz, ZZ_yz, slice_yz(h_3, ix_src), levels=[0.0], colors="k", linewidths=1.5, linestyles="--")
ax2a.contour(YY_yz, ZZ_yz, Sn_yz, levels=[0.01], colors="red", linewidths=1.5)
ax2a.set_title(f"YZ slice at x = {x_axis[ix_src]:.2f} m")
ax2a.set_xlabel("y [m]"); ax2a.set_ylabel("z [m]"); ax2a.set_aspect("equal")

H_xz = slice_xz(H_3, iy_src)
qx_xz_w = slice_xz(qx3, iy_src); qz_xz_w = slice_xz(qz3, iy_src)
cf2b = ax2b.contourf(XX_xz, ZZ_xz, H_xz, levels=30, cmap="coolwarm")
fig2.colorbar(cf2b, ax=ax2b, shrink=0.8, label="$H_w$ [m]")
ax2b.streamplot(x_axis, z_axis, qx_xz_w.T, qz_xz_w.T, color="k", linewidth=0.7, density=1.5)
ax2b.contour(XX_xz, ZZ_xz, slice_xz(h_3, iy_src), levels=[0.0], colors="k", linewidths=1.5, linestyles="--")
ax2b.contour(XX_xz, ZZ_xz, Sn_xz, levels=[0.01], colors="red", linewidths=1.5)
ax2b.set_title(f"XZ slice at y = {y_axis[iy_src]:.2f} m")
ax2b.set_xlabel("x [m]"); ax2b.set_ylabel("z [m]"); ax2b.set_aspect("equal")

fig2.suptitle("Water Head $H_w$ & Water Streamlines", fontsize=14)
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_water_slices.png"), dpi=150, bbox_inches="tight")
print("  Saved fig2_water_slices.png")
plt.close(fig2)


# ── Fig 3: Sn vertical profile evolution at (x_src, y_src) ─────────────
fig3, ax3 = plt.subplots(figsize=(8, 8), constrained_layout=True)
i_col = idx_3d[ix_src, iy_src, :]   # vertical line of indices
cmap_evo = plt.cm.inferno
n_snaps = len(Sn_history)
for idx, (s, t_s, sn_snap) in enumerate(Sn_history):
    col = cmap_evo(idx / max(n_snaps - 1, 1))
    show_lab = (idx % max(1, n_snaps//6) == 0 or idx == n_snaps-1)
    ax3.plot(sn_snap[i_col], z_axis, color=col, lw=1.5,
             label=f"t={t_s:.1f} s" if show_lab else None)
zwt0 = H_u + (H_d - H_u) * X3[ix_src, 0, 0] / Lx
ax3.axhline(zwt0, color='blue', ls='--', lw=1, alpha=0.5, label='initial WT')
ax3.set_xlabel(r"$S_n$ [-]"); ax3.set_ylabel("z [m]")
ax3.set_title(f"NAPL profile at (x={x_axis[ix_src]:.1f}, y={y_axis[iy_src]:.1f}) m")
ax3.legend(fontsize=8, loc='upper right'); ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.02, SN_SOURCE + 0.05)
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_Sn_evolution_vertical.png"), dpi=150, bbox_inches="tight")
print("  Saved fig3_Sn_evolution_vertical.png")
plt.close(fig3)


# ── Fig 4: NAPL mass balance ───────────────────────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
ax4a.plot(time_log, napl_mass_log, 'k-', lw=1.5, label="Total (incl. source)")
ax4a.plot(time_log, napl_mass_nosrc_log, 'r-', lw=1.5, label="Outside source")
ax4a.set_xlabel("Time [s]"); ax4a.set_ylabel(r"NAPL volume $\phi S_n V$  [m$^3$]")
ax4a.set_title("NAPL Mass Balance"); ax4a.legend(); ax4a.grid(True, alpha=0.3)

if len(napl_mass_nosrc_log) > 2:
    t_arr = np.array(time_log); m_arr = np.array(napl_mass_nosrc_log)
    rate = np.diff(m_arr) / np.maximum(np.diff(t_arr), 1e-30)
    ax4b.plot(t_arr[1:], rate, 'r-', lw=1)
    ax4b.set_xlabel("Time [s]"); ax4b.set_ylabel(r"d(NAPL)/dt [m$^3$/s]")
    ax4b.set_title("Source Discharge Rate"); ax4b.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_napl_mass_balance.png"), dpi=150, bbox_inches="tight")
print("  Saved fig4_napl_mass_balance.png")
plt.close(fig4)


# ── Fig 5: Convergence + NAPL front ───────────────────────────────────
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
ax5a.semilogy(time_log, l2_log, "b-", lw=1, label=r"L2($dh_w/dt$)")
ax5a.semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
ax5a.set_xlabel("Time [s]"); ax5a.set_ylabel("Residual")
ax5a.set_title("Convergence"); ax5a.legend(); ax5a.grid(True, which="both", alpha=0.3)

ax5b.plot(time_log, Sn_max_log, "r-", lw=1.5)
ax5b.set_xlabel("Time [s]"); ax5b.set_ylabel(r"max $S_n$ (outside source)")
ax5b.set_title("NAPL Front"); ax5b.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_convergence.png"), dpi=150, bbox_inches="tight")
print("  Saved fig5_convergence.png")
plt.close(fig5)


# ── Fig 6: Three-phase profile at slice center ───────────────────────
fig6, ax6 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax6.plot(Sw_3[ix_src, iy_src, :], z_axis, "b-", lw=2, label=r"$S_w$")
ax6.plot(Sn_3[ix_src, iy_src, :], z_axis, "r-", lw=2, label=r"$S_n$")
ax6.plot(Sa_3[ix_src, iy_src, :], z_axis, "g--", lw=1.5, label=r"$S_a$")
ax6.set_xlabel("Saturation [-]"); ax6.set_ylabel("z [m]")
ax6.set_title(f"Saturation profiles at (x={x_axis[ix_src]:.1f}, y={y_axis[iy_src]:.1f}) m")
ax6.legend(); ax6.grid(True, alpha=0.3); ax6.set_xlim(-0.05, 1.05)
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_saturation_profiles.png"), dpi=150, bbox_inches="tight")
print("  Saved fig6_saturation_profiles.png")
plt.close(fig6)


# ── Combined animation: YZ slice + XZ slice side-by-side ──────────────
if len(Sn_history) > 2:
    print("  Generating combined slice animation ...", flush=True)
    fig_a, (ax_a_yz, ax_a_xz) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    sn_max_all = max(SN_SOURCE, max(sn.max() for _, _, sn in Sn_history))
    levels_anim = np.linspace(0, sn_max_all, 21)

    def animate_frame(frame_idx):
        ax_a_yz.clear(); ax_a_xz.clear()
        s, t_s, sn_snap = Sn_history[frame_idx]
        sn3_snap = sn_snap.reshape(Nx, Ny, Nz)
        # Static water-table approximation: use final h
        h_yz_a = slice_yz(h_3, ix_src)
        h_xz_a = slice_xz(h_3, iy_src)

        # YZ
        ax_a_yz.contourf(YY_yz, ZZ_yz, slice_yz(sn3_snap, ix_src),
                          levels=levels_anim, cmap="YlOrRd", extend="max")
        ax_a_yz.contour(YY_yz, ZZ_yz, h_yz_a, levels=[0.0], colors="blue",
                         linewidths=2, linestyles="--")
        ax_a_yz.add_patch(Rectangle((SRC_Y0, SRC_Z0), SRC_Y1-SRC_Y0, SRC_Z1-SRC_Z0,
                                     lw=2, ec='red', fc='none', ls='--'))
        ax_a_yz.set_title(f"YZ at x={x_axis[ix_src]:.2f} m")
        ax_a_yz.set_xlabel("y [m]"); ax_a_yz.set_ylabel("z [m]"); ax_a_yz.set_aspect("equal")

        # XZ
        ax_a_xz.contourf(XX_xz, ZZ_xz, slice_xz(sn3_snap, iy_src),
                          levels=levels_anim, cmap="YlOrRd", extend="max")
        ax_a_xz.contour(XX_xz, ZZ_xz, h_xz_a, levels=[0.0], colors="blue",
                         linewidths=2, linestyles="--")
        ax_a_xz.add_patch(Rectangle((SRC_X0, SRC_Z0), SRC_X1-SRC_X0, SRC_Z1-SRC_Z0,
                                     lw=2, ec='red', fc='none', ls='--'))
        ax_a_xz.set_title(f"XZ at y={y_axis[iy_src]:.2f} m")
        ax_a_xz.set_xlabel("x [m]"); ax_a_xz.set_ylabel("z [m]"); ax_a_xz.set_aspect("equal")

        fig_a.suptitle(f"NAPL $S_n$  —  step {s},  t = {t_s:.1f} s", fontsize=14)

    anim = animation.FuncAnimation(fig_a, animate_frame,
                                    frames=len(Sn_history), interval=200)
    gif_path = os.path.join(OUTPUT_DIR, "napl_migration_slices.gif")
    try:
        anim.save(gif_path, writer="pillow", fps=5, dpi=100)
        print(f"  Saved {gif_path}  ({len(Sn_history)} frames)")
    except Exception as e:
        print(f"  Animation save failed: {e}")
    plt.close(fig_a)
else:
    print("  Skipping animation (< 3 snapshots)")


# ── Summary ───────────────────────────────────────────────────────────
dh_max = np.max(np.abs(h_w - h_init))
q_expected = k_sat * (H_u - H_d) / Lx
sn_total = np.sum(S_n * Vp * phi_0)
sn_max_out = np.max(S_n[~is_source]) if np.any(~is_source) else 0.0

print(f"\n{'='*70}")
print(f"  SUMMARY  (3D Three-phase: Water + LNAPL + Air)")
print(f"{'='*70}")
print(f"  Domain                        = {Lx} x {Ly} x {Lz} m")
print(f"  Grid                          = {Nx} x {Ny} x {Nz} = {N_part} particles")
print(f"  Max |h_final - h_initial|     = {dh_max:.6e} m")
print(f"  Final L2(dh_w/dt)             = {l2_log[-1]:.6e}")
print(f"  Final L2(dS_n/dt)             = {l2n_log[-1]:.6e}")
print(f"  Max Darcy |q_w|               = {np.max(qm3):.6e} m/s")
print(f"  Max S_n (outside source)      = {sn_max_out:.6f}")
print(f"  Total NAPL volume (phi*Sn*V)  = {sn_total:.6e} m^3")
print(f"  k_sat (water)                 = {k_sat:.6e} m/s")
print(f"  k_sat (NAPL)                  = {k_sat_n:.6e} m/s")
print(f"  Expected q_w ~ k_sat*dH/Lx    = {q_expected:.6e} m/s")
if HAS_H5:
    print(f"  HDF5 snapshots                = {hdf5_path}")
print(f"  Checkpoints                   = {CKPT_DIR}/")
print(f"{'='*70}")

plt.close("all")
print("\nDone.")
