#!/usr/bin/env python3
"""
SPH Three-Phase Seepage — Performance Benchmark (3D)
=====================================================

Self-contained benchmark replicating the 3D physics of sph_seepage_napl.py.
Times setup + main loop + per-component breakdown. No checkpoints, no
animation — just raw computation + timing.

Usage
-----
    NUMBA_NUM_THREADS=8 python benchmark_sph.py --nx 51 --nsteps 50
    NUMBA_DISABLE_JIT=1 python benchmark_sph.py --nx 21 --nsteps 20

Note: --nx sets Nx = Ny = Nz (cubic grid).
"""

import argparse
import os
import time as wall_time
import numpy as np

# ── Parse args BEFORE importing numba ──────────────────────────────────
parser = argparse.ArgumentParser(description="SPH three-phase 3D benchmark")
parser.add_argument("--nx", type=int, default=51, help="Grid side (Nx=Ny=Nz)")
parser.add_argument("--nsteps", type=int, default=50, help="Number of steps")
parser.add_argument("--outdir", type=str, default=".", help="Output directory")
args = parser.parse_args()

Nx_arg   = args.nx
N_steps  = args.nsteps
OUT_DIR  = args.outdir
os.makedirs(OUT_DIR, exist_ok=True)

n_threads = int(os.environ.get("NUMBA_NUM_THREADS", "1"))
jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    mode_str = f"Numba JIT, {n_threads} thread(s)"
    if jit_disabled:
        mode_str = "Numba DISABLED (pure Python)"
except ImportError:
    HAS_NUMBA = False
    mode_str = "Pure Python (no Numba)"

print(f"Mode           : {mode_str}")
print(f"Grid           : {Nx_arg} x {Nx_arg} x {Nx_arg} (3D)")
print(f"Steps          : {N_steps}")

# ======================================================================
# PHYSICAL PARAMETERS
# ======================================================================
k_abs=2.059e-11; phi_0=0.43; c_R=4.35e-7
rho_W=1000.0; mu_w=1.0e-3; g_acc=9.81
n_vG=2.68; p_caw0=676.55
gamma_w = rho_W * g_acc
k_sat = k_abs * rho_W * g_acc / mu_w
m_vG = 1.0 - 1.0 / n_vG
g_a = gamma_w / p_caw0
S_sat=1.0; S_res=0.045; S_wir=S_res; g_l=0.5
K_s=1.0/c_R; K_sat_l=2.0e9
alpha_hat=(1.0-phi_0)/K_s; beta_hat=1.0/K_sat_l
C_l = gamma_w * (phi_0 * S_sat * beta_hat + alpha_hat)

rho_N=830.0; mu_n=3.61e-3
gamma_n = rho_N * g_acc
k_sat_n = k_abs * rho_N * g_acc / mu_n
sigma_aw=0.065; sigma_nw=0.030
beta_nw = sigma_aw / sigma_nw
alpha_nw = beta_nw * g_a
CFL = 0.25

# ======================================================================
# DOMAIN (3D)
# ======================================================================
Lx, Ly, Lz = 10.0, 10.0, 10.0
Nx = Ny = Nz = Nx_arg
dx = Lx / (Nx - 1); dy = Ly / (Ny - 1); dz = Lz / (Nz - 1)

N_part = Nx * Ny * Nz
xp = np.zeros(N_part); yp = np.zeros(N_part); zp = np.zeros(N_part)
idx_3d = np.zeros((Nx, Ny, Nz), dtype=int)

pid = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            xp[pid] = i * dx; yp[pid] = j * dy; zp[pid] = k * dz
            idx_3d[i, j, k] = pid
            pid += 1

Vp = dx * dy * dz
h_sml = 1.3 * dx

print(f"Particles      : {N_part}")
print(f"dx = dy = dz   : {dx:.4f} m")

# ======================================================================
# KERNEL (3D cubic spline)
# ======================================================================
alpha_d = 3.0 / (2.0 * np.pi * h_sml**3)

def dW_dr(r):
    q = r / h_sml
    if q < 1e-14 or q >= 2.0: return 0.0
    if q < 1.0: return alpha_d * (-2.0*q + 1.5*q*q) / h_sml
    return alpha_d * (-0.5) * (2.0 - q)**2 / h_sml

# ======================================================================
# NEIGHBOURS + CSR (3D)
# ======================================================================
support_r = 2.0 * h_sml
print("Building neighbours ...", end=" ", flush=True)
t_setup0 = wall_time.perf_counter()

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([xp, yp, zp]))
nbl  = tree.query_ball_tree(tree, r=support_r)

_neighbours = [[] for _ in range(N_part)]
for i in range(N_part):
    xi, yi, zi = xp[i], yp[i], zp[i]
    for j in nbl[i]:
        if j == i: continue
        xji = xp[j] - xi; yji = yp[j] - yi; zji = zp[j] - zi
        r = np.sqrt(xji**2 + yji**2 + zji**2)
        if r < 1e-30: continue
        dWr = dW_dr(r)
        gWx = dWr * (-xji) / r
        gWy = dWr * (-yji) / r
        gWz = dWr * (-zji) / r
        dot = xji*gWx + yji*gWy + zji*gWz
        Fhat = dot / (r * r)
        _neighbours[i].append((j, r, xji, yji, zji, Fhat, gWx, gWy, gWz))

_total = sum(len(nb) for nb in _neighbours)
nbr_j    = np.empty(_total, dtype=np.int64)
nbr_Fhat = np.empty(_total, dtype=np.float64)
nbr_gWx  = np.empty(_total, dtype=np.float64)
nbr_gWy  = np.empty(_total, dtype=np.float64)
nbr_gWz  = np.empty(_total, dtype=np.float64)
nbr_ptr  = np.empty(N_part + 1, dtype=np.int64)

off = 0
for i in range(N_part):
    nbr_ptr[i] = off
    for (j, rij, xji, yji, zji, Fhat, gWx, gWy, gWz) in _neighbours[i]:
        nbr_j[off]=j; nbr_Fhat[off]=Fhat
        nbr_gWx[off]=gWx; nbr_gWy[off]=gWy; nbr_gWz[off]=gWz
        off += 1
nbr_ptr[N_part] = off
print(f"CSR {_total} entries.", flush=True)

# ======================================================================
# CORRECTION MATRICES (3x3)
# ======================================================================
print("Correction matrices (3x3) ...", end=" ", flush=True)
K_norm = np.zeros(N_part)
err_x  = np.zeros(N_part); err_y = np.zeros(N_part); err_z = np.zeros(N_part)
L_inv  = np.zeros((N_part, 3, 3))

for i in range(N_part):
    Kn=0.0; ex=0.0; ey=0.0; ez=0.0
    M = np.zeros((3, 3))
    for (j, rij, xji, yji, zji, Fhat, gWx, gWy, gWz) in _neighbours[i]:
        Kn += Vp * (xji**2 + yji**2 + zji**2) * Fhat
        ex += Vp * xji * Fhat
        ey += Vp * yji * Fhat
        ez += Vp * zji * Fhat
        M[0,0]+=Vp*xji*gWx; M[0,1]+=Vp*xji*gWy; M[0,2]+=Vp*xji*gWz
        M[1,0]+=Vp*yji*gWx; M[1,1]+=Vp*yji*gWy; M[1,2]+=Vp*yji*gWz
        M[2,0]+=Vp*zji*gWx; M[2,1]+=Vp*zji*gWy; M[2,2]+=Vp*zji*gWz
    K_norm[i] = 0.5 * Kn
    err_x[i] = ex; err_y[i] = ey; err_z[i] = ez
    try:
        L_inv[i] = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        L_inv[i] = np.eye(3)

del _neighbours
t_setup1 = wall_time.perf_counter()
t_setup = t_setup1 - t_setup0
print(f"done.  Setup: {t_setup:.2f} s", flush=True)

# ======================================================================
# CONSTITUTIVE (NumPy) — fallback path
# ======================================================================
def compute_Cs(h):
    Cs=np.zeros_like(h); mask=h<0; ah=np.abs(h[mask]); ahn=(g_a*ah)**n_vG
    Cs[mask]=phi_0*((S_sat-S_res)*m_vG*n_vG*g_a*(g_a*ah)**(n_vG-1)/(1+ahn)**(m_vG+1))
    return Cs

def compute_Sw_3ph(h_w, S_n):
    Sw_vg=np.where(h_w>=0,S_sat,S_wir+(1-S_wir)*(1+(g_a*np.abs(h_w))**n_vG)**(-m_vG))
    return np.clip(np.minimum(Sw_vg,1-S_n),S_wir,S_sat)

def compute_Sew(h_w, S_n):
    Sw=compute_Sw_3ph(h_w,S_n); return np.clip((Sw-S_wir)/(1-S_wir),1e-14,1-1e-10)

def compute_Set(h_w, S_n):
    Sw=compute_Sw_3ph(h_w,S_n); return np.clip((Sw+S_n-S_wir)/(1-S_wir),1e-14,1-1e-10)

def compute_krw_3ph(h_w, S_n):
    Sew=compute_Sew(h_w,S_n); inner=np.clip(1-Sew**(1/m_vG),0,1)
    return np.clip(Sew**g_l*(1-inner**m_vG)**2,0,1)

def compute_krn(h_w, S_n):
    Sew=compute_Sew(h_w,S_n); Set=compute_Set(h_w,S_n)
    Se_n=np.clip(Set-Sew,1e-14,None)
    tw=np.clip((1-Sew**(1/m_vG))**m_vG,0,1); tt=np.clip((1-Set**(1/m_vG))**m_vG,0,1)
    return np.clip(Se_n**g_l*np.clip(tw-tt,0,None)**2,0,1)

def compute_kw_3ph(h_w, S_n):
    k=k_sat*compute_krw_3ph(h_w,S_n); k[(h_w>=0)&(S_n<1e-12)]=k_sat; return k

def compute_kn_field(h_w, S_n):
    return k_sat_n * compute_krn(h_w, S_n)

def compute_Hn_field(h_w, S_n):
    Sew=compute_Sew(h_w,S_n)
    base=np.clip(Sew**(-1/m_vG)-1,0,1e12)
    h_cnw=(1/alpha_nw)*base**(1/n_vG)
    return (gamma_w/gamma_n)*(h_w+h_cnw)+zp

# Precomputed constants for scalar helpers
_gamma_ratio = gamma_w / gamma_n
_inv_alpha_nw = 1.0 / alpha_nw
_inv_m = 1.0 / m_vG
_inv_n = 1.0 / n_vG

# ======================================================================
# SCALAR NUMBA HELPERS
# ======================================================================
if HAS_NUMBA:
    @njit(inline='always')
    def _s_Sw(hw, sn):
        if hw >= 0.0: sw = S_sat
        else: sw = S_wir + (1.0-S_wir)*(1.0+(g_a*abs(hw))**n_vG)**(-m_vG)
        cap = 1.0 - sn
        if sw > cap: sw = cap
        if sw < S_wir: sw = S_wir
        if sw > S_sat: sw = S_sat
        return sw
    @njit(inline='always')
    def _s_Sew(sw):
        v = (sw - S_wir)/(1.0-S_wir)
        if v < 1e-14: v = 1e-14
        if v > 1.0-1e-10: v = 1.0-1e-10
        return v
    @njit(inline='always')
    def _s_Set(sw, sn):
        v = (sw+sn-S_wir)/(1.0-S_wir)
        if v < 1e-14: v = 1e-14
        if v > 1.0-1e-10: v = 1.0-1e-10
        return v
    @njit(inline='always')
    def _s_kw(hw, sn):
        if hw >= 0.0 and sn < 1e-12: return k_sat
        sw=_s_Sw(hw,sn); sew=_s_Sew(sw)
        inner=1.0-sew**_inv_m
        if inner<0: inner=0.0
        if inner>1: inner=1.0
        kr=sew**g_l*(1.0-inner**m_vG)**2
        if kr<0: kr=0.0
        if kr>1: kr=1.0
        return k_sat*kr
    @njit(inline='always')
    def _s_kn(hw, sn):
        sw=_s_Sw(hw,sn); sew=_s_Sew(sw); st=_s_Set(sw,sn)
        se_n=st-sew
        if se_n<1e-14: se_n=1e-14
        tw=(1.0-sew**_inv_m)**m_vG
        if tw<0: tw=0.0
        if tw>1: tw=1.0
        tt=(1.0-st**_inv_m)**m_vG
        if tt<0: tt=0.0
        if tt>1: tt=1.0
        diff=tw-tt
        if diff<0: diff=0.0
        kr=se_n**g_l*diff**2
        if kr<0: kr=0.0
        if kr>1: kr=1.0
        return k_sat_n*kr
    @njit(inline='always')
    def _s_Hn(hw, sn, z_i):
        sw=_s_Sw(hw,sn); sew=_s_Sew(sw)
        base=sew**(-_inv_m)-1.0
        if base<0: base=0.0
        if base>1e12: base=1e12
        h_cnw=_inv_alpha_nw*base**_inv_n
        return _gamma_ratio*(hw+h_cnw)+z_i
    @njit(inline='always')
    def _s_Ctilde(hw):
        if hw >= 0.0: return C_l
        ah=abs(hw); ahn=(g_a*ah)**n_vG
        dSr=(S_sat-S_res)*m_vG*n_vG*g_a*(g_a*ah)**(n_vG-1.0)/(1.0+ahn)**(m_vG+1.0)
        ct=C_l+phi_0*dSr
        if ct<C_l: ct=C_l
        return ct

# ======================================================================
# PARTICLE CLASSIFICATION + IC + SOURCE (3D)
# ======================================================================
H_u, H_d = 8.0, 6.0
ptype = np.zeros(N_part, dtype=np.int32)
tol_bc = 0.5 * dx
for i in range(N_part):
    if   xp[i]<tol_bc:       ptype[i]=1
    elif xp[i]>Lx-tol_bc:    ptype[i]=2
    elif zp[i]<tol_bc:       ptype[i]=3  # bottom (z=0)
    elif zp[i]>Lz-tol_bc:    ptype[i]=4  # top (z=Lz)
    elif yp[i]<tol_bc:       ptype[i]=5  # front (y=0)
    elif yp[i]>Ly-tol_bc:    ptype[i]=6  # back  (y=Ly)

h_w = H_u + (H_d-H_u)*xp/Lx - zp
h_w[ptype==1] = H_u - zp[ptype==1]
h_w[ptype==2] = H_d - zp[ptype==2]

# NAPL source: cube at centre x, centre y, top z
SRC_X0,SRC_X1 = 4.5, 5.5
SRC_Y0,SRC_Y1 = 4.5, 5.5
SRC_Z0,SRC_Z1 = 8.5, 9.5
SN_SOURCE = 0.80
is_source = (xp>=SRC_X0)&(xp<=SRC_X1)&(yp>=SRC_Y0)&(yp<=SRC_Y1)&(zp>=SRC_Z0)&(zp<=SRC_Z1)
S_n = np.zeros(N_part)
S_n[is_source] = SN_SOURCE

_skip_dirichlet = (ptype==1)|(ptype==2)
_skip_napl      = _skip_dirichlet | is_source

# ======================================================================
# SPH OPERATORS (3D)
# ======================================================================
if HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _sph_dhdt_fused(h_w, S_n, zp_arr, skip_mask,
                        nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                        L_inv, K_norm, err_x, err_y, err_z, Vp_val, N):
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i=h_w[i]; sn_i=S_n[i]
                Hi=hw_i+zp_arr[i]; ki=_s_kw(hw_i,sn_i)
                raw_gx=0.0; raw_gy=0.0; raw_gz=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; Hj=h_w[j]+zp_arr[j]; dH=Hj-Hi
                    raw_gx+=Vp_val*dH*nbr_gWx[k]
                    raw_gy+=Vp_val*dH*nbr_gWy[k]
                    raw_gz+=Vp_val*dH*nbr_gWz[k]
                L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L02=L_inv[i,0,2]
                L10=L_inv[i,1,0]; L11=L_inv[i,1,1]; L12=L_inv[i,1,2]
                L20=L_inv[i,2,0]; L21=L_inv[i,2,1]; L22=L_inv[i,2,2]
                gHx=L00*raw_gx+L01*raw_gy+L02*raw_gz
                gHy=L10*raw_gx+L11*raw_gy+L12*raw_gz
                gHz=L20*raw_gx+L21*raw_gy+L22*raw_gz
                lap=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; kj=_s_kw(h_w[j],S_n[j])
                    Hj=h_w[j]+zp_arr[j]; km=0.5*(ki+kj)
                    lap+=Vp_val*km*(Hj-Hi)*nbr_Fhat[k]
                corr=ki*(gHx*err_x[i]+gHy*err_y[i]+gHz*err_z[i])
                Kn=K_norm[i]
                if abs(Kn)>=1e-30:
                    Ct=_s_Ctilde(hw_i)
                    out[i]=(2.0/Kn)*(lap-corr)/Ct
        return out

    @njit(cache=True, parallel=True)
    def _sph_dSndt_fused(h_w, S_n, zp_arr, skip_mask,
                         nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                         L_inv, K_norm, err_x, err_y, err_z, Vp_val, N, phi_val):
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i=h_w[i]; sn_i=S_n[i]
                Hi=_s_Hn(hw_i,sn_i,zp_arr[i]); ki=_s_kn(hw_i,sn_i)
                raw_gx=0.0; raw_gy=0.0; raw_gz=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; Hj=_s_Hn(h_w[j],S_n[j],zp_arr[j]); dH=Hj-Hi
                    raw_gx+=Vp_val*dH*nbr_gWx[k]
                    raw_gy+=Vp_val*dH*nbr_gWy[k]
                    raw_gz+=Vp_val*dH*nbr_gWz[k]
                L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L02=L_inv[i,0,2]
                L10=L_inv[i,1,0]; L11=L_inv[i,1,1]; L12=L_inv[i,1,2]
                L20=L_inv[i,2,0]; L21=L_inv[i,2,1]; L22=L_inv[i,2,2]
                gHx=L00*raw_gx+L01*raw_gy+L02*raw_gz
                gHy=L10*raw_gx+L11*raw_gy+L12*raw_gz
                gHz=L20*raw_gx+L21*raw_gy+L22*raw_gz
                lap=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; kj=_s_kn(h_w[j],S_n[j])
                    Hj=_s_Hn(h_w[j],S_n[j],zp_arr[j]); km=0.5*(ki+kj)
                    lap+=Vp_val*km*(Hj-Hi)*nbr_Fhat[k]
                corr=ki*(gHx*err_x[i]+gHy*err_y[i]+gHz*err_z[i])
                Kn=K_norm[i]
                if abs(Kn)>=1e-30:
                    out[i]=(2.0/Kn)*(lap-corr)/phi_val
        return out

else:
    # Pure Python fallback — 3D with CSR arrays
    def _sph_div_k_gradH_py(H_f, k_h, C_store, skip_mask,
                             nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy, nbr_gWz,
                             L_inv, K_norm, err_x, err_y, err_z, Vp_val, N):
        out = np.zeros(N)
        for i in range(N):
            if skip_mask[i]: continue
            Hi=H_f[i]; ki=k_h[i]
            raw_gx=0.0; raw_gy=0.0; raw_gz=0.0
            for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                j=nbr_j[k]; dH=H_f[j]-Hi
                raw_gx+=Vp_val*dH*nbr_gWx[k]
                raw_gy+=Vp_val*dH*nbr_gWy[k]
                raw_gz+=Vp_val*dH*nbr_gWz[k]
            L00=L_inv[i,0,0]; L01=L_inv[i,0,1]; L02=L_inv[i,0,2]
            L10=L_inv[i,1,0]; L11=L_inv[i,1,1]; L12=L_inv[i,1,2]
            L20=L_inv[i,2,0]; L21=L_inv[i,2,1]; L22=L_inv[i,2,2]
            gHx=L00*raw_gx+L01*raw_gy+L02*raw_gz
            gHy=L10*raw_gx+L11*raw_gy+L12*raw_gz
            gHz=L20*raw_gx+L21*raw_gy+L22*raw_gz
            lap=0.0
            for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                j=nbr_j[k]; km=0.5*(ki+k_h[j])
                lap+=Vp_val*km*(H_f[j]-Hi)*nbr_Fhat[k]
            corr=ki*(gHx*err_x[i]+gHy*err_y[i]+gHz*err_z[i])
            Kn=K_norm[i]
            if abs(Kn)<1e-30: continue
            out[i]=(2.0/Kn)*(lap-corr)/C_store[i]
        return out


# Pre-allocate field arrays (used by NumPy fallback path)
_fld_kw = np.empty(N_part); _fld_Hw = np.empty(N_part)
_fld_Ct = np.empty(N_part); _fld_kn = np.empty(N_part)
_fld_Hn = np.empty(N_part); _fld_Cn = np.full(N_part, phi_0)

def _precompute_step(hw, Sn):
    """NumPy fallback precompute."""
    _fld_kw[:]=compute_kw_3ph(hw,Sn); _fld_Hw[:]=hw+zp
    Cs=compute_Cs(hw); np.maximum(C_l+Cs,C_l,out=_fld_Ct)
    _fld_kn[:]=compute_kn_field(hw,Sn); _fld_Hn[:]=compute_Hn_field(hw,Sn)

def compute_dhdt(hw, Sn):
    if HAS_NUMBA:
        return _sph_dhdt_fused(hw, Sn, zp, _skip_dirichlet,
                                nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)
    else:
        _precompute_step(hw, Sn)
        return _sph_div_k_gradH_py(_fld_Hw,_fld_kw,_fld_Ct,_skip_dirichlet,
                                    nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                    L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)

def compute_dSndt(hw, Sn):
    if HAS_NUMBA:
        return _sph_dSndt_fused(hw, Sn, zp, _skip_napl,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                 L_inv,K_norm,err_x,err_y,err_z,Vp,N_part,phi_0)
    else:
        return _sph_div_k_gradH_py(_fld_Hn,_fld_kn,_fld_Cn,_skip_napl,
                                    nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                    L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)

def stable_dt(hw, Sn):
    Cs=compute_Cs(hw); Ct=C_l+Cs; C_min=max(np.min(Ct),C_l)
    dt_w=CFL*C_min*h_sml**2/k_sat; dt_n=CFL*phi_0*h_sml**2/k_sat_n
    return min(dt_w,dt_n)

# BC enforcement (vectorised with precomputed indices, 3D)
_idx_left  = np.where(ptype==1)[0]
_idx_right = np.where(ptype==2)[0]
_idx_bot   = np.where(ptype==3)[0]
_idx_top   = np.where(ptype==4)[0]
_idx_front = np.where(ptype==5)[0]
_idx_back  = np.where(ptype==6)[0]
_idx_src   = np.where(is_source)[0]

def _mirror_idx(ptype_val, jk_fn):
    out = np.empty(np.sum(ptype==ptype_val), dtype=np.int64)
    src = np.where(ptype==ptype_val)[0]
    for n, i in enumerate(src):
        ix, jy, kz = jk_fn(i)
        ix = min(ix, Nx-1); jy = min(jy, Ny-1); kz = min(kz, Nz-1)
        out[n] = idx_3d[ix, jy, kz]
    return out

_mirror_bot   = _mirror_idx(3, lambda i: (int(round(xp[i]/dx)), int(round(yp[i]/dy)), 1))
_mirror_top   = _mirror_idx(4, lambda i: (int(round(xp[i]/dx)), int(round(yp[i]/dy)), Nz-2))
_mirror_front = _mirror_idx(5, lambda i: (int(round(xp[i]/dx)), 1,                     int(round(zp[i]/dz))))
_mirror_back  = _mirror_idx(6, lambda i: (int(round(xp[i]/dx)), Ny-2,                  int(round(zp[i]/dz))))

_dz_bot   = zp[_mirror_bot]   - zp[_idx_bot]
_dz_top   = zp[_mirror_top]   - zp[_idx_top]
_dz_front = zp[_mirror_front] - zp[_idx_front]
_dz_back  = zp[_mirror_back]  - zp[_idx_back]
_h_dir_left  = H_u - zp[_idx_left]
_h_dir_right = H_d - zp[_idx_right]

# Source-conflict override (see sph_seepage_napl.py for rationale)
_idx_bot_srcconf   = _idx_bot[is_source[_mirror_bot]]
_idx_top_srcconf   = _idx_top[is_source[_mirror_top]]
_idx_front_srcconf = _idx_front[is_source[_mirror_front]]
_idx_back_srcconf  = _idx_back[is_source[_mirror_back]]

def enforce_bcs(hw, Sn):
    hw[_idx_left]  = _h_dir_left
    hw[_idx_right] = _h_dir_right
    hw[_idx_bot]   = hw[_mirror_bot]   + _dz_bot
    hw[_idx_top]   = hw[_mirror_top]   + _dz_top
    hw[_idx_front] = hw[_mirror_front] + _dz_front
    hw[_idx_back]  = hw[_mirror_back]  + _dz_back
    Sn[_idx_left]  = 0.0; Sn[_idx_right] = 0.0
    Sn[_idx_bot]   = Sn[_mirror_bot]
    Sn[_idx_top]   = Sn[_mirror_top]
    Sn[_idx_front] = Sn[_mirror_front]
    Sn[_idx_back]  = Sn[_mirror_back]
    # Source-conflict override: prevent source leaking through impermeable walls at coarse grids
    Sn[_idx_bot_srcconf]   = 0.0
    Sn[_idx_top_srcconf]   = 0.0
    Sn[_idx_front_srcconf] = 0.0
    Sn[_idx_back_srcconf]  = 0.0
    Sn[_idx_src]   = SN_SOURCE

# ======================================================================
# JIT WARMUP
# ======================================================================
print("JIT warmup ...", end=" ", flush=True)
t_jit0 = wall_time.perf_counter()
_ = compute_dhdt(h_w, S_n)
_ = compute_dSndt(h_w, S_n)
t_jit1 = wall_time.perf_counter()
t_jit = t_jit1 - t_jit0
print(f"{t_jit:.2f} s")

# ======================================================================
# TIMED MAIN LOOP
# ======================================================================
mask_conv    = (ptype==0)|(ptype>=3)
mask_not_src = ~is_source & mask_conv

t_constitutive = 0.0
t_sph_kernel   = 0.0
t_update_bc    = 0.0
t_metrics      = 0.0

time_log=[]; l2w_log=[]; l2n_log=[]; snmax_log=[]
t_phys = 0.0

print(f"\n{'='*70}")
print(f"  BENCHMARK: Nx=Ny=Nz={Nx}, {N_steps} steps, {mode_str}")
print(f"{'='*70}")

t_loop0 = wall_time.perf_counter()

for step in range(1, N_steps + 1):
    dt = stable_dt(h_w, S_n)

    if HAS_NUMBA:
        tc0 = wall_time.perf_counter()
        tc1 = wall_time.perf_counter()
        t_constitutive += tc1 - tc0

        tk0 = wall_time.perf_counter()
        dhdt  = _sph_dhdt_fused(h_w, S_n, zp, _skip_dirichlet,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                 L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)
        dSndt = _sph_dSndt_fused(h_w, S_n, zp, _skip_napl,
                                  nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                  L_inv,K_norm,err_x,err_y,err_z,Vp,N_part,phi_0)
        tk1 = wall_time.perf_counter()
        t_sph_kernel += tk1 - tk0
    else:
        tc0 = wall_time.perf_counter()
        _precompute_step(h_w, S_n)
        tc1 = wall_time.perf_counter()
        t_constitutive += tc1 - tc0

        tk0 = wall_time.perf_counter()
        dhdt  = _sph_div_k_gradH_py(_fld_Hw,_fld_kw,_fld_Ct,_skip_dirichlet,
                                      nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                      L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)
        dSndt = _sph_div_k_gradH_py(_fld_Hn,_fld_kn,_fld_Cn,_skip_napl,
                                      nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,nbr_gWz,
                                      L_inv,K_norm,err_x,err_y,err_z,Vp,N_part)
        tk1 = wall_time.perf_counter()
        t_sph_kernel += tk1 - tk0

    tu0 = wall_time.perf_counter()
    h_w += dt * dhdt
    S_n += dt * dSndt
    S_n = np.clip(S_n, 0.0, 1.0 - S_res)
    enforce_bcs(h_w, S_n)
    tu1 = wall_time.perf_counter()
    t_update_bc += tu1 - tu0

    t_phys += dt

    tm0 = wall_time.perf_counter()
    l2w = np.sqrt(np.mean(dhdt[mask_conv]**2))
    l2n = np.sqrt(np.mean(dSndt[mask_not_src]**2)) if np.any(mask_not_src) else 0.0
    snm = np.max(S_n[~is_source]) if np.any(~is_source) else 0.0
    time_log.append(t_phys); l2w_log.append(l2w); l2n_log.append(l2n); snmax_log.append(snm)
    tm1 = wall_time.perf_counter()
    t_metrics += tm1 - tm0

    if step % max(1, N_steps // 5) == 0 or step == 1:
        print(f"  step {step:5d}  t={t_phys:.4e}  L2w={l2w:.4e}  L2n={l2n:.4e}  Sn_max={snm:.4f}")

t_loop1 = wall_time.perf_counter()
t_loop = t_loop1 - t_loop0
t_total = t_setup + t_jit + t_loop

# ======================================================================
# STATISTICS
# ======================================================================
t_sum = t_constitutive + t_sph_kernel + t_update_bc + t_metrics
if t_sum < 1e-9: t_sum = 1e-9
ms_per_step = t_loop / N_steps * 1000
particles_per_sec = N_part * N_steps / max(t_loop, 1e-9)
q_expected = k_sat * (H_u - H_d) / Lx

stats_lines = [
    f"SPH Three-Phase 3D Benchmark Results",
    f"{'='*50}",
    f"Mode              : {mode_str}",
    f"Grid              : {Nx} x {Ny} x {Nz}  ({N_part} particles)",
    f"Steps             : {N_steps}",
    f"dx                : {dx:.4f} m",
    f"Avg nbrs/particle : {_total/N_part:.1f}",
    f"",
    f"TIMING",
    f"{'-'*50}",
    f"Setup (nbrs+corr) : {t_setup:8.3f} s",
    f"JIT warmup         : {t_jit:8.3f} s",
    f"Main loop          : {t_loop:8.3f} s  ({ms_per_step:.2f} ms/step)",
    f"  Constitutive     : {t_constitutive:8.3f} s  ({t_constitutive/t_sum*100:5.1f}%)",
    f"  SPH kernels      : {t_sph_kernel:8.3f} s  ({t_sph_kernel/t_sum*100:5.1f}%)",
    f"  Update + BCs     : {t_update_bc:8.3f} s  ({t_update_bc/t_sum*100:5.1f}%)",
    f"  Metrics          : {t_metrics:8.3f} s  ({t_metrics/t_sum*100:5.1f}%)",
    f"Total              : {t_total:8.3f} s",
    f"",
    f"THROUGHPUT",
    f"{'-'*50}",
    f"ms/step            : {ms_per_step:.3f}",
    f"particles*steps/s  : {particles_per_sec:.0f}",
    f"",
    f"PHYSICS CHECK",
    f"{'-'*50}",
    f"Final L2(dh_w/dt)  : {l2w_log[-1]:.6e}",
    f"Final L2(dS_n/dt)  : {l2n_log[-1]:.6e}",
    f"Max Sn (outside)   : {snmax_log[-1]:.6f}",
    f"Physical time      : {t_phys:.4e} s",
    f"Expected q_w       : {q_expected:.6e} m/s",
]

print(f"\n{'='*50}")
for line in stats_lines: print(line)
print(f"{'='*50}")

tag = f"Nx{Nx}_Tt{n_threads}"
if jit_disabled: tag = f"Nx{Nx}_PurePy"
stats_path = os.path.join(OUT_DIR, f"benchmark_{tag}.txt")
with open(stats_path, "w") as f:
    for line in stats_lines: f.write(line + "\n")
    f.write(f"\n# MACHINE_READABLE\n")
    f.write(f"NX={Nx}\nNPART={N_part}\nNSTEPS={N_steps}\nTHREADS={n_threads}\n")
    f.write(f"JIT_DISABLED={int(jit_disabled)}\n")
    f.write(f"T_SETUP={t_setup:.6f}\nT_JIT={t_jit:.6f}\nT_LOOP={t_loop:.6f}\n")
    f.write(f"T_CONSTITUTIVE={t_constitutive:.6f}\nT_SPH_KERNEL={t_sph_kernel:.6f}\n")
    f.write(f"T_UPDATE_BC={t_update_bc:.6f}\nT_METRICS={t_metrics:.6f}\n")
    f.write(f"MS_PER_STEP={ms_per_step:.6f}\n")
    f.write(f"L2W_FINAL={l2w_log[-1]:.10e}\nL2N_FINAL={l2n_log[-1]:.10e}\n")
    f.write(f"SNMAX_FINAL={snmax_log[-1]:.10e}\n")
print(f"\nStats saved: {stats_path}")

# ======================================================================
# DIAGNOSTIC PLOT (3D — 2D slices at NAPL source centre)
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Sn_3 = S_n.reshape(Nx, Ny, Nz)
h_3  = h_w.reshape(Nx, Ny, Nz)
X3 = xp.reshape(Nx, Ny, Nz); Y3 = yp.reshape(Nx, Ny, Nz); Z3 = zp.reshape(Nx, Ny, Nz)

ix_src = max(0, min(int(round(0.5*(SRC_X0+SRC_X1)/dx)), Nx-1))
iy_src = max(0, min(int(round(0.5*(SRC_Y0+SRC_Y1)/dy)), Ny-1))
y_axis = Y3[0, :, 0]; z_axis = Z3[0, 0, :]; x_axis = X3[:, 0, 0]
YY_yz, ZZ_yz = np.meshgrid(y_axis, z_axis, indexing="ij")
XX_xz, ZZ_xz = np.meshgrid(x_axis, z_axis, indexing="ij")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

axes[0].semilogy(time_log, l2w_log, "b-", lw=1, label=r"L2($dh_w/dt$)")
axes[0].semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("Residual")
axes[0].set_title("Convergence"); axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

levels_sn = np.linspace(0, SN_SOURCE, 21)
Sn_yz = Sn_3[ix_src, :, :]; h_yz = h_3[ix_src, :, :]
cf = axes[1].contourf(YY_yz, ZZ_yz, Sn_yz, levels=levels_sn, cmap="YlOrRd", extend="max")
fig.colorbar(cf, ax=axes[1], shrink=0.85, label=r"$S_n$")
axes[1].contour(YY_yz, ZZ_yz, h_yz, levels=[0.0], colors="blue", linewidths=1.5, linestyles="--")
axes[1].set_title(f"YZ @ x={x_axis[ix_src]:.2f}m, t={t_phys:.1f}s")
axes[1].set_xlabel("y [m]"); axes[1].set_ylabel("z [m]"); axes[1].set_aspect("equal")

Sn_xz = Sn_3[:, iy_src, :]; h_xz = h_3[:, iy_src, :]
cf = axes[2].contourf(XX_xz, ZZ_xz, Sn_xz, levels=levels_sn, cmap="YlOrRd", extend="max")
fig.colorbar(cf, ax=axes[2], shrink=0.85, label=r"$S_n$")
axes[2].contour(XX_xz, ZZ_xz, h_xz, levels=[0.0], colors="blue", linewidths=1.5, linestyles="--")
axes[2].set_title(f"XZ @ y={y_axis[iy_src]:.2f}m, t={t_phys:.1f}s")
axes[2].set_xlabel("x [m]"); axes[2].set_ylabel("z [m]"); axes[2].set_aspect("equal")

fig.suptitle(f"Benchmark 3D: {Nx}³, {mode_str}", fontsize=12, y=1.02)
fig_path = os.path.join(OUT_DIR, f"benchmark_{tag}.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved : {fig_path}")
print("Done.")
