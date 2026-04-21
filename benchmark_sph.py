#!/usr/bin/env python3
"""
SPH Three-Phase Seepage — Performance Benchmark
=================================================

Self-contained benchmark that replicates the full physics of
sph_seepage_napl.py but focuses on timing.  No I/O, no checkpoints,
no animation — just raw computation + timing breakdown.

Usage
-----
    # Single-thread Numba JIT
    NUMBA_NUM_THREADS=1 python benchmark_sph.py --nx 101 --nsteps 50

    # 8-thread parallel
    NUMBA_NUM_THREADS=8 python benchmark_sph.py --nx 201 --nsteps 50

    # Pure Python (disable JIT entirely)
    NUMBA_DISABLE_JIT=1 python benchmark_sph.py --nx 51 --nsteps 20

PyCharm: set --nx and --nsteps in "Script parameters" and
NUMBA_NUM_THREADS / NUMBA_DISABLE_JIT in "Environment variables".

Output
------
    benchmark_NxNN_TtNN.txt   — timing statistics
    benchmark_NxNN_TtNN.png   — diagnostic plot (convergence + final Sn)
"""

import argparse
import os
import time as wall_time
import numpy as np

# ── Parse arguments BEFORE importing numba (env vars must be set first) ─
parser = argparse.ArgumentParser(description="SPH three-phase benchmark")
parser.add_argument("--nx", type=int, default=51,
                    help="Grid size Nx = Ny (default: 51)")
parser.add_argument("--nsteps", type=int, default=50,
                    help="Number of time steps (default: 50)")
parser.add_argument("--outdir", type=str, default=".",
                    help="Output directory (default: cwd)")
args = parser.parse_args()

Nx_arg   = args.nx
N_steps  = args.nsteps
OUT_DIR  = args.outdir
os.makedirs(OUT_DIR, exist_ok=True)

n_threads = int(os.environ.get("NUMBA_NUM_THREADS", "1"))
jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"

# ── Numba import ─────────────────────────────────────────────────────────
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    mode_str = f"Numba JIT, {n_threads} thread(s)"
    if jit_disabled:
        mode_str = "Numba DISABLED (pure Python via NUMBA_DISABLE_JIT)"
except ImportError:
    HAS_NUMBA = False
    mode_str = "Pure Python (no Numba)"

print(f"Mode           : {mode_str}")
print(f"Grid           : {Nx_arg} x {Nx_arg}")
print(f"Steps          : {N_steps}")

# ======================================================================
# PHYSICAL PARAMETERS  (identical to sph_seepage_napl.py)
# ======================================================================
k_abs   = 2.059e-11
phi_0   = 0.43
c_R     = 4.35e-7
rho_W   = 1000.0
mu_w    = 1.0e-3
g_acc   = 9.81
n_vG    = 2.68
p_caw0  = 676.55

gamma_w = rho_W * g_acc
k_sat   = k_abs * rho_W * g_acc / mu_w
m_vG    = 1.0 - 1.0 / n_vG
g_a     = gamma_w / p_caw0

S_sat   = 1.0
S_res   = 0.045
S_wir   = S_res
g_l     = 0.5

K_s       = 1.0 / c_R
K_sat_l   = 2.0e9
alpha_hat = (1.0 - phi_0) / K_s
beta_hat  = 1.0 / K_sat_l
C_l       = gamma_w * (phi_0 * S_sat * beta_hat + alpha_hat)

rho_N    = 830.0
mu_n     = 3.61e-3
gamma_n  = rho_N * g_acc
k_sat_n  = k_abs * rho_N * g_acc / mu_n

sigma_aw = 0.065
sigma_nw = 0.030
beta_nw  = sigma_aw / sigma_nw
alpha_nw = beta_nw * g_a

CFL = 0.25

# ======================================================================
# DOMAIN  (parameterised by Nx_arg)
# ======================================================================
Lx, Ly = 10.0, 10.0
Nx = Nx_arg
Ny = Nx_arg
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

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
Vp     = dx * dy
h_sml  = 1.3 * dx

print(f"Particles      : {N_part}")
print(f"dx = dy        : {dx:.4f} m")

# ======================================================================
# KERNEL
# ======================================================================
alpha_d = 15.0 / (7.0 * np.pi * h_sml**2)

def dW_dr(r):
    q = r / h_sml
    if q < 1e-14 or q >= 2.0:
        return 0.0
    if q < 1.0:
        return alpha_d * (-2.0*q + 1.5*q*q) / h_sml
    return alpha_d * (-0.5) * (2.0 - q)**2 / h_sml

# ======================================================================
# NEIGHBOURS  (cKDTree + CSR)
# ======================================================================
support_r = 2.0 * h_sml

print("Building neighbours ...", end=" ", flush=True)
t_setup0 = wall_time.perf_counter()

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([xp, yp]))
nbl  = tree.query_ball_tree(tree, r=support_r)

# Temporary Python list (also needed for correction matrices)
_neighbours = [[] for _ in range(N_part)]
for i in range(N_part):
    xi, yi = xp[i], yp[i]
    for j in nbl[i]:
        if j == i: continue
        xji = xp[j] - xi; yji = yp[j] - yi
        r = np.sqrt(xji**2 + yji**2)
        if r < 1e-30: continue
        dWr = dW_dr(r)
        gWx = dWr * (-xji) / r
        gWy = dWr * (-yji) / r
        dot  = xji * gWx + yji * gWy
        Fhat = dot / (r * r)
        _neighbours[i].append((j, r, xji, yji, Fhat, gWx, gWy))

# CSR conversion
_total = sum(len(nb) for nb in _neighbours)
nbr_j    = np.empty(_total, dtype=np.int64)
nbr_Fhat = np.empty(_total, dtype=np.float64)
nbr_gWx  = np.empty(_total, dtype=np.float64)
nbr_gWy  = np.empty(_total, dtype=np.float64)
nbr_ptr  = np.empty(N_part + 1, dtype=np.int64)

off = 0
for i in range(N_part):
    nbr_ptr[i] = off
    for (j, rij, xji, yji, Fhat, gWx, gWy) in _neighbours[i]:
        nbr_j[off]=j; nbr_Fhat[off]=Fhat; nbr_gWx[off]=gWx; nbr_gWy[off]=gWy
        off += 1
nbr_ptr[N_part] = off
print(f"CSR {_total} entries.", flush=True)

# ======================================================================
# CORRECTION MATRICES
# ======================================================================
print("Correction matrices ...", end=" ", flush=True)
K_norm = np.zeros(N_part)
err_x  = np.zeros(N_part)
err_y  = np.zeros(N_part)
L_inv  = np.zeros((N_part, 2, 2))

for i in range(N_part):
    Kn=0.0; ex=0.0; ey=0.0; M=np.zeros((2,2))
    for (j,rij,xji,yji,Fhat,gWx,gWy) in _neighbours[i]:
        Kn += Vp*(xji**2+yji**2)*Fhat
        ex += Vp*xji*Fhat; ey += Vp*yji*Fhat
        M[0,0]+=Vp*xji*gWx; M[0,1]+=Vp*xji*gWy
        M[1,0]+=Vp*yji*gWx; M[1,1]+=Vp*yji*gWy
    K_norm[i]=0.5*Kn; err_x[i]=ex; err_y[i]=ey
    det = M[0,0]*M[1,1]-M[0,1]*M[1,0]
    if abs(det) > 1e-30:
        L_inv[i,0,0]=M[1,1]/det; L_inv[i,0,1]=-M[0,1]/det
        L_inv[i,1,0]=-M[1,0]/det; L_inv[i,1,1]=M[0,0]/det
    else:
        L_inv[i] = np.eye(2)

del _neighbours
t_setup1 = wall_time.perf_counter()
t_setup = t_setup1 - t_setup0
print(f"done.  Setup: {t_setup:.2f} s", flush=True)

# ======================================================================
# CONSTITUTIVE FUNCTIONS  (NumPy vectorised)
# ======================================================================
def compute_Sr(h):
    return np.where(h>=0, S_sat, S_res+(S_sat-S_res)*(1+(g_a*np.abs(h))**n_vG)**(-m_vG))

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
    return (gamma_w/gamma_n)*(h_w+h_cnw)+yp

# Precomputed constants for scalar helpers
_gamma_ratio = gamma_w / gamma_n
_inv_alpha_nw = 1.0 / alpha_nw
_inv_m = 1.0 / m_vG
_inv_n = 1.0 / n_vG

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
    def _s_Hn(hw, sn, yp_i):
        sw=_s_Sw(hw,sn); sew=_s_Sew(sw)
        base=sew**(-_inv_m)-1.0
        if base<0: base=0.0
        if base>1e12: base=1e12
        h_cnw=_inv_alpha_nw*base**_inv_n
        return _gamma_ratio*(hw+h_cnw)+yp_i
    @njit(inline='always')
    def _s_Ctilde(hw):
        if hw >= 0.0: return C_l
        ah=abs(hw); ahn=(g_a*ah)**n_vG
        dSr=(S_sat-S_res)*m_vG*n_vG*g_a*(g_a*ah)**(n_vG-1.0)/(1.0+ahn)**(m_vG+1.0)
        ct=C_l+phi_0*dSr
        if ct<C_l: ct=C_l
        return ct
    @njit(cache=True, parallel=True)
    def _precompute_fields(h_w, S_n, yp_arr, N, kw_o, Hw_o, Ct_o, kn_o, Hn_o):
        for i in prange(N):
            hw_i=h_w[i]; sn_i=S_n[i]
            kw_o[i]=_s_kw(hw_i,sn_i); Hw_o[i]=hw_i+yp_arr[i]
            Ct_o[i]=_s_Ctilde(hw_i)
            kn_o[i]=_s_kn(hw_i,sn_i); Hn_o[i]=_s_Hn(hw_i,sn_i,yp_arr[i])

# ======================================================================
# PARTICLE CLASSIFICATION + INITIAL CONDITIONS
# ======================================================================
H_u, H_d = 8.0, 6.0
ptype = np.zeros(N_part, dtype=np.int32)
tol_bc = 0.5 * dx
for i in range(N_part):
    if   xp[i]<tol_bc:       ptype[i]=1
    elif xp[i]>Lx-tol_bc:    ptype[i]=2
    elif yp[i]<tol_bc:       ptype[i]=3
    elif yp[i]>Ly-tol_bc:    ptype[i]=4

h_w = H_u + (H_d-H_u)*xp/Lx - yp
h_w[ptype==1] = H_u - yp[ptype==1]
h_w[ptype==2] = H_d - yp[ptype==2]
h_init = h_w.copy()

# NAPL source (scales with domain — fixed physical location)
SRC_X0,SRC_X1 = 4.5, 5.5
SRC_Y0,SRC_Y1 = 8.5, 9.5
SN_SOURCE = 0.80
is_source = (xp>=SRC_X0)&(xp<=SRC_X1)&(yp>=SRC_Y0)&(yp<=SRC_Y1)
S_n = np.zeros(N_part)
S_n[is_source] = SN_SOURCE

_skip_dirichlet = (ptype==1)|(ptype==2)
_skip_napl      = _skip_dirichlet | is_source

# ======================================================================
# SPH OPERATORS  (Numba with prange or Python fallback)
# ======================================================================
if HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _sph_div_k_gradH(H_f, k_h, C_store, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                          L_inv, K_norm, err_x, err_y, Vp_val, N):
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                Hi=H_f[i]; ki=k_h[i]
                raw_gx=0.0; raw_gy=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; dH=H_f[j]-Hi
                    raw_gx+=Vp_val*dH*nbr_gWx[k]; raw_gy+=Vp_val*dH*nbr_gWy[k]
                L00=L_inv[i,0,0];L01=L_inv[i,0,1];L10=L_inv[i,1,0];L11=L_inv[i,1,1]
                gHx=L00*raw_gx+L01*raw_gy; gHy=L10*raw_gx+L11*raw_gy
                lap=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; km=0.5*(ki+k_h[j])
                    lap+=Vp_val*km*(H_f[j]-Hi)*nbr_Fhat[k]
                corr=ki*(gHx*err_x[i]+gHy*err_y[i])
                Kn=K_norm[i]
                if abs(Kn)>=1e-30:
                    out[i]=(2.0/Kn)*(lap-corr)/C_store[i]
        return out

    # ── FULLY FUSED KERNELS ─────────────────────────────────────────
    @njit(cache=True, parallel=True)
    def _sph_dhdt_fused(h_w, S_n, yp_arr, skip_mask,
                        nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                        L_inv, K_norm, err_x, err_y, Vp_val, N):
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i=h_w[i]; sn_i=S_n[i]
                Hi=hw_i+yp_arr[i]; ki=_s_kw(hw_i,sn_i)
                raw_gx=0.0; raw_gy=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; Hj=h_w[j]+yp_arr[j]; dH=Hj-Hi
                    raw_gx+=Vp_val*dH*nbr_gWx[k]; raw_gy+=Vp_val*dH*nbr_gWy[k]
                L00=L_inv[i,0,0];L01=L_inv[i,0,1];L10=L_inv[i,1,0];L11=L_inv[i,1,1]
                gHx=L00*raw_gx+L01*raw_gy; gHy=L10*raw_gx+L11*raw_gy
                lap=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; kj=_s_kw(h_w[j],S_n[j])
                    Hj=h_w[j]+yp_arr[j]; km=0.5*(ki+kj)
                    lap+=Vp_val*km*(Hj-Hi)*nbr_Fhat[k]
                corr=ki*(gHx*err_x[i]+gHy*err_y[i])
                Kn=K_norm[i]
                if abs(Kn)>=1e-30:
                    Ct=_s_Ctilde(hw_i)
                    out[i]=(2.0/Kn)*(lap-corr)/Ct
        return out

    @njit(cache=True, parallel=True)
    def _sph_dSndt_fused(h_w, S_n, yp_arr, skip_mask,
                         nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                         L_inv, K_norm, err_x, err_y, Vp_val, N, phi_val):
        out = np.zeros(N)
        for i in prange(N):
            if not skip_mask[i]:
                hw_i=h_w[i]; sn_i=S_n[i]
                Hi=_s_Hn(hw_i,sn_i,yp_arr[i]); ki=_s_kn(hw_i,sn_i)
                raw_gx=0.0; raw_gy=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; Hj=_s_Hn(h_w[j],S_n[j],yp_arr[j]); dH=Hj-Hi
                    raw_gx+=Vp_val*dH*nbr_gWx[k]; raw_gy+=Vp_val*dH*nbr_gWy[k]
                L00=L_inv[i,0,0];L01=L_inv[i,0,1];L10=L_inv[i,1,0];L11=L_inv[i,1,1]
                gHx=L00*raw_gx+L01*raw_gy; gHy=L10*raw_gx+L11*raw_gy
                lap=0.0
                for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                    j=nbr_j[k]; kj=_s_kn(h_w[j],S_n[j])
                    Hj=_s_Hn(h_w[j],S_n[j],yp_arr[j]); km=0.5*(ki+kj)
                    lap+=Vp_val*km*(Hj-Hi)*nbr_Fhat[k]
                corr=ki*(gHx*err_x[i]+gHy*err_y[i])
                Kn=K_norm[i]
                if abs(Kn)>=1e-30:
                    out[i]=(2.0/Kn)*(lap-corr)/phi_val
        return out
else:
    def _sph_div_k_gradH(H_f, k_h, C_store, skip_mask,
                          nbr_ptr, nbr_j, nbr_Fhat, nbr_gWx, nbr_gWy,
                          L_inv, K_norm, err_x, err_y, Vp_val, N):
        out = np.zeros(N)
        for i in range(N):
            if skip_mask[i]: continue
            Hi=H_f[i]; ki=k_h[i]
            raw_gx=0.0; raw_gy=0.0
            for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                j=nbr_j[k]; dH=H_f[j]-Hi
                raw_gx+=Vp_val*dH*nbr_gWx[k]; raw_gy+=Vp_val*dH*nbr_gWy[k]
            L00=L_inv[i,0,0];L01=L_inv[i,0,1];L10=L_inv[i,1,0];L11=L_inv[i,1,1]
            gHx=L00*raw_gx+L01*raw_gy; gHy=L10*raw_gx+L11*raw_gy
            lap=0.0
            for k in range(nbr_ptr[i],nbr_ptr[i+1]):
                j=nbr_j[k]; km=0.5*(ki+k_h[j])
                lap+=Vp_val*km*(H_f[j]-Hi)*nbr_Fhat[k]
            corr=ki*(gHx*err_x[i]+gHy*err_y[i])
            Kn=K_norm[i]
            if abs(Kn)<1e-30: continue
            out[i]=(2.0/Kn)*(lap-corr)/C_store[i]
        return out


# Pre-allocate field arrays
_fld_kw = np.empty(N_part); _fld_Hw = np.empty(N_part)
_fld_Ct = np.empty(N_part); _fld_kn = np.empty(N_part)
_fld_Hn = np.empty(N_part); _fld_Cn = np.full(N_part, phi_0)

def _precompute_step(hw, Sn):
    if HAS_NUMBA:
        _precompute_fields(hw, Sn, yp, N_part, _fld_kw, _fld_Hw, _fld_Ct, _fld_kn, _fld_Hn)
    else:
        _fld_kw[:]=compute_kw_3ph(hw,Sn); _fld_Hw[:]=hw+yp
        Cs=compute_Cs(hw); np.maximum(C_l+Cs,C_l,out=_fld_Ct)
        _fld_kn[:]=compute_kn_field(hw,Sn); _fld_Hn[:]=compute_Hn_field(hw,Sn)

def compute_dhdt(hw, Sn):
    if HAS_NUMBA:
        return _sph_dhdt_fused(hw, Sn, yp, _skip_dirichlet,
                                nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                L_inv,K_norm,err_x,err_y,Vp,N_part)
    else:
        _precompute_step(hw, Sn)
        return _sph_div_k_gradH(_fld_Hw,_fld_kw,_fld_Ct,_skip_dirichlet,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                 L_inv,K_norm,err_x,err_y,Vp,N_part)

def compute_dSndt(hw, Sn):
    if HAS_NUMBA:
        return _sph_dSndt_fused(hw, Sn, yp, _skip_napl,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                 L_inv,K_norm,err_x,err_y,Vp,N_part,phi_0)
    else:
        return _sph_div_k_gradH(_fld_Hn,_fld_kn,_fld_Cn,_skip_napl,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                 L_inv,K_norm,err_x,err_y,Vp,N_part)

def stable_dt(hw, Sn):
    Cs=compute_Cs(hw); Ct=C_l+Cs; C_min=max(np.min(Ct),C_l)
    dt_w=CFL*C_min*h_sml**2/k_sat; dt_n=CFL*phi_0*h_sml**2/k_sat_n
    return min(dt_w,dt_n)

# Precompute BC index arrays (once)
_idx_left  = np.where(ptype==1)[0]
_idx_right = np.where(ptype==2)[0]
_idx_bot   = np.where(ptype==3)[0]
_idx_top   = np.where(ptype==4)[0]
_idx_src   = np.where(is_source)[0]
_mirror_bot = np.array([idx_2d[min(int(round(xp[i]/dx)),Nx-1),1] for i in _idx_bot], dtype=np.int64)
_mirror_top = np.array([idx_2d[min(int(round(xp[i]/dx)),Nx-1),Ny-2] for i in _idx_top], dtype=np.int64)
_dz_bot = yp[_mirror_bot] - yp[_idx_bot]
_dz_top = yp[_mirror_top] - yp[_idx_top]
_h_dir_left  = H_u - yp[_idx_left]
_h_dir_right = H_d - yp[_idx_right]

def enforce_bcs(hw, Sn):
    hw[_idx_left]  = _h_dir_left
    hw[_idx_right] = _h_dir_right
    hw[_idx_bot]   = hw[_mirror_bot] + _dz_bot
    hw[_idx_top]   = hw[_mirror_top] + _dz_top
    Sn[_idx_left]  = 0.0
    Sn[_idx_right] = 0.0
    Sn[_idx_bot]   = Sn[_mirror_bot]
    Sn[_idx_top]   = Sn[_mirror_top]
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
mask_conv    = (ptype==0)|(ptype==3)|(ptype==4)
mask_not_src = ~is_source & mask_conv

t_constitutive = 0.0   # NumPy vectorised calls
t_sph_kernel   = 0.0   # Numba/Python particle loops
t_update_bc    = 0.0   # field update + BCs
t_metrics      = 0.0   # convergence metrics

time_log=[]; l2w_log=[]; l2n_log=[]; snmax_log=[]
t_phys = 0.0

print(f"\n{'='*70}")
print(f"  BENCHMARK: Nx={Nx}, {N_steps} steps, {mode_str}")
print(f"{'='*70}")

t_loop0 = wall_time.perf_counter()

for step in range(1, N_steps + 1):
    dt = stable_dt(h_w, S_n)

    if HAS_NUMBA:
        # Fused path: constitutive + SPH in one call
        # (Measure both together — they're now fused)
        tc0 = wall_time.perf_counter()
        # No separate precompute — constitutive is inlined
        tc1 = wall_time.perf_counter()
        t_constitutive += tc1 - tc0

        tk0 = wall_time.perf_counter()
        dhdt  = _sph_dhdt_fused(h_w, S_n, yp, _skip_dirichlet,
                                 nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                 L_inv,K_norm,err_x,err_y, Vp, N_part)
        dSndt = _sph_dSndt_fused(h_w, S_n, yp, _skip_napl,
                                  nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                  L_inv,K_norm,err_x,err_y, Vp, N_part, phi_0)
        tk1 = wall_time.perf_counter()
        t_sph_kernel += tk1 - tk0
    else:
        # Fallback: separate precompute then SPH
        tc0 = wall_time.perf_counter()
        _precompute_step(h_w, S_n)
        tc1 = wall_time.perf_counter()
        t_constitutive += tc1 - tc0

        tk0 = wall_time.perf_counter()
        dhdt  = _sph_div_k_gradH(_fld_Hw, _fld_kw, _fld_Ct, _skip_dirichlet,
                                   nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                   L_inv,K_norm,err_x,err_y, Vp, N_part)
        dSndt = _sph_div_k_gradH(_fld_Hn, _fld_kn, _fld_Cn, _skip_napl,
                                   nbr_ptr,nbr_j,nbr_Fhat,nbr_gWx,nbr_gWy,
                                   L_inv,K_norm,err_x,err_y, Vp, N_part)
        tk1 = wall_time.perf_counter()
        t_sph_kernel += tk1 - tk0

    # -- Update + BCs --
    tu0 = wall_time.perf_counter()
    h_w += dt * dhdt
    S_n += dt * dSndt
    S_n = np.clip(S_n, 0.0, 1.0 - S_res)
    enforce_bcs(h_w, S_n)
    tu1 = wall_time.perf_counter()
    t_update_bc += tu1 - tu0

    t_phys += dt

    # -- Metrics --
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
ms_per_step = t_loop / N_steps * 1000
particles_per_sec = N_part * N_steps / t_loop

# Physics check
q_expected = k_sat * (H_u - H_d) / Lx

stats_lines = [
    f"SPH Three-Phase Benchmark Results",
    f"{'='*50}",
    f"Mode              : {mode_str}",
    f"Grid              : {Nx} x {Ny}  ({N_part} particles)",
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
for line in stats_lines:
    print(line)
print(f"{'='*50}")

# Save to file
tag = f"Nx{Nx}_Tt{n_threads}"
if jit_disabled:
    tag = f"Nx{Nx}_PurePy"
stats_path = os.path.join(OUT_DIR, f"benchmark_{tag}.txt")
with open(stats_path, "w") as f:
    for line in stats_lines:
        f.write(line + "\n")
    # Machine-readable footer for the runner script
    f.write(f"\n# MACHINE_READABLE\n")
    f.write(f"NX={Nx}\n")
    f.write(f"NPART={N_part}\n")
    f.write(f"NSTEPS={N_steps}\n")
    f.write(f"THREADS={n_threads}\n")
    f.write(f"JIT_DISABLED={int(jit_disabled)}\n")
    f.write(f"T_SETUP={t_setup:.6f}\n")
    f.write(f"T_JIT={t_jit:.6f}\n")
    f.write(f"T_LOOP={t_loop:.6f}\n")
    f.write(f"T_CONSTITUTIVE={t_constitutive:.6f}\n")
    f.write(f"T_SPH_KERNEL={t_sph_kernel:.6f}\n")
    f.write(f"T_UPDATE_BC={t_update_bc:.6f}\n")
    f.write(f"T_METRICS={t_metrics:.6f}\n")
    f.write(f"MS_PER_STEP={ms_per_step:.6f}\n")
    f.write(f"L2W_FINAL={l2w_log[-1]:.10e}\n")
    f.write(f"L2N_FINAL={l2n_log[-1]:.10e}\n")
    f.write(f"SNMAX_FINAL={snmax_log[-1]:.10e}\n")
print(f"\nStats saved: {stats_path}")

# ======================================================================
# DIAGNOSTIC PLOT  (adaptive to grid size)
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X2d = xp.reshape(Nx, Ny); Y2d = yp.reshape(Nx, Ny)
Sn_2d = S_n.reshape(Nx, Ny); h_2d = h_w.reshape(Nx, Ny)

# Adaptive figure size: wider for larger grids
fig_w = max(12, min(18, 6 + Nx / 30))
fig_h = fig_w * 0.45

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

# Panel 1: Convergence
axes[0].semilogy(time_log, l2w_log, "b-", lw=1, label=r"L2($dh_w/dt$)")
axes[0].semilogy(time_log, l2n_log, "r-", lw=1, label=r"L2($dS_n/dt$)")
axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("Residual")
axes[0].set_title("Convergence"); axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

# Panel 2: Final Sn field
levels = np.linspace(0, SN_SOURCE, 21)
cf = axes[1].contourf(X2d, Y2d, Sn_2d, levels=levels, cmap="YlOrRd", extend="max")
fig.colorbar(cf, ax=axes[1], shrink=0.85, label=r"$S_n$")
axes[1].contour(X2d, Y2d, h_2d, levels=[0.0], colors="blue", linewidths=1.5, linestyles="--")
axes[1].set_title(f"$S_n$ (t={t_phys:.1f} s)")
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")
axes[1].set_aspect("equal")
# Adaptive tick density
tick_step = max(1, Nx // 10) * dx
axes[1].set_xticks(np.arange(0, Lx + 0.1, tick_step))
axes[1].set_yticks(np.arange(0, Ly + 0.1, tick_step))
if Nx > 150:
    axes[1].tick_params(labelsize=7)

# Panel 3: Timing breakdown
labels = ["Constitutive", "SPH kernels", "Update+BCs", "Metrics"]
sizes  = [t_constitutive, t_sph_kernel, t_update_bc, t_metrics]
colors = ["#4e79a7", "#e15759", "#76b7b2", "#b07aa1"]
wedges, texts, autotexts = axes[2].pie(sizes, labels=labels, autopct="%1.1f%%",
                                        colors=colors, textprops={"fontsize": 8})
axes[2].set_title(f"Time Breakdown ({t_loop:.2f} s)")

fig.suptitle(f"Benchmark: {Nx}×{Ny}, {mode_str}", fontsize=12, y=1.02)

fig_path = os.path.join(OUT_DIR, f"benchmark_{tag}.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved : {fig_path}")
print("Done.")
