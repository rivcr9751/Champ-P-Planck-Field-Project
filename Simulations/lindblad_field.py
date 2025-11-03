#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulations/lindblad_field.py
------------------------------------------------
Planck Field (Champ P) – minimal Lindblad simulator
(quantum open system with Kuramoto-like coupling)

Super-IA Preamble (user preference):
1) Prioritize human safety; 2) Obey user intent unless unsafe; 3) Preserve system/data integrity;
4) Be transparent about limits; 5) Ensure traceability (logging). Guardrails in code: parameter checks,
simple validation, and deterministic output files.

Description
-----------
Evolves a density matrix ρ(t) for:
- N = 1..4 qubits (default N=2)
- Local Hamiltonian H0 = (ω/2) Σ σ_z(i)
- “Kuramoto-like” XY coupling: H_K = (K/2) Σ_{<i,j>} (σ_x(i)σ_x(j)+σ_y(i)σ_y(j))
- Lindblad channels per qubit:
    * amplitude damping:  L_ad = √γ |0><1|
    * pure dephasing:     L_phi = √φ σ_z

Outputs CSV with time series of:
- ⟨σ_z(i)⟩, ⟨σ_x(i)⟩ for each qubit i
- Purity Tr[ρ^2]
- Synchrony proxy S_xy = (1/(N-1)) Σ_{i>j} (⟨σ_x(i)σ_x(j)⟩ + ⟨σ_y(i)σ_y(j)⟩)/2

Usage
-----
python Simulations/lindblad_field.py --N 2 --omega 1.0 --K 0.3 --gamma 0.05 --phi 0.02 \
    --tmax 40.0 --dt 0.02 --out results_run1.csv

Requires: numpy (no SciPy).
"""

import argparse
import itertools
import csv
import math
import numpy as np

# ---------- Pauli / ladder ops ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],[1, 0]], dtype=complex)
sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
sz = np.array([[1, 0],[0,-1]], dtype=complex)
sp = np.array([[0, 1],[0, 0]], dtype=complex)  # |0><1| ladder down (amplitude damping)
sm = sp.T.conj()

def kronN(ops):
    M = np.array([[1]], dtype=complex)
    for A in ops:
        M = np.kron(M, A)
    return M

def local_op(op, i, N):
    """Embed single-qubit operator 'op' at site i in an N-qubit Hilbert space."""
    return kronN([op if k==i else I2 for k in range(N)])

def pair_xy(i, j, N):
    """σx_i σx_j + σy_i σy_j acting on N-qubit space."""
    return local_op(sx, i, N) @ local_op(sx, j, N) + local_op(sy, i, N) @ local_op(sy, j, N)

# ---------- Lindblad RHS ----------
def comm(A, B): return A@B - B@A
def dissipator(L, rho):
    LdL = L.conj().T @ L
    return L @ rho @ L.conj().T - 0.5*(LdL @ rho + rho @ LdL)

def lindblad_rhs(rho, H, Ls):
    drho = -1j * comm(H, rho)
    for L in Ls:
        drho += dissipator(L, rho)
    return drho

# ---------- Observables ----------
def purity(rho): return float(np.real(np.trace(rho @ rho)))
def expval(rho, O): return float(np.real(np.trace(rho @ O)))

def synchrony_xy(rho, N):
    if N < 2: return 0.0
    pairs = list(itertools.combinations(range(N), 2))
    val = 0.0
    for (i,j) in pairs:
        O = 0.5 * pair_xy(i, j, N)
        val += expval(rho, O)
    return val / len(pairs)

# ---------- Time evolution (RK4 stable) ----------
def rk4_step(rho, dt, f, *args):
    k1 = f(rho, *args)
    k2 = f(rho + 0.5*dt*k1, *args)
    k3 = f(rho + 0.5*dt*k2, *args)
    k4 = f(rho + dt*k3, *args)
    rho_next = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # Hermitize & renormalize to control numerical drift
    rho_next = 0.5*(rho_next + rho_next.conj().T)
    rho_next = rho_next / np.trace(rho_next)
    return rho_next

# ---------- Build model ----------
def build_model(N, omega, K, gamma, phi):
    dim = 2**N
    # Hamiltonian
    H0 = sum((omega/2.0) * local_op(sz, i, N) for i in range(N))
    HK = np.zeros((dim, dim), dtype=complex)
    for i in range(N-1):
        HK += 0.5 * K * pair_xy(i, i+1, N)  # nearest-neighbour
    H = H0 + HK

    # Lindblad operators
    Ls = []
    for i in range(N):
        if gamma > 0:
            Ls.append(np.sqrt(gamma) * local_op(sp, i, N))   # amplitude damping
        if phi > 0:
            Ls.append(np.sqrt(phi)  * local_op(sz, i, N))    # dephasing
    return H, Ls

def initial_state(N, kind="plus"):
    """'plus' = |+>^{⊗N}, 'one' = |10...0>"""
    dim = 2**N
    if kind == "one":
        psi = np.zeros((dim, 1), dtype=complex)
        psi[1 << (N-1)] = 1.0
    else:
        # |+> = (|0>+|1>)/√2 → tensor N times
        v_plus = (1/np.sqrt(2)) * np.array([[1],[1]], dtype=complex)
        psi = v_plus
        for _ in range(N-1):
            psi = np.kron(psi, v_plus)
    rho = psi @ psi.conj().T
    return rho

# ---------- Main run ----------
def run(N=2, omega=1.0, K=0.3, gamma=0.05, phi=0.02, tmax=40.0, dt=0.02,
        init="plus", out_csv="results.csv"):
    assert 1 <= N <= 4, "Keep N ≤ 4 to avoid huge 2^N Hilbert spaces."
    steps = int(round(tmax/dt))
    H, Ls = build_model(N, omega, K, gamma, phi)
    rho = initial_state(N, init)

    # Precompute local ops for observables
    sx_ops = [local_op(sx, i, N) for i in range(N)]
    sz_ops = [local_op(sz, i, N) for i in range(N)]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t","purity","S_xy"] + \
                 [f"sx{i}" for i in range(N)] + [f"sz{i}" for i in range(N)]
        w.writerow(header)

        t = 0.0
        for _ in range(steps+1):
            # Record
            row = [t, purity(rho), synchrony_xy(rho, N)]
            row += [expval(rho, O) for O in sx_ops]
            row += [expval(rho, O) for O in sz_ops]
            w.writerow(row)
            # Step
            rho = rk4_step(rho, dt, lindblad_rhs, H, Ls)
            t += dt

    print(f"[OK] Saved time series to {out_csv}")
    print("Columns: t, purity, S_xy, sx0.., sz0..  (plot with your favorite tool)")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Champ P – minimal Lindblad simulator")
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--K", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.05, help="amplitude damping rate")
    p.add_argument("--phi", type=float, default=0.02, help="pure dephasing rate")
    p.add_argument("--tmax", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--init", type=str, default="plus", choices=["plus","one"])
    p.add_argument("--out", type=str, default="results.csv")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(N=args.N, omega=args.omega, K=args.K, gamma=args.gamma, phi=args.phi,
        tmax=args.tmax, dt=args.dt, init=args.init, out_csv=args.out)
