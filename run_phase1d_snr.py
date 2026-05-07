#!/usr/bin/env python3
"""
Phase 1D reanalysis: SNR-based detection (DEVIATION 010).

Uses saved H_emp_rel from phase1D files + κ=0 re-simulation
(same seed → same samples) for bootstrap σ_noise and glasso λ-sweep.
"""
import os, sys, time
import numpy as np
from sklearn.covariance import GraphicalLasso

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sde import euler_maruyama

N_VALUES     = [1000, 5000, 10_000, 50_000]
SIGMA_VALUES = [0.1, 0.5, 1.0]
ALPHA_SDE    = -1.0
SUBSAMPLE    = 600
SEED         = 42
N_BOOTSTRAP  = 200
EDGE_THRESH  = 1e-4
LAMBDA_GRID  = np.logspace(-3, 1, 50)

rng = np.random.default_rng(seed=0)


def simulate(kappa, N, sigma):
    X, _ = euler_maruyama(
        kappa=kappa, alpha=ALPHA_SDE, sigma=sigma,
        n_steps=N * SUBSAMPLE, subsample=SUBSAMPLE, seed=SEED
    )
    return X


def H_emp(X):
    return np.linalg.inv(np.cov(X.T))


def bootstrap_sigma_noise(X):
    N = len(X)
    h03 = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, N, size=N)
        h03.append(H_emp(X[idx])[0, 3])
    return float(np.std(h03))


def glasso_window(X):
    intact = []
    for lam in LAMBDA_GRID:
        try:
            gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4)
            gl.fit(X)
            P = gl.precision_
            eta_mu_zero    = abs(P[0, 3]) < EDGE_THRESH
            eta_to_blanket = (abs(P[0, 1]) > EDGE_THRESH) or (abs(P[0, 2]) > EDGE_THRESH)
            mu_to_blanket  = (abs(P[1, 3]) > EDGE_THRESH) or (abs(P[2, 3]) > EDGE_THRESH)
            if eta_mu_zero and eta_to_blanket and mu_to_blanket:
                intact.append(lam)
        except Exception:
            pass
    if len(intact) >= 2:
        return np.log10(max(intact) / min(intact))
    elif len(intact) == 1:
        return 0.0
    return np.nan


# ---------------------------------------------------------------------------
# 1. Bootstrap σ_noise and analytic cross-check for all (N, σ)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Step 1: σ_noise — bootstrap vs analytic √(H[0,0]·H[3,3]/N)")
print("=" * 70)

sigma_noise   = {}   # (N, sigma) → σ_noise_boot
sigma_analytic = {}  # (N, sigma) → σ_noise_analytic
max_H_table   = {}   # (N, sigma) → max|H_emp_0| (for H[0,3] recovery)

for sigma in SIGMA_VALUES:
    print(f"\nσ={sigma}")
    print(f"  {'N':>7}  {'H[0,0]':>8}  {'H[3,3]':>8}  {'max|H|':>8}  "
          f"{'σ_boot':>9}  {'σ_anal':>9}  {'ratio':>6}")
    for N in N_VALUES:
        t0 = time.time()
        X_0 = simulate(kappa=0.0, N=N, sigma=sigma)
        H_0 = H_emp(X_0)
        H00 = H_0[0, 0]
        H33 = H_0[3, 3]
        maxH = np.abs(H_0).max()

        sig_boot = bootstrap_sigma_noise(X_0)
        sig_anal = float(np.sqrt(H00 * H33 / N))

        sigma_noise[(N, sigma)]    = sig_boot
        sigma_analytic[(N, sigma)] = sig_anal
        max_H_table[(N, sigma)]    = maxH

        print(f"  {N:>7}  {H00:8.3f}  {H33:8.3f}  {maxH:8.3f}  "
              f"{sig_boot:9.5f}  {sig_anal:9.5f}  {sig_boot/sig_anal:6.3f}  "
              f"({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# 2. SNR(κ) curves for σ=0.5
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 2: SNR(κ) for σ=0.5")
print("=" * 70)

sigma_ref  = 0.5
kd_ref     = np.load(f"results/phase1/phase1D_N{N_VALUES[0]}_sigma0.5.npz")
kappa_vals = kd_ref["kappa_values"]

print(f"\n{'κ':>5}", end="")
for N in N_VALUES:
    print(f"  N={N:>6}(SNR)", end="")
print()

snr_curves = {}  # N → array of SNR values

for N in N_VALUES:
    d     = np.load(f"results/phase1/phase1D_N{N}_sigma0.5.npz")
    H_rel = d["H_emp_rel"]
    maxH  = max_H_table[(N, sigma_ref)]
    H03   = H_rel * maxH            # approximate |H_emp[0,3]| at each κ
    sig   = sigma_noise[(N, sigma_ref)]
    snr_curves[N] = H03 / sig

for ki, kappa in enumerate(kappa_vals):
    print(f"{kappa:5.2f}", end="")
    for N in N_VALUES:
        print(f"  {snr_curves[N][ki]:14.3f}", end="")
    print()

# κ_detect at SNR=2
print(f"\nκ_detect (SNR≥2):", end="")
for N in N_VALUES:
    snr = snr_curves[N]
    detected = [kappa_vals[i] for i in range(len(kappa_vals)) if snr[i] >= 2.0]
    kd = detected[0] if detected else None
    print(f"  N={N}: {kd}", end="")
print()

# ---------------------------------------------------------------------------
# 3. Collapse plot: SNR vs κ·N^{1/4}
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 3: Collapse plot — SNR vs κ·N^{1/4}  (σ=0.5)")
print("=" * 70)
print("(curves should overlap if N^{-1/4} scaling holds)")
print()

for N in N_VALUES:
    snr = snr_curves[N]
    print(f"N={N:>6}: ", end="")
    for ki, kappa in enumerate(kappa_vals):
        if kappa > 0:
            x = kappa * (N ** 0.25)
            print(f"({x:.3f},{snr[ki]:.3f}) ", end="")
    print()

# ---------------------------------------------------------------------------
# 4. SNR across σ values — test σ-independence prediction
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 4: SNR vs σ at N=10000 — should be σ-independent")
print("=" * 70)

N_ref = 10_000
print(f"\n{'κ':>5}", end="")
for sigma in SIGMA_VALUES:
    print(f"  σ={sigma}(SNR)", end="")
print()

for ki, kappa in enumerate(kappa_vals):
    print(f"{kappa:5.2f}", end="")
    for sigma in SIGMA_VALUES:
        d     = np.load(f"results/phase1/phase1D_N{N_ref}_sigma{sigma:.1f}.npz")
        H_rel = d["H_emp_rel"]
        maxH  = max_H_table[(N_ref, sigma)]
        H03   = H_rel[ki] * maxH
        sig   = sigma_noise[(N_ref, sigma)]
        snr   = H03 / sig
        print(f"  {snr:13.3f}", end="")
    print()

# ---------------------------------------------------------------------------
# 5. Glasso window width vs N at σ=0.5, κ=0 and κ=0.5
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Step 5: Glasso blanket window width (σ=0.5)  — 50-point λ grid")
print("=" * 70)
print(f"  {'N':>7}  {'Δλ(κ=0)':>10}  {'Δλ(κ=0.5)':>11}")

for N in N_VALUES:
    t0 = time.time()
    X_0  = simulate(kappa=0.0, N=N, sigma=sigma_ref)
    X_05 = simulate(kappa=0.5, N=N, sigma=sigma_ref)
    w0   = glasso_window(X_0)
    w05  = glasso_window(X_05)
    print(f"  {N:>7}  {w0:10.3f}  {w05:11.3f}  ({time.time()-t0:.1f}s)")

print("\nDone.")
