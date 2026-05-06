#!/usr/bin/env python3
"""
Phase 0B benchmark: single FPE solve at alpha=-1, kappa=0.
Reports stability bound, convergence, integration, residual, mu marginal,
and Hessian diagnostics vs Lyapunov ground truth.
"""
import time
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lyapunov import solve_lyapunov, DEFAULTS
from fokker_planck import solve_fpe, compute_hessian_log_p

# --- Grid (Phase 0B spec: 25 pts per dim, domain ±4*sigma_marginal) ---
Sigma, H_lyap, _, _ = solve_lyapunov(kappa=0.0, sigma=DEFAULTS['sigma'])
sigma_marginal = float(np.sqrt(np.diag(Sigma).max()))
domain = 4.0 * sigma_marginal
n = 25
grids = [np.linspace(-domain, domain, n) for _ in range(4)]
dx = float(grids[0][1] - grids[0][0])

print(f"sigma_marginal={sigma_marginal:.4f}  domain=+-{domain:.4f}  "
      f"dx={dx:.4f}  n_per_dim={n}  N={n**4}")
print(f"alpha={DEFAULTS['alpha']}  kappa={DEFAULTS['kappa']}  sigma={DEFAULTS['sigma']}")

# --- FPE solve ---
print("\nRunning FPE solve (power iteration)...")
t0 = time.time()
p_grid, info = solve_fpe(grids)
wall_time = time.time() - t0

# --- Stability bound report ---
print("\n--- Stability bound ---")
print(f"  D_max  = max |A[i,i]| over interior = {info['D_max']:.4f}")
print(f"  1/D_max = dt_stable                 = {info['dt_stable']:.6f}")
print(f"  dt used                             = {info['dt_used']:.4f}")
print(f"  dt < dt_stable (stability satisfied): {info['dt_ok']}")

# --- Convergence report ---
print(f"\n=== Benchmark: alpha={DEFAULTS['alpha']}, kappa={DEFAULTS['kappa']} ===")
print(f"  Wall time:                    {wall_time:.2f} s")
print(f"  Power iteration count:        {info['n_iters']}")
print(f"  Converged (final_err < tol):  {info['converged']}")
print(f"  Final iteration error:        {info['final_err']:.2e}")
print(f"  True residual ||A p||_inf:    {info['true_residual']:.2e}")
print(f"  Integration error:            {info['integration_error']:.2e}  (need < 0.01)")
print(f"  INTEGRATION: {'PASS' if info['integration_error'] < 0.01 else 'FAIL'}")

# --- Marginal density of mu ---
print("\n--- Marginal density of μ (integrated over η, s, a) ---")
mu_grid  = info['mu_grid']
mu_marg  = info['mu_marginal']
print(f"  {'μ':>8}   {'p(μ)':>12}")
for mu_val, p_val in zip(mu_grid, mu_marg):
    print(f"  {mu_val:+8.3f}   {p_val:.6e}")
marg_integral = float(mu_marg.sum() * dx)
print(f"  Integral of marginal: {marg_integral:.6f}  (should be 1.0)")

# --- Hessian of log p ---
print("\nComputing Hessian at high-density points...")
t1 = time.time()
H_samples, coords, _ = compute_hessian_log_p(p_grid, grids)
print(f"  {time.time()-t1:.2f}s,  n_pts sampled={len(H_samples)}")

print("\n=== H[0,3] at high-density points (must be ~0 at kappa=0) ===")
h03 = H_samples[:, 0, 3]
print(f"  max |H[0,3]|  = {np.abs(h03).max():.4e}")
print(f"  mean |H[0,3]| = {np.abs(h03).mean():.4e}")
print(f"  std  H[0,3]   = {h03.std():.4e}")

print("\n=== Diagonal H(log p): FPE mean vs expected −H_lyap[d,d] ===")
state_names = ['η', 's', 'a', 'μ']
for d in range(4):
    fpe_val  = float(H_samples[:, d, d].mean())
    expected = float(-H_lyap[d, d])          # Hessian of log p = −precision
    rel_err  = (fpe_val - expected) / abs(expected) if expected != 0 else float('nan')
    print(f"  [{d},{d}]({state_names[d]}): FPE={fpe_val:.4f}  "
          f"expect={expected:.4f}  rel_err={rel_err:+.4f}")

# --- Save ---
os.makedirs('results/phase0', exist_ok=True)
outfile = 'results/phase0/phase0B_benchmark_alpha-1.npz'
np.savez(outfile,
         p_grid=p_grid,
         mu_marginal=mu_marg,
         mu_grid=mu_grid,
         H_samples=H_samples,
         coords=coords,
         H_lyap=H_lyap,
         Sigma=Sigma,
         n_iters=info['n_iters'],
         final_err=info['final_err'],
         converged=info['converged'],
         integration_error=info['integration_error'],
         true_residual=info['true_residual'],
         D_max=info['D_max'],
         dt_stable=info['dt_stable'],
         dt_used=info['dt_used'])
print(f"\nSaved to {outfile}")
