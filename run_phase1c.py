#!/usr/bin/env python3
"""
Phase 1C: Graphical lasso blanket window across 11 κ values.

For each κ:
  1. Simulate 6M steps (subsample=600) → N=10,000 samples
  2. Fit GraphicalLasso over 30-point log λ grid
  3. For each λ: check blanket structure (H[0,3]=0, blanket edges nonzero)
  4. Find blanket window [λ_low, λ_high] and width in decades
  5. Save per-κ .npz + 11-point summary

Completion criteria (from CLAUDE.md):
  - Window width > 0.5 decades at κ=0
  - Window narrows monotonically with κ
"""
import os, sys, time
import numpy as np
from sklearn.covariance import GraphicalLasso

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lyapunov import DEFAULTS
from sde import euler_maruyama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAPPA_VALUES = np.linspace(0.0, 0.5, 11)
LAMBDA_GRID  = np.logspace(-2.5, 0.5, 30)   # ~0.003 to ~3.16
EDGE_THRESH  = 1e-4                          # |P[i,j]| below → absent edge

# Load Phase 1B H_emp ground truth for reference
_p1b = np.load("results/phase1/phase1B_sweep.npz")
H_EMP_03_GT = _p1b["H_emp_03"]  # (11,) positive precision H_emp[0,3]

# ---------------------------------------------------------------------------
# Blanket check
# ---------------------------------------------------------------------------

def blanket_structure(P):
    """
    Returns (eta_mu_zero, eta_to_blanket, mu_to_blanket, blanket_intact).
    Blanket = {s=1, a=2} separating η=0 from μ=3.
    Intact iff: H[0,3]=0, (H[0,1]≠0 or H[0,2]≠0), (H[1,3]≠0 or H[2,3]≠0).
    """
    eta_mu_zero    = abs(P[0, 3]) < EDGE_THRESH
    eta_to_blanket = (abs(P[0, 1]) > EDGE_THRESH) or (abs(P[0, 2]) > EDGE_THRESH)
    mu_to_blanket  = (abs(P[1, 3]) > EDGE_THRESH) or (abs(P[2, 3]) > EDGE_THRESH)
    intact = eta_mu_zero and eta_to_blanket and mu_to_blanket
    return eta_mu_zero, eta_to_blanket, mu_to_blanket, intact

# ---------------------------------------------------------------------------
# Per-κ run
# ---------------------------------------------------------------------------

def run_kappa(kappa, kappa_idx):
    print(f"\n[κ={kappa:.3f}  ({kappa_idx+1}/11)]", flush=True)
    t0 = time.time()
    X, n_cross = euler_maruyama(
        kappa=kappa, alpha=DEFAULTS["alpha"], sigma=DEFAULTS["sigma"],
        n_steps=6_000_000, subsample=600, seed=42
    )
    print(f"  sim: {time.time()-t0:.1f}s  n={len(X)}", flush=True)

    intact_lams    = []
    all_H03        = []
    all_intact     = []
    all_structures = []

    for lam in LAMBDA_GRID:
        try:
            gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4)
            gl.fit(X)
            P  = gl.precision_
            em, etb, mtb, intact = blanket_structure(P)
            all_H03.append(P[0, 3])
            all_intact.append(intact)
            all_structures.append((em, etb, mtb))
            if intact:
                intact_lams.append(lam)
        except Exception as exc:
            all_H03.append(np.nan)
            all_intact.append(False)
            all_structures.append((False, False, False))
            print(f"  λ={lam:.4f} FAIL: {exc}", flush=True)

    if len(intact_lams) >= 2:
        lam_low  = min(intact_lams)
        lam_high = max(intact_lams)
        width    = np.log10(lam_high / lam_low)
    elif len(intact_lams) == 1:
        lam_low = lam_high = intact_lams[0]
        width = 0.0
    else:
        lam_low = lam_high = np.nan
        width = -np.inf

    H03_gt = float(H_EMP_03_GT[kappa_idx])
    print(f"  λ_low={lam_low:.4f}  λ_high={lam_high:.4f}  "
          f"width={width:.3f} dec  n_intact={len(intact_lams)}  "
          f"H_emp[0,3]={H03_gt:.4f}", flush=True)

    os.makedirs("results/phase1", exist_ok=True)
    np.savez(
        f"results/phase1/phase1C_kappa_{kappa:.3f}.npz",
        lambda_grid    = LAMBDA_GRID,
        H03_per_lambda = np.array(all_H03, dtype=np.float64),
        intact_per_lambda = np.array(all_intact, dtype=bool),
        lam_low        = np.float64(lam_low)  if not np.isnan(lam_low)  else np.nan,
        lam_high       = np.float64(lam_high) if not np.isnan(lam_high) else np.nan,
        window_decades = np.float64(width),
        H_emp_03       = np.float64(H03_gt),
        kappa          = np.float64(kappa),
    )

    return dict(
        lam_low=lam_low, lam_high=lam_high,
        width=width, n_intact=len(intact_lams),
        H_emp_03=H03_gt,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"λ grid: {len(LAMBDA_GRID)} values  "
          f"[{LAMBDA_GRID[0]:.4f}, {LAMBDA_GRID[-1]:.4f}]", flush=True)
    print(f"Edge threshold: {EDGE_THRESH}", flush=True)

    results = []
    t_total = time.time()
    for i, kappa in enumerate(KAPPA_VALUES):
        results.append(run_kappa(kappa, i))

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Summary
    widths = np.array([r["width"] for r in results])
    np.savez(
        "results/phase1/phase1C_sweep.npz",
        kappa_values   = KAPPA_VALUES,
        lambda_grid    = LAMBDA_GRID,
        lam_low        = np.array([r["lam_low"]  for r in results]),
        lam_high       = np.array([r["lam_high"] for r in results]),
        window_decades = widths,
        H_emp_03       = np.array([r["H_emp_03"] for r in results]),
    )
    print("Saved results/phase1/phase1C_sweep.npz", flush=True)

    # Completion criteria
    print("\n--- Completion criteria ---")
    w0 = results[0]["width"]
    monotone = all(widths[i] >= widths[i+1] for i in range(len(widths)-1))
    print(f"  Window at κ=0: {w0:.3f} decades  "
          f"{'PASS' if w0 > 0.5 else 'FAIL'}  (threshold > 0.5)")
    print(f"  Monotone narrowing: {'PASS' if monotone else 'FAIL'}")

    print(f"\n{'κ':>6}  {'λ_low':>8}  {'λ_high':>8}  {'width(dec)':>10}  "
          f"{'n_intact':>8}  {'H_emp[0,3]':>11}")
    for i, (kappa, r) in enumerate(zip(KAPPA_VALUES, results)):
        print(f"{kappa:6.3f}  {r['lam_low']:8.4f}  {r['lam_high']:8.4f}  "
              f"{r['width']:10.3f}  {r['n_intact']:8d}  {r['H_emp_03']:11.4f}")
