#!/usr/bin/env python3
"""
Phase 1D: κ_detect comparison — raw precision matrix vs graphical lasso.

Note (DEVIATION 009 / paper_notes.md): score matching is excluded because
W* = −Σ̂_σ⁻¹ in the linear regime — it is the same estimator as H_emp.

Sweep: N ∈ {1000, 5000, 10000, 50000} × σ ∈ {0.1, 0.5, 1.0}
       κ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}

For each (N, σ, κ):
  - Simulate N decorrelated samples (subsample=600, seed=42)
  - H_emp method:   H_emp = Σ̂⁻¹; detect if |H[0,3]|/max|H| ≥ 0.01
  - Graphical lasso: GraphicalLassoCV (cv=5); detect if |Ĥ[0,3]| > 1e-4

Output:
  - results/phase1/phase1D_N{N}_sigma{sigma}.npz  per (N, σ)
  - results/phase1/phase1D_summary.npz              κ_detect table + slopes
"""
import os, sys, time
import numpy as np
from sklearn.covariance import GraphicalLassoCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sde import euler_maruyama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_VALUES     = [1000, 5000, 10_000, 50_000]
SIGMA_VALUES = [0.1, 0.5, 1.0]
KAPPA_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ALPHA_SDE    = -1.0      # bifurcation parameter (fixed: linear regime)
SUBSAMPLE    = 600       # decorrelation lag (6τ at τ=2.0, from DEVIATION 006)
SEED         = 42
EDGE_THRESH  = 1e-4     # |Ĥ[i,j]| < EDGE_THRESH → absent edge
H_EMP_THRESH = 0.01     # |H_emp[0,3]|/max detection threshold

GLASSO_CV_FOLDS  = 5
GLASSO_N_ALPHAS  = 8    # CV grid points
GLASSO_MAX_ITER  = 500

# ---------------------------------------------------------------------------
# Per-(N, σ, κ) worker
# ---------------------------------------------------------------------------

def run_one(kappa, N, sigma):
    """
    Simulate, compute H_emp and graphical lasso.
    Returns (H_emp_rel, H_emp_03_abs, glasso_nonzero, glasso_H03,
             glasso_alpha_cv, t_sim, t_Hemp, t_glasso).
    """
    n_steps = N * SUBSAMPLE

    t_sim_start = time.time()
    X, _ = euler_maruyama(
        kappa=kappa, alpha=ALPHA_SDE, sigma=sigma,
        n_steps=n_steps, subsample=SUBSAMPLE, seed=SEED
    )
    t_sim = time.time() - t_sim_start

    # ── H_emp: raw precision matrix ────────────────────────────────────────
    t0 = time.time()
    Sigma_hat = np.cov(X.T)
    H_emp     = np.linalg.inv(Sigma_hat)
    H03_abs   = abs(H_emp[0, 3])
    H_max     = np.abs(H_emp).max() + 1e-12
    H_rel     = H03_abs / H_max
    t_Hemp    = time.time() - t0

    # ── Graphical lasso with CV λ ──────────────────────────────────────────
    t0 = time.time()
    glcv = GraphicalLassoCV(
        cv=GLASSO_CV_FOLDS, n_refinements=4,
        max_iter=GLASSO_MAX_ITER, tol=1e-4, n_jobs=1,
    )
    glcv.fit(X)
    H_hat        = glcv.precision_
    alpha_cv     = float(glcv.alpha_)
    glasso_H03   = float(H_hat[0, 3])
    glasso_nz    = abs(glasso_H03) > EDGE_THRESH
    t_glasso     = time.time() - t0

    return (H_rel, H03_abs, glasso_nz, glasso_H03, alpha_cv,
            t_sim, t_Hemp, t_glasso)


def kappa_detect(kappa_list, detect_flags):
    """Smallest κ where detect_flags[i] is True; None if never."""
    for kappa, flag in zip(kappa_list, detect_flags):
        if flag:
            return kappa
    return None

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("results/phase1", exist_ok=True)

    # Storage for summary
    # shape: (len(N_VALUES), len(SIGMA_VALUES))
    kd_Hemp_table   = np.full((len(N_VALUES), len(SIGMA_VALUES)), np.nan)
    kd_glasso_table = np.full((len(N_VALUES), len(SIGMA_VALUES)), np.nan)
    t_Hemp_table    = np.zeros_like(kd_Hemp_table)
    t_glasso_table  = np.zeros_like(kd_Hemp_table)

    t_grand_start = time.time()

    for ni, N in enumerate(N_VALUES):
        for si, sigma in enumerate(SIGMA_VALUES):
            print(f"\n{'='*64}", flush=True)
            print(f"N={N:>6}   σ={sigma}   (sim steps per κ = {N*SUBSAMPLE:,})",
                  flush=True)
            print(f"{'κ':>6}  {'H_rel':>8}  {'glasso':>7}  {'α_cv':>8}  "
                  f"{'t_sim':>6}  {'t_H':>6}  {'t_GL':>6}", flush=True)

            H_rels, glasso_nzs = [], []
            t_Hemp_sum = t_glasso_sum = 0.0

            for ki, kappa in enumerate(KAPPA_VALUES):
                (H_rel, H03_abs, gl_nz, gl_H03, alpha_cv,
                 t_sim, t_Hemp, t_glasso) = run_one(kappa, N, sigma)

                H_rels.append(H_rel)
                glasso_nzs.append(gl_nz)
                t_Hemp_sum   += t_Hemp
                t_glasso_sum += t_glasso

                print(f"{kappa:6.2f}  {H_rel:8.4f}  "
                      f"{'NZ' if gl_nz else 'ZERO':>7}  "
                      f"{alpha_cv:8.4f}  "
                      f"{t_sim:6.1f}  {t_Hemp:6.3f}  {t_glasso:6.1f}",
                      flush=True)

            kd_H   = kappa_detect(KAPPA_VALUES, [r >= H_EMP_THRESH for r in H_rels])
            kd_gl  = kappa_detect(KAPPA_VALUES, glasso_nzs)

            kd_Hemp_table[ni, si]   = kd_H   if kd_H   is not None else np.nan
            kd_glasso_table[ni, si] = kd_gl  if kd_gl  is not None else np.nan
            t_Hemp_table[ni, si]    = t_Hemp_sum
            t_glasso_table[ni, si]  = t_glasso_sum

            print(f"  κ_detect(H_emp)  = {kd_H}", flush=True)
            print(f"  κ_detect(glasso) = {kd_gl}", flush=True)
            print(f"  Wall time: H_emp={t_Hemp_sum:.3f}s  "
                  f"glasso={t_glasso_sum:.1f}s", flush=True)

            np.savez(
                f"results/phase1/phase1D_N{N}_sigma{sigma:.1f}.npz",
                kappa_values        = np.array(KAPPA_VALUES),
                H_emp_rel           = np.array(H_rels),
                glasso_nonzero      = np.array(glasso_nzs, dtype=bool),
                kappa_detect_Hemp   = np.float64(kd_H)  if kd_H  is not None else np.nan,
                kappa_detect_glasso = np.float64(kd_gl) if kd_gl is not None else np.nan,
                t_Hemp_total        = np.float64(t_Hemp_sum),
                t_glasso_total      = np.float64(t_glasso_sum),
                N                   = np.int64(N),
                sigma               = np.float64(sigma),
            )

    elapsed = time.time() - t_grand_start
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # ── κ_detect table ────────────────────────────────────────────────────
    print("\n=== κ_detect(H_emp) table ===")
    print(f"{'':>10}", end="")
    for sigma in SIGMA_VALUES:
        print(f"  σ={sigma}", end="")
    print()
    for ni, N in enumerate(N_VALUES):
        print(f"N={N:>6}", end="")
        for si in range(len(SIGMA_VALUES)):
            v = kd_Hemp_table[ni, si]
            print(f"  {v:.2f}" if not np.isnan(v) else "    --", end="")
        print()

    print("\n=== κ_detect(glasso) table ===")
    print(f"{'':>10}", end="")
    for sigma in SIGMA_VALUES:
        print(f"  σ={sigma}", end="")
    print()
    for ni, N in enumerate(N_VALUES):
        print(f"N={N:>6}", end="")
        for si in range(len(SIGMA_VALUES)):
            v = kd_glasso_table[ni, si]
            print(f"  {v:.2f}" if not np.isnan(v) else "    --", end="")
        print()

    # ── Log-log slope (σ=0.5 column) ─────────────────────────────────────
    si_ref = SIGMA_VALUES.index(0.5)
    log_N  = np.log10(N_VALUES)

    print("\n=== Log-log slope κ_detect vs N (σ=0.5, target ≈ −0.25) ===")
    for name, table in [("H_emp", kd_Hemp_table), ("glasso", kd_glasso_table)]:
        kds = table[:, si_ref]
        valid_mask = ~np.isnan(kds) & (kds > 0)
        if valid_mask.sum() >= 2:
            slope, intercept = np.polyfit(log_N[valid_mask],
                                          np.log10(kds[valid_mask]), 1)
            flag = "" if abs(slope - (-0.25)) < 0.15 else "  ← FLAG: differs from -1/4"
            print(f"  {name}: slope = {slope:.3f}{flag}")
        else:
            print(f"  {name}: insufficient valid points for slope")

    # ── Wall time table ───────────────────────────────────────────────────
    print("\n=== Wall time: glasso / H_emp ratio (σ=0.5) ===")
    for ni, N in enumerate(N_VALUES):
        ratio = t_glasso_table[ni, si_ref] / (t_Hemp_table[ni, si_ref] + 1e-9)
        print(f"  N={N:>6}  glasso={t_glasso_table[ni,si_ref]:.1f}s  "
              f"H_emp={t_Hemp_table[ni,si_ref]:.3f}s  ratio={ratio:.0f}×")

    # ── Save summary ──────────────────────────────────────────────────────
    np.savez(
        "results/phase1/phase1D_summary.npz",
        N_values            = np.array(N_VALUES),
        sigma_values        = np.array(SIGMA_VALUES),
        kappa_values        = np.array(KAPPA_VALUES),
        kd_Hemp_table       = kd_Hemp_table,
        kd_glasso_table     = kd_glasso_table,
        t_Hemp_table        = t_Hemp_table,
        t_glasso_table      = t_glasso_table,
    )
    print("\nSaved results/phase1/phase1D_summary.npz", flush=True)
