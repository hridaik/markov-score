#!/usr/bin/env python3
"""
Phase 0B benchmark — sample precision matrix method.

Runs Euler-Maruyama for the 4D chemosensing SDE, then estimates the
precision matrix H_emp = inv(cov(X)) globally and, for α > 0,
per-basin (split by sign of μ).

Primary diagnostic: H_emp[0,3] — must be ~0 at κ=0 for the blanket
to be intact. Diagonal H_emp compared to nonlinear self-consistency
prediction for μ (Lyapunov is approximate for μ only).

Usage:
    python bench_phase0B_mckde.py --alpha -1.0
    python bench_phase0B_mckde.py --alpha  1.0
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lyapunov import solve_lyapunov, DEFAULTS
from sde import euler_maruyama


def precision_report(X, label="global"):
    """Compute and return H_emp from samples X. Print diagnostics."""
    n = len(X)
    Sigma = np.cov(X.T)
    H = np.linalg.inv(Sigma)
    names = ["η", "s", "a", "μ"]

    print(f"\n  [{label}]  n={n}")
    print(f"  H[0,3] = {H[0,3]:+.4e}   |H[0,3]|/max|H_ij| = "
          f"{abs(H[0,3])/np.abs(H).max():.4e}")
    print(f"  Diagonal: " + "  ".join(
        f"H[{d},{d}]({names[d]})={H[d,d]:.4f}" for d in range(4)))
    return H, Sigma


def run(alpha, seed=42):
    state_names = ["η", "s", "a", "μ"]
    print(f"\n{'='*60}")
    print(f"Phase 0B benchmark  alpha={alpha:+.1f}  seed={seed}")
    print(f"{'='*60}")

    # Lyapunov ground truth: valid only for α < 0 (linearisation stable).
    # At α=0 the ring_gain_ratio divides by |α|; at α>0 J is not Hurwitz.
    lyap_valid = (alpha < 0)
    if lyap_valid:
        Sigma_lyap, H_lyap, _, _ = solve_lyapunov(
            alpha=alpha, kappa=DEFAULTS["kappa"], sigma=DEFAULTS["sigma"])
    else:
        Sigma_lyap = H_lyap = None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    t0 = time.time()
    X, n_crossings = euler_maruyama(alpha=alpha, seed=seed)
    sim_time = time.time() - t0
    print(f"Simulation: {sim_time:.1f}s  n={len(X)}  "
          f"μ sign-crossings={n_crossings}")

    # ------------------------------------------------------------------
    # Global precision matrix
    # ------------------------------------------------------------------
    print("\n--- Global precision matrix ---")
    H_global, Sigma_global = precision_report(X, label="all samples")

    print(f"\n  Full H_emp (global):")
    for i in range(4):
        row = "  ".join(f"{H_global[i,j]:+7.4f}" for j in range(4))
        print(f"    [{state_names[i]}]  {row}")

    if lyap_valid:
        print(f"\n  Lyapunov reference (exact for η,s,a; approx for μ):")
        for i in range(4):
            row = "  ".join(f"{H_lyap[i,j]:+7.4f}" for j in range(4))
            print(f"    [{state_names[i]}]  {row}")

    blanket_rel = abs(H_global[0, 3]) / np.abs(H_global).max()
    blanket_pass = blanket_rel < 1e-2
    print(f"\n  BLANKET CHECK |H[0,3]|/max|H_ij| = {blanket_rel:.4e}  "
          f"({'PASS' if blanket_pass else 'FAIL'} threshold 1e-2)")

    # ------------------------------------------------------------------
    # Per-basin analysis (α > 0 only)
    # ------------------------------------------------------------------
    H_pos = H_neg = None
    if alpha > 0:
        print(f"\n--- Per-basin analysis (α={alpha:+.1f}) ---")
        print(f"  Total μ basin crossings: {n_crossings}")
        if n_crossings < 20:
            print(f"  WARNING: < 20 basin crossings — trajectory may not be ergodic")

        pos_mask = X[:, 3] > 0
        neg_mask = X[:, 3] < 0
        n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
        print(f"  Basin μ>0: {n_pos} samples ({100*n_pos/len(X):.1f}%)")
        print(f"  Basin μ<0: {n_neg} samples ({100*n_neg/len(X):.1f}%)")

        if n_pos >= 50:
            H_pos, _ = precision_report(X[pos_mask], label="μ>0 basin")
            brel = abs(H_pos[0,3]) / np.abs(H_pos).max()
            print(f"  BLANKET CHECK (μ>0) |H[0,3]|/max|H_ij| = {brel:.4e}  "
                  f"({'PASS' if brel < 1e-2 else 'FAIL'})")
        if n_neg >= 50:
            H_neg, _ = precision_report(X[neg_mask], label="μ<0 basin")
            brel = abs(H_neg[0,3]) / np.abs(H_neg).max()
            print(f"  BLANKET CHECK (μ<0) |H[0,3]|/max|H_ij| = {brel:.4e}  "
                  f"({'PASS' if brel < 1e-2 else 'FAIL'})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs("results/phase0", exist_ok=True)
    outfile = f"results/phase0/phase0B_mckde_alpha{alpha:+.1f}.npz"
    save_dict = dict(
        X_samples=X,
        H_global=H_global,
        Sigma_global=Sigma_global,
        n_crossings=n_crossings,
        alpha=alpha,
        seed=seed,
    )
    if lyap_valid:
        save_dict["H_lyap"] = H_lyap
        save_dict["Sigma_lyap"] = Sigma_lyap
    if H_pos is not None:
        save_dict["H_pos"] = H_pos
    if H_neg is not None:
        save_dict["H_neg"] = H_neg
    np.savez(outfile, **save_dict)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(alpha=args.alpha, seed=args.seed)
