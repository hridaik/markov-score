# PROGRESS.md

## Current phase: Phase 1 (Linear regime) — NOT STARTED

## Phase status

| Phase | Status | Key numbers | Blocker |
|-------|--------|-------------|---------|
| 0A — Lyapunov solution | **COMPLETE** | H[0,3]/max=1.76e-15 at κ=0; H[0,3]~κ² (O(κ²) growth); κ=0.40 first exceeds 0.1; all J Hurwitz; Σ SPD throughout (min eig 0.083, cond 3.0) | — |
| 0B — Fokker-Planck numerics | **COMPLETE** | Linear (α<0): global PASS all cases; α=0: global FAIL (non-Gaussian, expected); α>0: global FAIL, per-basin PASS. Primary finding: blanket is basin-specific in nonlinear systems. α=+2 ergodicity caveat (DEVIATION 005). | — |
| 1A — Simulation infrastructure | Not started | — | Depends on 0A |
| 1B — Score matching | Not started | — | Depends on 1A |
| 1C — Graphical lasso | Not started | — | Depends on 1A |
| 1D — Method comparison | Not started | — | Depends on 1B, 1C |
| 2A — Bifurcation sweep | Not started | — | Depends on Phase 1 |
| 2B — Mixture graphical model | Conditional | — | Depends on 2A results |
| 3A — Jacobian estimation | Not started | — | Depends on 1A |
| 3B — J-vs-H comparison | Not started | — | Depends on 3A, Phase 1 |
| 3C — Blanket phase transition | Conditional | — | Depends on 3B, 2A |
| 4A — Subsampling sweep | Not started | — | Depends on Phase 1 |
| 5A — Summary figures | Not started | — | Depends on all above |
| 5B — Conclusions | Not started | — | Depends on 5A |

## Next action on resume
Awaiting Phase 1A go-ahead. Phase 1A: simulation infrastructure —
simulate N=10k trajectories at α=−1, κ=0 and verify sample covariance
matches Lyapunov within 5%; verify autocorrelation at subsampling lag < 0.05.

## Notes for supervisor
- Phase 0A COMPLETE — all five completion criteria pass (DEVIATION 002 relative criterion).
  Key finding: H[0,3] grows as O(κ²). See CONTEXT.md.
- Phase 0B COMPLETE — sample precision matrix sweep over α∈{−2,−1,−0.5,0,+1,+2}.
  Key numbers:
    Linear (α<0): global |H[0,3]|/max < 1e-2 in all cases (PASS).
    Bifurcation (α=0): global FAIL as expected — H[3,3]=4.89, non-Gaussian marginal.
    Bistable (α>0): global FAIL; per-basin PASS at α=+1 (both within sampling noise)
      and α=+2 (PASS, ergodicity caveat DEVIATION 005).
  Primary finding: statistical blanket is a basin-specific property in nonlinear systems;
    global precision matrix is not the right estimator in the bistable regime.
  KDE Hessian abandoned (h⁴ amplification unusable for small off-diagonal entries).
  μ precision discrepancy vs Lyapunov explained by nonlinear self-consistency
    (H[3,3]_emp≈10 vs H_lyap=8 at α=−1); see CONTEXT.md.
- Deviations logged: DEVIATION 001–005. See DEVIATIONS.md.
- Results saved: results/phase0/ (phase0A_sweep.npz + phase0B_mckde_alpha*.npz for all α).
- src/sde.py: Euler-Maruyama integrator (reusable for Phase 1A onwards).