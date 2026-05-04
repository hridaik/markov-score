# PROGRESS.md

## Current phase: Phase 0 (Ground Truth) — NOT STARTED

## Phase status

| Phase | Status | Key numbers | Blocker |
|-------|--------|-------------|---------|
| 0A — Lyapunov solution | **COMPLETE** | H[0,3]/max=1.76e-15 at κ=0; H[0,3]~κ² (O(κ²) growth); κ=0.40 first exceeds 0.1; all J Hurwitz; Σ SPD throughout (min eig 0.083, cond 3.0) | — |
| 0B — Fokker-Planck numerics | Not started | — | Depends on 0A |
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
Phase 0A complete. Begin Phase 0B (Fokker-Planck numerics, nonlinear regime)
after supervisor confirms 0A and grants go-ahead. Requires CHECKPOINT before
running any FPE grid solve.

## Notes for supervisor
- Phase 0A COMPLETE — all five completion criteria pass (DEVIATION 002 relative criterion).
- Key finding: H[0,3] grows as O(κ²) not O(κ¹). Cause: η↔μ symmetry of default
  parameters forces first-order perturbation to zero. See CONTEXT.md for full
  derivation and implications for Phase 3B.
- Deviations logged: DEVIATION 001 (c=0.5 fix), DEVIATION 002 (relative threshold).
- Results saved: results/phase0/phase0A_sweep.npz (21 κ values, all matrices and diagnostics).