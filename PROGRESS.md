# PROGRESS.md

## Current phase: Phase 1 (Linear regime) — IN PROGRESS

## Phase status

| Phase | Status | Key numbers | Blocker |
|-------|--------|-------------|---------|
| 0A — Lyapunov solution | **COMPLETE** | H[0,3]/max=1.76e-15 at κ=0; H[0,3]~κ² (O(κ²) growth); κ=0.40 first exceeds 0.1; all J Hurwitz; Σ SPD throughout (min eig 0.083, cond 3.0) | — |
| 0B — Fokker-Planck numerics | **COMPLETE** | Linear (α<0): global PASS all cases; α=0: global FAIL (non-Gaussian, expected); α>0: global FAIL, per-basin PASS. Primary finding: blanket is basin-specific in nonlinear systems. α=+2 ergodicity caveat (DEVIATION 005). | — |
| 1A — Simulation infrastructure | **COMPLETE** | Frobenius (η,s,a block) = 0.0434 PASS; ACF at lag-600 = max 0.021 PASS; subsample=600 (DEVIATION 006) | — |
| 1B — Score matching | **COMPLETE** | κ=0: \|W*[0,3]\|/max=0.0053 PASS; r(-W*[0,3],H_emp[0,3])=1.000 PASS; Frobenius(η,s,a)=0.024 PASS. Analytic W*=−Σ̂_σ⁻¹ (DEVIATION 009). H_emp is correct ground truth (3–5× > H_lyap at large κ, nonlinear self-consistency). | — |
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
Phase 1B complete (linear regime). Awaiting go-ahead for Phase 1C (graphical lasso).
Standard simulation config for all Phase 1 work: α=−1.0, κ sweep [0,0.5] (11 pts),
n_steps=6×10⁶, subsample=600, dt=0.01, seed=42 → N=10,000 decorrelated samples.
Ground truth for Phase 1B: H_emp = Σ̂⁻¹ (sample precision), NOT H_lyap.
H_emp[0,3] at κ=0.5 = 0.474 vs H_lyap = 0.148 (documented in CONTEXT.md).

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
- Phase 1A COMPLETE — Frobenius (η,s,a block) = 0.0434 PASS; ACF max 0.021 PASS;
  subsample=600 (DEVIATION 006).
- Phase 1B COMPLETE — analytic W*=−Σ̂_σ⁻¹ (DEVIATION 009, linear network is exact
  closed-form). |W*[0,3]|/max=0.0053 PASS; r(-W*,H_emp)=1.000 PASS;
  Frobenius(η,s,a)=0.024 PASS. H_emp[0,3]=0.474 at κ=0.5 (3× H_lyap due to
  nonlinear self-consistency; CONTEXT.md updated). Deviations 007,009 logged.