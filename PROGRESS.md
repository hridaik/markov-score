# PROGRESS.md

## Current phase: Phase 1 (Linear regime) — COMPLETE

## Phase status

| Phase | Status | Key numbers | Blocker |
|-------|--------|-------------|---------|
| 0A — Lyapunov solution | **COMPLETE** | H[0,3]/max=1.76e-15 at κ=0; H[0,3]~κ² (O(κ²) growth); κ=0.40 first exceeds 0.1; all J Hurwitz; Σ SPD throughout (min eig 0.083, cond 3.0) | — |
| 0B — Fokker-Planck numerics | **COMPLETE** | Linear (α<0): global PASS all cases; α=0: global FAIL (non-Gaussian, expected); α>0: global FAIL, per-basin PASS. Primary finding: blanket is basin-specific in nonlinear systems. α=+2 ergodicity caveat (DEVIATION 005). | — |
| 1A — Simulation infrastructure | **COMPLETE** | Frobenius (η,s,a block) = 0.0434 PASS; ACF at lag-600 = max 0.021 PASS; subsample=600 (DEVIATION 006) | — |
| 1B — Score matching | **COMPLETE** | κ=0: \|W*[0,3]\|/max=0.0053 PASS; r(-W*[0,3],H_emp[0,3])=1.000 PASS; Frobenius(η,s,a)=0.024 PASS. Analytic W*=−Σ̂_σ⁻¹ (DEVIATION 009). H_emp is correct ground truth (3–5× > H_lyap at large κ, nonlinear self-consistency). | — |
| 1C — Graphical lasso | **COMPLETE** | κ=0: window=1.034 dec PASS (>0.5); monotone narrowing PASS (1.034→0.724 dec); λ grid [0.003, 3.16], 30 pts. Window driven by upper boundary (λ_high 0.034→0.021), λ_low at grid floor. | — |
| 1D — Method comparison | **COMPLETE** | H_emp κ_detect (SNR=2, σ=0.5): N=5000→0.30, N=10000→0.25, N=50000→0.15 (N=1000 below grid); slope=−0.304≈−1/4 ✓. Glasso: window positive at all κ (1.47–1.55 dec at κ=0, 0.41–0.82 dec at κ=0.5) — cannot detect solenoidal leakage; H_emp 43–158× faster. DEVIATION 010. | — |
| 2A — Bifurcation sweep | Not started | — | Depends on Phase 1 |
| 2B — Mixture graphical model | Conditional | — | Depends on 2A results |
| 3A — Jacobian estimation | Not started | — | Depends on 1A |
| 3B — J-vs-H comparison | Not started | — | Depends on 3A, Phase 1 |
| 3C — Blanket phase transition | Conditional | — | Depends on 3B, 2A |
| 4A — Subsampling sweep | Not started | — | Depends on Phase 1 |
| 5A — Summary figures | Not started | — | Depends on all above |
| 5B — Conclusions | Not started | — | Depends on 5A |

## Next action on resume
Phase 1 COMPLETE. Awaiting go-ahead for Phase 2A (bifurcation sweep).
Phase 2A: sweep α from −1 to +3 in steps of 0.25, κ=0 throughout, σ=0.5.
For each α: N=10,000 decorrelated samples (subsample=600). Global H_emp for α<0;
per-basin H_emp for α>0 (per DEVIATION 004/005 precedent). Identify α_crit for
graphical lasso failure and per-basin blanket structure. Hessian constancy check
(std(H(x))/mean|H_diag|) gates every result as per Phase 1 arc finding.

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
- Phase 1D COMPLETE (DEVIATION 010) — SNR-based analysis. Six Phase 1 findings:
  1. Score matching = precision matrix in linear regime (W*=−Σ̂_σ⁻¹, analytic proof).
  2. O(κ²) scaling from Z₂ symmetry (κ^1.93 Lyapunov; κ^1.2 empirical in detectable range).
  3. Nonlinear self-consistency: H_emp[0,3]=0.474 vs H_lyap=0.148 at κ=0.5 (3.2×).
  4. Blanket is basin-specific in nonlinear systems (per-basin H_emp correct for α>0).
  5. Graphical lasso cannot detect blanket violation onset: L1 penalty drives H[0,3]→0
     by design; window narrows 1.5→0.6 decades but never closes; wrong tool for κ_detect.
  6. N^{-1/4} scaling confirmed: slope=−0.304, κ_detect σ=0.5: N=5000→0.30, N=10000→0.25,
     N=50000→0.15. σ_noise √(H[0,0]·H[3,3]/N) validated (ratio 0.87–1.14 vs bootstrap).
     SNR~σ^{0.8} (mild σ-dependence from nonlinear self-consistency).
  Results: results/phase1/phase1D_*.npz + run_phase1d_snr.py.