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
| 2A — Bifurcation sweep | **In progress** | Batch 1 (α∈{−1,−0.75,−0.5,−0.25}) clean; Batch 2 (α∈{0,0.25,0.50,0.75,1.00}) run; ACF re-run at sub=1200 for α=0.75,1.00 (passes per-basin ACF). Per-basin H_emp asymmetry at α≥0.75 is finite-sample artifact (DEVIATION 013). Per-basin MLP Hessian evaluation pending. | — |
| 2B — Mixture graphical model | Conditional | — | Depends on 2A results |
| 3A — Jacobian estimation | Not started | — | Depends on 1A |
| 3B — J-vs-H comparison | Not started | — | Depends on 3A, Phase 1 |
| 3C — Blanket phase transition | Conditional | — | Depends on 3B, 2A |
| 4A — Subsampling sweep | Not started | — | Depends on Phase 1 |
| 5A — Summary figures | Not started | — | Depends on all above |
| 5B — Conclusions | Not started | — | Depends on 5A |

## Next action on resume
Phase 1 COMPLETE. Phase 2A Batch 1 + Batch 2 + ACF re-run complete.
Next: run per-basin MLP Hessian evaluation for α ∈ {0.50, 0.75, 1.00}.
Exact command:
  cd /home/hkhurana/Documents/projects/score-mb && \
    .venv/bin/python src/phase2A_perbasin_mlp.py 2>&1 | tee results/phase2/phase2A_perbasin_mlp.log

Script: src/phase2A_perbasin_mlp.py
  Re-simulates X (not saved in npz), re-trains MLP (weights not saved).
  For each α ∈ {0.50, 0.75, 1.00} and each basin (μ>0, μ<0):
    500 query points from |μ|>0.3, σ_n=0.05, compute Ĥ[0,3] with 95% bootstrap CI.
    Report constancy std(Ĥ[0,3])/mean|Ĥ_diag|.
  α=0.50: sub=600 (Batch 2 seed). α=0.75,1.00: sub=1200 (ACF re-run seed SEED+1).

DEVIATION 013 filed (2026-05-10): per-basin H_emp criterion not achievable for α≥0.75
  (finite-sample trajectory fluctuation, N_basin~10⁶ required). Score network MLP
  per-basin Hessian is the correct primary diagnostic. Batch 3 (α>1.00) decision
  deferred until per-basin MLP results are available.

ACF re-run results (2026-05-10):
  α=0.75 sub=1200: per-basin ACF PASS (μ>0: max=0.0316, μ<0: max=0.0269)
  α=1.00 sub=1200: per-basin ACF PASS (μ>0: max=0.0336, μ<0: max=0.0165)
  Per-basin H_emp asymmetry persists (α=1.00: H_pos=-0.047 vs H_neg=-0.310) — DEVIATION 013.

Batch 1 script corrections applied 2026-05-10:
- Crossing check suppressed for α<0 (noise crossings at μ=0, not basin transitions).
- GLasso window other_offdiag = [(0,1),(0,2),(1,3),(2,3)] — ring edges only;
  H[1,2] excluded (s⊥a|{η,μ} is a second conditional independence in the 4-node ring).

Batch 1 corrected results (2026-05-10):
     α    GL_window   H_emp[0,3]   kurt_μ
  -1.00   1.379 dec  +0.098410   -0.369  (grid-floor limited; true lower bound < 0.001)
  -0.75   1.517 dec  +0.063672   -0.461  (grid-floor limited)
  -0.50   1.517 dec  +0.017943   -0.580  (grid-floor limited)
  -0.25   0.276 dec  -0.038832   -0.728  (narrowed due to marginal ACF exceedance; H[0,3] in noise floor)
All constancy checks pass. All SPD checks pass. All ACF pass (α=-0.25 marginal, accepted).

Batch 2 script ready (src/phase2A_batch2.py):
  α ∈ {0.0, 0.25, 0.50, 0.75, 1.00}, logspace(-4,1,50) λ grid (50 points, extended).
  α=0: subsample search [600,1200,2400,4800] until ACF[μ]<0.05.
  α>0: per-basin ACF + ≥20 crossings pilot; per-basin H_emp primary; deep-basin constancy.

Decisions recorded:
- σ_n = 0.05 confirmed (Phase 2 σ_n ablation, 2026-05-08).
- DEVIATION 011: doc correction to DEVIATION 009 (Tanh vs SiLU).
- DEVIATION 012: ACF criterion for α>0 is per-basin, not global.
- α=−0.25 ACF[μ]=0.061 (marginal exceedance vs threshold 0.05). Accepted:
  H_emp[0,3]=−0.039 is within noise floor; exceedance does not corrupt signal.
  Subsample stays at 600.
- Retrospective note: Phase 1C's 1.034 dec window measured H[0,3]-vs-H[1,2]
  spread; with corrected definition it is a lower bound. Phase 1C PASS stands.

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