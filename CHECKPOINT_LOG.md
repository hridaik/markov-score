# CHECKPOINT_LOG.md

## Log of all checkpoints and decisions

### CP-001 — 2026-05-05 — Phase 0B
**Proposed action:** Run FPE benchmark solve at α=−1 (single α value)
to verify solver convergence before full sweep.
**Justification:**
  1. Previous diagnostic: single-κ Phase 0A verification confirmed
     matrix construction in lyapunov.py is correct.
  2. Rules out: implementation bugs in build_jacobian and solve_lyapunov.
  3. Tests: whether GMRES + ILU(0) converges on the 4D FPE system
     within reasonable time and produces a valid density.
  4. Confirming: integration error < 0.01, convergence in < 300 iterations.
     Refuting: non-convergence, integration error > 0.01, or H values
     inconsistent with Lyapunov ground truth.
**Decision:** go ahead (approved by supervisor — retroactive log)
**Result:** Converged in 137 iterations, runtime 323s. Integration error
≈ 0. H values incorrect — traced to missing compressibility term.
Compressibility correction since added. Matrix correctness unverified.
Protocol violation: computation exceeded 5 minutes without a
checkpoint being filed. Logged retroactively.

### CP-002 — 2026-05-07 — Phase 1D
**Proposed action:** Run Phase 1D sweep (run_phase1d.py): N∈{1000,5000,10000,50000}
× σ∈{0.1,0.5,1.0} × κ∈9 values, comparing H_emp (raw precision) vs GraphicalLassoCV.
**Justification:**
  1. Previous diagnostic: Phase 1C complete (glasso window 1.034 dec at κ=0, monotone
     narrowing). Phase 1B complete (W*=−Σ̂_σ⁻¹ analytic, r=1.000 vs H_emp).
  2. Rules out: measurement-level issues in both estimators before method comparison.
  3. Tests: κ_detect(N) scaling — whether slope ≈ −1/4 as predicted from O(κ²) signal
     + O(1/√N) noise. Whether glasso out-performs H_emp at detecting solenoidal leakage.
  4. Confirming: slope ≈ −0.25 ± 0.15, monotone decrease of κ_detect with N.
  5. Refuting: non-monotone κ_detect, slope far from −0.25, or glasso false positives.
**Decision:** go ahead (supervisor ran script; output not pasted, results loaded from .npz)
**Result:** Binary thresholds produced non-monotone κ_detect. Diagnosed: absolute
thresholds don't adapt to noise floor. DEVIATION 010 filed. Reanalysis approved.

### CP-003 — 2026-05-07 — Phase 1D (reanalysis)
**Proposed action:** Run run_phase1d_snr.py: SNR-based reanalysis of Phase 1D.
Re-simulate at κ=0 per (N,σ) for bootstrap σ_noise, use saved H_emp_rel for signal,
run 50-pt λ-sweep for glasso window widths at κ=0 and κ=0.5.
**Justification:**
  1. Previous diagnostic: binary threshold (H_EMP_THRESH=0.01) produced non-monotone
     κ_detect because: (a) relative ratio H_rel non-monotone at small κ (single-seed
     variance dominates), (b) glasso CV false positives at large N (smaller CV penalty
     crosses 1e-4 threshold). DEVIATION 010 approved.
  2. Rules out: threshold as the source of non-monotonicity (replaced by SNR=2 criterion
     with noise floor estimated from bootstrap).
  3. Tests: whether κ_detect(N) slope ≈ −1/4 with SNR criterion; whether glasso window
     provides a useful continuous diagnostic.
  4. Confirming: slope ≈ −0.25, clean collapse SNR vs κ·N^{1/4}.
  5. Refuting: slope far from −0.25, glasso window closes before κ=0.5.
**Decision:** go ahead (supervisor explicit: "Run directly and report")
**Result:** H_emp κ_detect slope = −0.304 ≈ −1/4 ✓ (N=5000→0.30, N=10000→0.25,
N=50000→0.15). Glasso window never closes (0.4–0.8 dec at κ=0.5) → cannot detect
solenoidal leakage. H_emp 43–158× faster. σ_noise formula validated. Phase 1D complete.

### Template:
```
### CP-[number] — [date] — Phase [N]
**Proposed action:** 
**Justification:**
  1. Previous diagnostic showed: 
  2. This rules out: 
  3. Proposed change tests: 
  4. Confirming outcome: 
  5. Refuting outcome: 
**Decision:** go ahead / rejected / modified
**Result:** 
```