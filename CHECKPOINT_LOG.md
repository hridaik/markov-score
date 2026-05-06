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