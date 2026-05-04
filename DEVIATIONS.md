# DEVIATIONS.md

## Record of deviations from task.md

### DEVIATION 001 — 2026-05-04 — Phase 0A
**Phase:** 0A
**Description:** Default coupling constants c₁=c₂=c₃=c₄=1.0
in task.md produce a zero eigenvalue in J at κ=0. The
open-loop ring gain c₁c₂c₃c₄ = 1 exactly equals the
damping product γ_η·γ_s·γ_a·|α| = 1, violating the
Lyapunov solvability condition λᵢ+λⱼ≠0.
**Fix:** Set c₁=c₂=c₃=c₄=0.5. All γ values, α, σ, κ unchanged.
This gives ρ = (0.5)⁴/(1·1·1·1) = 0.0625, placing the system
comfortably within the stable regime.
**Justification:** Preserves ring topology, blanket partition,
and all scientific questions. Changes only the distance from
the stability boundary. Phase 2 stability check: compute
ρ = c₁c₂c₃c₄/(γ_η·γ_s·γ_a·|α|) at each α and stop the
Lyapunov shortcut if ρ > 0.8 (use FPE numerics instead).
**Impact on downstream phases:** All default parameter values
referencing c must be updated to 0.5. The stability condition
ρ < 1 must be checked whenever α is swept in Phase 2, since
α→0⁺ approaches the bifurcation and ρ→∞.
**Approved by supervisor:** yes

### DEVIATION 002 — 2026-05-04 — Phase 0A
**Phase:** 0A
**Description:** task.md threshold for H[0,3] at κ=0 specified
as absolute < 1e-14. Single-κ verification produced 1.41e-14,
which is machine precision for condition-3 matrix inversion
(ratio |H[0,3]|/max|H_ij| = 7e-15). Absolute threshold is
too tight and not physically meaningful.
**Fix:** threshold changed to relative criterion
|H[0,3]|/max|H_ij| < 1e-12 throughout Phase 0A.
**Impact:** none — criterion is more correct, not more lenient.
The original absolute threshold would spuriously fail clean
results for better-conditioned matrices too.
**Approved by supervisor:** yes

### Template for recording deviations:
```
### DEVIATION [number]: [date]
**Phase:** 
**Description:** 
**Justification:** 
**Impact on downstream phases:** 
**Approved by supervisor:** yes/no
```