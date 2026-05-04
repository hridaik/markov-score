# CONTEXT.md — Theoretical Background and Design Rationale

## Why this system

The chemosensing cell model was chosen because it has three 
properties simultaneously: a physically meaningful Markov blanket 
(the membrane), tunable nonlinearity (the bistable internal switch), 
and a single-parameter control over solenoidal cross-boundary 
coupling (κ). No simpler system has all three.

## The solenoidal coupling κ — what it is and isn't

The parameter κ adds +κ to J_ημ and −κ to J_μη. This antisymmetric 
contribution enters the solenoidal operator Q in the Helmholtz 
decomposition J = (Q − Γ)H, where Γ is the diffusion and H is 
the precision.

Crucially: κ does NOT break the system's stability or change 
the diffusion. It creates probability currents that circulate 
between the internal and external states without following 
the free energy gradient. Physically, it's like a pump that 
moves probability around on level sets of the density — it 
doesn't change where the system "wants" to be, but it changes 
the routes probability takes to get there.

The key insight from the original document: these cross-boundary 
currents can create statistical dependencies (nonzero H_ημ) even 
though they are not gradient-driven. The steady-state density 
is the fixed point of the full dynamics including solenoidal flow, 
so the solenoidal part affects the shape of p_ss even though it 
doesn't affect its existence.

## The ring coupling creates solenoidal flow at κ=0

Important subtlety: the system η → s → μ → a → η is a directed 
ring. Even at κ=0, the system is NOT at detailed balance. There 
are already solenoidal currents circulating around the ring. 
The difference is that these currents stay within each "side" 
of the blanket — they don't cross the {s,a} boundary. The κ 
parameter specifically creates cross-boundary solenoidal flow.

This means we should NOT expect the Helmholtz Q to be zero at 
κ=0. We should only expect the cross-boundary entries of Q to 
be zero (or, more precisely, for Q to respect the partition 
structure).

## Why the bifurcation matters

At α < 0, the internal state μ has a single stable fixed point. 
The steady-state density is approximately Gaussian (exactly 
Gaussian in the linearised system). All estimation methods are 
in their comfort zone.

At α > 0, the internal state is bistable (two wells separated 
by a barrier). The steady-state density is bimodal in μ. This 
matters for two distinct reasons:

1. **Estimation difficulty:** The graphical lasso assumes a 
   single Gaussian, which cannot represent a bimodal distribution. 
   The score network must learn a score function with complex 
   structure near the separatrix (where ∇ log p switches 
   direction sharply).

2. **Conceptual question:** Even with perfect estimation, is 
   H_ημ(x) = 0 everywhere in state space, or only within each 
   basin? The Hammersley-Clifford theorem requires 
   ∂² log p / ∂η ∂μ = 0 for ALL x, not just on average. Near 
   the separatrix, log p has high curvature and the Hessian 
   could behave differently. This is an open theoretical 
   question that the numerics in Phase 0B will address.

## The temporal resolution hypothesis

The claim is that coarse-graining in time (subsampling the 
trajectory) naturally suppresses solenoidal effects and makes 
the estimated statistical structure more "causal."

The physical argument: solenoidal currents circulate probability 
without changing the marginal density. They operate on a 
timescale set by the circulation period. The dissipative dynamics 
(the gradient part) operate on a timescale set by the relaxation 
rates γ. If the solenoidal timescale is faster than the 
dissipative timescale, then subsampling at the dissipative 
timescale averages out the solenoidal contribution.

This is NOT guaranteed to be true — it depends on the specific 
system. The κ coupling could have arbitrarily slow solenoidal 
circulation if κ is small. Phase 4 tests this empirically.

## O(κ²) scaling of H[0,3] — Phase 0A finding

The Phase 0A sweep showed |H[0,3]| ~ κ^1.93 (≈ κ²), not the 
O(κ) scaling naive perturbation theory might suggest.

Root cause: the default parameter set has exact Z₂ symmetry 
η↔μ, s↔a (all γ equal, all c equal). Under this symmetry, 
Σ₀[0,0] = Σ₀[3,3]. The first-order Lyapunov correction Σ₁ 
satisfies a Lyapunov equation with forcing term proportional 
to (Σ₀[3,3] − Σ₀[0,0]), which is identically zero. So 
Σ₁[0,3] = 0 and H[0,3] = O(κ²).

Consequences for downstream phases:
- The noise floor in estimation is set by the estimator's 
  variance, which scales as O(1/√N). The signal grows as 
  O(κ²). Detection threshold κ_H satisfies κ_H² ~ 1/√N, 
  giving κ_H ~ N^{-1/4} rather than the O(N^{-1/2}) scaling 
  expected for a linear signal. Phase 1B and 1C must be 
  interpreted with this in mind.
- The gap κ_H − κ_J (where κ_J = 0⁺ since J[0,3] = κ 
  linearly) will be governed by the quadratic regime. 
  At N=10,000, expect κ_H ~ 0.1 or larger.
- This scaling is specific to the symmetric parameter set. 
  Any future asymmetric parameter variant would restore O(κ) 
  scaling and shrink κ_H substantially.

This is not a pathology. It is correct physics. If any 
estimation result shows H[0,3] growing faster than κ², 
the excess is an estimation artifact.

## Decisions not yet made

- **Score matching variant:** task.md specifies denoising score 
  matching (Vincent 2011). If this fails to converge in the 
  nonlinear regime, sliced score matching (Song et al. 2019) 
  is the fallback. The choice should be based on convergence 
  diagnostics, not preference.
  
- **FPE solver:** If the 4D grid is prohibitive, Monte Carlo 
  KDE is the fallback. The choice should be based on the 
  memory/time constraints encountered in Phase 0B, not 
  anticipated in advance.

- **Phase transition detection:** How to detect the blanket 
  phase transition in Phase 3C is underdetermined. Persistent 
  homology is one option; simple mode-counting via gradient 
  flow is simpler. Decision deferred to Phase 3C.