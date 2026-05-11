# Paper notes — findings worth writing up formally

## 1. Linear score network = sample precision matrix (provable)
DSM optimality condition for W in nn.Linear(d,d,bias=False):
  W* = −Σ̂_σ⁻¹ = −(Σ̂_x + σ_n²I)⁻¹
Proof: dL/dW = 0 → W* E[x̃x̃ᵀ] = −I → W* = −Σ̂_σ⁻¹.
Implication: for Gaussian p, score matching offers nothing
beyond direct covariance inversion. Scientific value of score
matching lies in the non-Gaussian regime.

## 1b. Phase 1D scope: score matching excluded from comparison
In the linear regime (α<0, Gaussian p), the linear score network W* = −Σ̂_σ⁻¹
is identical to direct covariance inversion (up to the σ_n² regularisation bias of ~2.4%).
Phase 1D comparison is therefore raw precision matrix (H_emp) vs graphical lasso only.
Running score matching separately would add no new information and would be misleading
(it would appear to be a third method when it is the same estimator by a different route).
The meaningful comparison in Phase 2 (nonlinear) is where score matching genuinely differs
from both: the density is non-Gaussian, the precision matrix breaks down globally, and
the score network can learn position-dependent Hessian structure unavailable to H_emp.

## 1c. Graphical lasso cannot detect solenoidal leakage
Graphical lasso regularization drives H[0,3]→0 by design. The blanket
window [λ_low, λ_high] remains positive (0.4–0.8 decades) even at κ=0.5
where the true H[0,3]=0.474. The lasso zeros out the large off-diagonal
entry at the right penalty — indistinguishable from the κ=0 case where
the true entry IS zero. κ_detect for graphical lasso is undefined: the
window narrows but never closes.
Implication: graphical lasso is the wrong tool for solenoidal leakage
detection. Appropriate for recovering sparse graph structure; inappropriate
for detecting violations of conditional independence from solenoidal flow.

## 1d. H_emp κ_detect ∝ N^{−0.30} ≈ N^{−1/4}
κ_detect (σ=0.5, SNR=2): N=5000→0.30, N=10000→0.25, N=50000→0.15.
Log-log slope = −0.304 over N∈{5000,10000,50000}.
Predicted: N^{-1/4} from O(κ²) signal + O(1/√N) noise floor.
Empirical H_emp[0,3] ~ κ^{1.2} in the observable range (not κ^2)
due to nonlinear self-consistency; effective scaling κ_detect ~ N^{-0.29}.
N^{-1/4} prediction holds approximately despite observable-range exponent
differing from the asymptotic κ^2. N=1000 requires κ>0.5 for detection.

## 2. O(κ²) scaling of H_ημ from Z₂ symmetry
At κ=0 with all γ equal and all c equal: Σ₀[0,0] = Σ₀[3,3]
(Z₂ symmetry η↔μ). First-order Lyapunov correction Σ₁[0,3]
has forcing term ∝ (Σ₀[3,3]−Σ₀[0,0]) = 0. So H_ημ = O(κ²),
not O(κ). Confirmed numerically (power-law fit κ^1.93).
Implication: symmetric systems are harder to detect solenoidal
leakage in — detection threshold κ_H scales as N^{-1/4}
rather than N^{-1/2}.

## 3. Nonlinear self-consistency: H_emp ≠ H_lyap for μ entries
Cubic restoring force stiffens effective μ dynamics beyond
linearisation. Self-consistency equation:
2(1+3E[μ²])Var(μ) = σ² → Var(μ) ≈ 0.097 vs Lyapunov 0.125.
Propagates to off-diagonal: H_emp[0,3] ≈ 3-5× H_lyap[0,3]
at large κ. H_emp is the correct ground truth; H_lyap is the
linearised approximation.

## 4. Blanket is basin-specific in nonlinear systems
Global precision matrix fails at bifurcation (α=0, H[3,3]=4.89,
large μ variance) and in bistable regime (bimodal density,
global H encodes inter-basin separation not within-basin CI).
Per-basin analysis recovers correct blanket structure.
Implication: ergodic within-basin sampling sufficient;
global ergodicity not required for blanket identification.

## 5. Graphical lasso cannot detect blanket violation onset
The L1 penalty forces H[0,3]→0 at some λ for any κ, making
glasso structurally unable to distinguish true zero from
penalised nonzero. Window width narrows (1.5→0.6 decades
from κ=0 to κ=0.5) but never closes.
Distinction: glasso identifies blanket structure (which
entries are nonzero at a given κ); H_emp with SNR
thresholding detects blanket violation onset.
These serve different purposes and should not be
compared on the same task.
H_emp is 43–158× faster for the detection task.

## 6. N^{-1/4} scaling confirmed empirically
Log-log slope −0.304 vs predicted −0.25. Effective
signal is κ^{1.2} in the observable range (not κ^2)
due to higher-order nonlinear terms; asymptotic O(κ²)
holds only for κ ≪ 0.1 where SNR < 1. The N^{-1/4}
scaling survives this impurity. σ-dependence mild:
SNR ~ σ^{0.8} (nonlinear correction; theory predicts
σ-independence in pure linear regime).

## 7. Per-basin H_emp requires ΔU/σ² ≫ 1 (minimum threshold ~ 0.5–1)
Per-basin H_emp is only interpretable when each basin is statistically
isolated from the other. The diagnostic is the barrier-to-noise ratio
ΔU/σ² where ΔU is the potential barrier between the two wells.
At α=0.25, ΔU/σ²=0.016 — the two basins are completely mixed and
the per-basin split by sign(μ) produces contaminated samples where
roughly half the "μ>0" points are visiting the μ<0 basin.
At α=0.50, ΔU/σ²=0.172 — isolation is marginal but sufficient for
a valid H_emp estimate (constancy passes, H[0,3]/max < 0.025).
Practical threshold: require ΔU/σ² ≥ 0.5 for per-basin H_emp to be
meaningful. Below this, the per-basin approach cannot be salvaged by
larger N — the contamination is structural (trajectory mixes basins
on every step), not statistical.
Implication for papers using per-basin precision estimates in bistable
systems: always report ΔU/σ² and confirm it exceeds this threshold.

## 8. Large-α failure mode: per-basin H_emp requires N_basin ~ 10⁶
In the large-α regime (α≥0.75), the wells are deep (ΔU/σ² ≫ 1) but
the two basins diverge from each other in their per-basin H_emp by
up to 15σ of sampling noise. Z₂ symmetry proves the two basins are
statistically identical in the N→∞ limit, so the divergence is a
finite-sample trajectory fluctuation: a single N=10,000 trajectory
may visit basin A 5000 times and basin B 5000 times, but the within-
basin samples are not iid — they are serially correlated. Effective
independent sample count per basin ≈ N_basin / τ_mix_basin, where
τ_mix_basin ≫ τ_mix_global at large α.
Estimate: achieving 1σ agreement between basins requires
N_basin ~ τ_mix_basin × (σ_H/ΔH_target)² ~ 10⁶.
Consequence: per-basin H_emp is not a practical primary diagnostic
for large α. The score network MLP does not share this failure mode:
it learns a continuous score function from all N=10,000 points
simultaneously (most near the basin centers), then evaluates the
Hessian at within-basin query points analytically. The inversion step
(which is where sample covariance fails at small per-basin N) is
never performed.
