# Markov Blanket Identification by Score Matching
## Research Plan — task.md

### Scientific question

Can data-driven estimation of the Hessian of log-density (via score matching or sparse precision estimation) correctly recover Markov blanket structure in a nonlinear stochastic dynamical system, and under what conditions does this statistical identification break down?

### The model system

A four-state stochastic differential equation modelling a minimal chemosensing cell.

**State vector:** x = (η, s, a, μ)ᵀ

- η: extracellular chemical concentration (external)
- s: membrane receptor state (sensory / blanket)
- a: membrane secretion rate (active / blanket)
- μ: intracellular concentration (internal)

**Dynamics:**

```
dη = (−γ_η · η  +  c₁ · a                     + κ · μ) dt  +  σ dW₁
ds = (−γ_s · s  +  c₂ · η                            ) dt  +  σ dW₂
da = (−γ_a · a  +  c₃ · μ                            ) dt  +  σ dW₃
dμ = (α·μ − μ³  +  c₄ · s                     − κ · η) dt  +  σ dW₄
```

**Parameter roles:**
- γ_η, γ_s, γ_a: linear relaxation rates (all positive; μ relaxation is nonlinear)
- c₁, c₂, c₃, c₄: coupling constants defining the ring η → s → μ → a → η
- α: bifurcation parameter. α < 0 → monostable; α > 0 → bistable internal state
- κ: solenoidal leak. Antisymmetric coupling (η,μ) that breaks the Markov blanket causally. κ = 0 → blanket intact
- σ: noise intensity (isotropic for simplicity)

**Jacobian structure at κ = 0:**
```
J = [ -γ_η    0     c₁      0        ]
    [  c₂   -γ_s    0       0        ]
    [  0      0    -γ_a     c₃       ]
    [  0     c₄     0     α - 3μ²   ]
```

J_ημ = J_μη = 0 by construction → Markov blanket {s, a} separates {μ} from {η}.

At κ > 0, J_ημ = κ and J_μη = −κ. The coupling is antisymmetric → purely solenoidal (contributes to Q, not to the symmetric dissipative part).

**Default parameter set:**
```
γ_η = γ_s = γ_a = 1.0
c₁ = c₂ = c₃ = c₄ = 1.0
σ = 0.5
α = −1.0 (linear regime default)
κ = 0.0 (blanket intact default)
dt = 0.01 (integration time step)
```

These are held constant unless a specific test varies one of them.

---

## Phase 0: Ground truth

### 0A — Lyapunov solution (linear regime)

**What:** For α < 0, linearise around x* = 0. The Jacobian is constant. Solve the continuous Lyapunov equation JΣ + ΣJᵀ = −D (where D = σ²I) for steady-state covariance Σ. Compute H = Σ⁻¹.

**Vary:** κ ∈ {0.0, 0.05, 0.1, 0.15, ..., 1.0}

**Outputs:**
1. H(κ) — the full 4×4 precision matrix as a function of κ
2. H_ημ(κ) — the specific entry that should be zero at κ=0
3. Condition number of Σ (stability check)
4. Eigenvalues of J (must all have negative real part for NESS to exist)

**Completion criteria:**
- At κ = 0: H_ημ = 0 to machine precision (< 1e-14)
- At κ > 0: H_ημ(κ) is a smooth function, growing from zero
- All eigenvalues of J have negative real part for all κ tested
- Σ is symmetric positive definite for all κ tested

### 0B — Fokker-Planck numerics (nonlinear regime)

**What:** Solve the 4D stationary Fokker-Planck equation numerically for the full nonlinear system. Obtain p_ss(x) on a grid. Compute log p_ss and its Hessian pointwise.

**Grid:** 25 points per dimension (390,625 total). Domain chosen adaptively: ±4σ_marginal for each state, where σ_marginal estimated from pilot simulation.

**Vary:** α ∈ {−1.0, −0.5, 0.0, 0.5, 1.0, 1.5, 2.0} at κ = 0

**Outputs:**
1. p_ss(x) on the grid
2. H_ημ(x) evaluated at 1000 randomly sampled grid points
3. max|H_ημ(x)| and mean|H_ημ(x)| across the grid
4. Marginal density of μ (to verify unimodal vs bimodal transition)

**Completion criteria:**
- Density integrates to 1.0 ± 0.01
- At κ = 0, α < 0: max|H_ημ(x)| < 1e-6 across grid
- At κ = 0, α > 0: report whether H_ημ(x) = 0 everywhere or only in basins
- Marginal of μ is unimodal for α < 0, bimodal for α > α_crit (report α_crit)

**Fallback:** If 4D FPE is computationally prohibitive (memory > 32GB or time > 2 hours), replace with long-trajectory Monte Carlo estimation of the joint density using kernel density estimation with bandwidth chosen by cross-validation. Document this as a deviation.

---

## Phase 1: Linear regime estimation

### 1A — Simulation infrastructure

**What:** Implement Euler-Maruyama integrator for the SDE system. Validate against the Lyapunov solution from Phase 0A.

**Validation:**
1. Simulate 10⁶ steps, subsample every 100 steps → 10,000 samples
2. Compute sample covariance Σ̂
3. Compare ‖Σ̂ − Σ_true‖_F / ‖Σ_true‖_F — must be < 0.05
4. Check autocorrelation at lag = 100 steps — must be < 0.05 for all states (confirms approximate independence of subsampled snapshots)

**Completion criteria:**
- Relative Frobenius error < 0.05
- Autocorrelation at subsampling lag < 0.05
- Runtime for 10⁶ steps < 30 seconds

### 1B — Score matching

**What:** Train a score network s_θ(x) ≈ ∇ log p(x) using denoising score matching. Compute Ĥ(x) = −∇_x s_θ(x) by autodifferentiation. Since the system is linear, Ĥ should be approximately constant; average over samples to get Ĥ_mean.

**Architecture:** MLP, 2 hidden layers, 64 units, SiLU activation. Input dim 4, output dim 4.

**Training:** Denoising score matching (Vincent 2011).
- Noise scale σ_n: sweep {0.01, 0.05, 0.1, 0.5} and select by validation loss on held-out 20% of data
- Optimiser: Adam, lr = 1e-3, batch size 256
- Training epochs: up to 500, early stopping on validation loss with patience 50

**Measurements:**
1. ‖Ĥ_mean − H_true‖_F / ‖H_true‖_F (overall recovery error)
2. |Ĥ_ημ| (the blanket-critical entry)
3. std(Ĥ_ημ(x)) across samples (is it really constant?)
4. Repeat for κ ∈ {0, 0.1, 0.2, ..., 1.0}
5. Repeat for N ∈ {1000, 5000, 10000, 50000}

**Completion criteria:**
- At κ = 0: |Ĥ_ημ| < 0.05 (with true nonzero entries being O(1))
- Recovery error < 0.1 at N = 10000
- |Ĥ_ημ(κ)| tracks H_ημ(κ) from Phase 0A with correlation > 0.95
- Clear separation between noise floor (κ=0) and signal (κ>0) at N ≥ 5000

### 1C — Graphical lasso

**What:** Fit sparse precision matrix using graphical lasso. Sweep regularisation parameter λ. Identify the blanket window.

**Method:** sklearn.covariance.GraphicalLassoCV for automatic λ selection, plus manual sweep λ ∈ logspace(−3, 1, 50) for the window analysis.

**Measurements:**
1. At each λ: binary vector of which H entries are zeroed
2. Blanket window: range of λ where H_ημ = 0 but all other entries nonzero
3. Width of blanket window Δλ as a function of κ
4. False positive rate (true nonzeros incorrectly zeroed) and false negative rate (true zeros incorrectly nonzero) at the CV-selected λ

**Completion criteria:**
- At κ = 0: blanket window width Δλ > 0.5 decades (clear, wide window)
- At κ = 0, CV-selected λ: correctly identifies H_ημ = 0
- Blanket window width decreases monotonically with κ
- Cross-check: graphical lasso H agrees with score matching Ĥ_mean to within 10%

### 1D — Method comparison

**What:** Direct comparison of score matching vs graphical lasso in the linear regime.

**Measurements:**
1. Detection sensitivity κ_detect for each method at each N
2. Computational cost (wall time) for each method
3. Robustness to σ: repeat at σ ∈ {0.1, 0.5, 1.0, 2.0}

**Completion criteria:**
- Quantitative table of κ_detect(N, method)
- Clear statement of which method is superior in the linear regime and why

---

## Phase 2: Nonlinear regime

### 2A — Bifurcation sweep

**What:** Sweep α from −1 to +3 in steps of 0.25, holding κ = 0. At each α, generate 10,000 steady-state samples and apply both estimation methods.

**Key challenge:** For α > 0, the system is bistable and the mixing time is long. Must verify that the simulation has visited both basins adequately.

**Basin mixing diagnostic:** 
- Count transitions between basins (μ crossing zero) per trajectory
- Require ≥ 20 transitions in the sampling window
- If mixing is insufficient, increase simulation length (not σ)
- Report effective sample size after accounting for autocorrelation

**Measurements:**
1. Score matching: |Ĥ_ημ| as a function of α
2. Graphical lasso at CV-selected λ: |Ĥ_ημ| as a function of α
3. Per-basin analysis: restrict samples to μ > 0 and μ < 0 separately, estimate H in each basin
4. Kurtosis of μ marginal (quantifies departure from Gaussianity)

**Completion criteria:**
- At α < 0: both methods correctly find blanket (consistent with Phase 1)
- At α > 0, κ = 0: per-basin H_ημ ≈ 0 even if global H_ημ ≠ 0
- Clear identification of α_crit where global graphical lasso fails
- Score matching performance characterised across the full sweep

### 2B — Mixture graphical model (conditional on 2A results)

**Only run if:** Phase 2A shows that single graphical lasso fails in bimodal regime but per-basin analysis succeeds.

**What:** Fit a two-component Gaussian mixture model, then apply graphical lasso to each component's precision matrix separately.

**Method:**
1. Fit GMM with 2 components using EM
2. Assign each sample to its most likely component
3. Apply graphical lasso within each component
4. Check H_ημ = 0 within each component

**Completion criteria:**
- Mixture graphical lasso correctly identifies blanket where single glasso failed
- Component assignments agree with sign of μ (the natural basin indicator) for > 90% of samples

---

## Phase 3: Solenoidal diagnostic

### 3A — Jacobian estimation

**What:** Estimate J from trajectory data using short-time linear regression.

**Method:** At each time step, compute (x(t+dt) − x(t))/dt and regress against x(t). For the linear system, this gives a consistent estimate of J. Use the full trajectory (not subsampled) with appropriate standard errors.

**Validation:** Compare Ĵ to J_true from Phase 0. Require ‖Ĵ − J_true‖_F / ‖J_true‖_F < 0.05.

**Completion criteria:**
- Relative Frobenius error < 0.05
- Correct identification of zero/nonzero pattern in J at κ = 0

### 3B — J-vs-H sparsity comparison

**What:** For each κ ∈ {0, 0.05, 0.1, ..., 2.0}, estimate both J (from trajectories) and H (from steady-state samples). Compare their sparsity patterns.

**Measurements:**
1. κ_J = 0⁺ (the Jacobian zero breaks immediately with any κ > 0)
2. κ_H: the smallest κ at which |Ĥ_ημ| is statistically distinguishable from the κ=0 noise floor (use bootstrap 95% CI)
3. The gap κ_H − κ_J: measures how robust the statistical blanket is to solenoidal perturbation
4. Ratio |Ĥ_ημ(κ)| / |Ĵ_ημ(κ)|: does statistical coupling grow faster or slower than causal coupling?

**Completion criteria:**
- κ_H > κ_J (the statistical blanket is more robust than the causal one)
- Quantitative estimate of κ_H with confidence interval
- Clear visualisation of the sparsity disagreement as a function of κ

### 3C — Blanket phase transition (nonlinear, conditional on 3B)

**Only run if:** Phase 3B succeeds and Phase 2A shows interesting nonlinear behavior.

**What:** Repeat 3B in the nonlinear regime (α = 2.0). Sweep κ with fine resolution near where the blanket breaks.

**Hypothesis:** H_ημ(κ) may have a discontinuity or sharp kink at a critical κ* in the nonlinear case, corresponding to a topological change in the steady-state density.

**Measurements:**
1. H_ημ(κ) at fine resolution (Δκ = 0.02) near the expected transition
2. Numerical derivative dH_ημ/dκ — look for divergence
3. Number of modes in the full 4D density as a function of κ (use persistent homology or simpler mode-counting)

**Completion criteria:**
- Clear statement: phase transition found / not found
- If found: estimate of κ* with uncertainty, and characterisation of what changes topologically

---

## Phase 4: Temporal resolution

### 4A — Subsampling sweep

**What:** At fixed κ = 0.5 (blanket broken causally but not dominantly), simulate a very long trajectory at dt = 0.001. Subsample at Δt ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}. Estimate H at each Δt.

**Measurements:**
1. |Ĥ_ημ(Δt)| — the blanket entry as a function of temporal resolution
2. Δt* where |Ĥ_ημ| becomes statistically indistinguishable from zero
3. Compare Δt* to τ_relax = 1/min(Re(λ_J)) where λ_J are eigenvalues of J

**Control:** Repeat at κ = 0. |Ĥ_ημ| should be ≈ 0 at all Δt.

**Completion criteria:**
- |Ĥ_ημ| decreases with increasing Δt at κ = 0.5
- Δt* and τ_relax are within one order of magnitude
- Control (κ=0) shows no Δt dependence

---

## Phase 5: Synthesis

### 5A — Summary figures

Produce the following publication-quality figures:

1. **System diagram** — the chemosensing cell with coupling structure
2. **Ground truth curves** — H_ημ(κ) from Lyapunov solution
3. **Score matching recovery** — estimated vs true H_ημ across κ, with error bars across N
4. **Graphical lasso blanket window** — Δλ vs κ heatmap
5. **Bifurcation diagram** — blanket detection accuracy vs α, both methods
6. **Solenoidal diagnostic** — J sparsity vs H sparsity as a function of κ, with κ_H marked
7. **Temporal resolution** — |Ĥ_ημ| vs Δt, with τ_relax marked
8. **Phase transition** (conditional) — H_ημ(κ) in the nonlinear regime, near κ*

### 5B — Conclusions document

Structured summary:
- What works: conditions under which score matching / glasso correctly identify blankets
- What fails: conditions under which they don't, and why
- What's new: any findings that weren't predicted (especially the phase transition and temporal resolution results)
- What's next: open questions this study raises

---

## Implementation notes

**Language:** Python 3.10+
**Core dependencies:** numpy, scipy, torch (for score matching), scikit-learn (for graphical lasso), matplotlib
**No exotic dependencies.** Everything must run on a single machine with ≤ 32GB RAM.

**Code structure:**
```
src/
  sde.py           — SDE integrator and parameter management
  lyapunov.py      — Analytical Lyapunov solver for Phase 0A
  fokker_planck.py  — Numerical FPE solver for Phase 0B
  score_matching.py — Score network and training loop
  glasso.py         — Graphical lasso wrapper and blanket window analysis
  jacobian_est.py   — Jacobian estimation from trajectories
  diagnostics.py    — All diagnostic functions and validators
  plotting.py       — Figure generation
tests/
  test_sde.py       — SDE integrator tests
  test_lyapunov.py  — Lyapunov solution tests
  test_score.py     — Score matching tests
  test_glasso.py    — Graphical lasso tests
results/
  phase0/           — Ground truth data
  phase1/           — Linear regime results
  phase2/           — Nonlinear regime results
  phase3/           — Solenoidal diagnostic results
  phase4/           — Temporal resolution results
  figures/          — Publication figures
```

**Random seeds:** All experiments use seed = 42 for the main run. Sensitivity to seed tested by repeating key results at seeds {0, 1, 2, 3, 4} and reporting mean ± std.

**Data persistence:** All intermediate results saved as .npz files with full parameter metadata. No result should require re-running a simulation to reproduce.