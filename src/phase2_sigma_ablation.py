"""
Phase 2 sigma_n ablation at α=+1, κ=0, σ=0.5.

Trains three DSM score networks (SiLU MLP, task.md architecture) with
σ_n ∈ {0.02, 0.05, 0.1}.  Reports Hessian constancy diagnostics for
σ_n selection before Phase 2A sweep.

Selection criterion:
  lowest std(Ĥ[0,3]) / mean|Ĥ_diag| within-basin with val/L* < 1.05.
  Prefer smaller σ_n on ties (less basin-structure smearing).

Output: results/phase2/phase2_sigma_ablation.npz
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from sde import euler_maruyama

# ---------------------------------------------------------------------------
# Constants — all held fixed for this ablation
# ---------------------------------------------------------------------------
ALPHA         = 1.0
KAPPA         = 0.0
SIGMA         = 0.5
N_SAMPLES     = 10_000
SUBSAMPLE     = 600
N_STEPS       = N_SAMPLES * SUBSAMPLE   # 6,000,000
SIGMA_N_LIST  = [0.02, 0.05, 0.1]
N_EPOCHS      = 500
BATCH_SIZE    = 256
LR            = 1e-3
PATIENCE      = 50
VAL_FRAC      = 0.2
HIDDEN        = 64
DEPTH         = 2        # 2 hidden layers → task.md architecture
N_QUERY       = 500      # noisy query points for Hessian constancy
MU_THRESH     = 0.3      # |μ| > MU_THRESH defines "deep within basin"
N_BOOTSTRAP   = 1000
SEED          = 42

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model — SiLU MLP (task.md Phase 1B architecture)
# ---------------------------------------------------------------------------
class ScoreNetSiLU(nn.Module):
    """R^4 → R^4 score network.  2 hidden layers, 64 units, SiLU activation."""

    def __init__(self, sigma_n: float, hidden: int = HIDDEN, depth: int = DEPTH):
        super().__init__()
        self.sigma_n = sigma_n
        layers: list[nn.Module] = [nn.Linear(4, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DSM loss
# ---------------------------------------------------------------------------
def dsm_loss(model: ScoreNetSiLU, x_batch: torch.Tensor) -> torch.Tensor:
    """E[‖s_θ(x̃) + ε/σ_n‖²]  (Vincent 2011)."""
    sigma_n = model.sigma_n
    eps = torch.randn_like(x_batch)
    x_noisy = x_batch + sigma_n * eps
    score = model(x_noisy)
    target = -eps / sigma_n
    return ((score - target) ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Irreducible noise floor
# ---------------------------------------------------------------------------
def compute_L_star(X: np.ndarray, sigma_n: float) -> float:
    """
    Gaussian-approximation irreducible floor for the DSM loss used here.
    L* = d/σ_n² − trace((Σ̂ + σ_n²I)⁻¹)

    Derivation: at the optimal s_θ* = ∇log p_σ, the minimum loss equals
    d/σ_n² − E[‖∇log p_σ(x̃)‖²].  For Gaussian p_σ = N(0, Σ̂+σ_n²I),
    E[‖∇log p_σ‖²] = trace((Σ̂+σ_n²I)⁻¹).

    Note: at α=+1 the density is bimodal, so the Gaussian approximation
    underestimates L*.  The ratio val/L* is used only as a convergence
    indicator (network reached ≈optimal for its capacity), not as an
    absolute bound.
    """
    d = X.shape[1]
    Sigma = np.cov(X.T)
    M = Sigma + sigma_n ** 2 * np.eye(d)
    return float(d / sigma_n ** 2 - np.trace(np.linalg.inv(M)))


# ---------------------------------------------------------------------------
# ACF check
# ---------------------------------------------------------------------------
def lag1_acf(X: np.ndarray) -> np.ndarray:
    """ACF at lag 1 (one subsampled step) for each state variable."""
    acfs = []
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        var = (x ** 2).mean()
        acfs.append(float((x[:-1] * x[1:]).mean() / var) if var > 1e-12 else 0.0)
    return np.array(acfs)


# ---------------------------------------------------------------------------
# Training with early stopping
# ---------------------------------------------------------------------------
def train_network(
    X_train: np.ndarray,
    X_val: np.ndarray,
    sigma_n: float,
    device: torch.device,
    seed: int = SEED,
) -> tuple:
    """
    Train SiLU MLP with early stopping.

    Returns (model, train_losses, val_losses, best_val_loss, epochs_run).
    """
    torch.manual_seed(seed)
    model = ScoreNetSiLU(sigma_n=sigma_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val = float("inf")
    patience_ct = 0
    best_state: dict = {}
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(N_EPOCHS):
        model.train()
        total, n_b = 0.0, 0
        for (batch,) in loader:
            opt.zero_grad()
            loss = dsm_loss(model, batch)
            loss.backward()
            opt.step()
            total += loss.item()
            n_b += 1
        tr_loss = total / n_b

        model.eval()
        with torch.no_grad():
            val_loss = dsm_loss(model, X_vl).item()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"    Early stop at epoch {epoch + 1}/{N_EPOCHS}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1:4d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best_val={best_val:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, train_losses, val_losses, best_val, epoch + 1


# ---------------------------------------------------------------------------
# Hessian (Jacobian of score) computation
# ---------------------------------------------------------------------------
def jacobian_of_score(model: ScoreNetSiLU, X_query: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """
    Compute Hessian of log p = Jacobian of s_θ at each query point.

    H[i, j] = ∂s_i/∂x_j  (4×4 matrix per query point).

    Uses per-point autodiff loop.  For 500 points this takes < 2 min on GPU.
    Returns: (N_QUERY, 4, 4) numpy array.
    """
    model.eval()
    X_q = torch.tensor(X_query, dtype=torch.float32).to(device)
    H_list = []
    for i in range(len(X_q)):
        xi = X_q[i:i + 1].requires_grad_(True)
        si = model(xi).squeeze(0)
        rows = []
        for k in range(4):
            gk = torch.autograd.grad(si[k], xi,
                                     retain_graph=(k < 3),
                                     create_graph=False)[0]
            rows.append(gk.squeeze(0).detach().cpu())
        H_list.append(torch.stack(rows).numpy())   # (4, 4)
    return np.array(H_list)                         # (N_QUERY, 4, 4)


def hessian_diagnostics(
    model: ScoreNetSiLU,
    X_basin: np.ndarray,
    sigma_n: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    """
    Evaluate Hessian constancy and |Ĥ_ημ| over N_QUERY noisy query points
    drawn from X_basin (samples with |μ| > MU_THRESH).

    Query points: x̃ = x + σ_n ε,  x drawn from X_basin,  ε ~ N(0,I).

    Returns dict:
        H_samples          (N_QUERY, 4, 4)
        constancy_eta_mu   std(H[0,3]) / mean|H_diag|
        constancy_diag     mean over diag entries of std(H[i,i]) / mean|H_diag|
        H_eta_mu_mean      mean |H[0,3]|
        H_eta_mu_ci        95% bootstrap CI on mean |H[0,3]|
    """
    idx = rng.integers(0, len(X_basin), size=N_QUERY)
    x_clean = X_basin[idx]
    eps = rng.standard_normal(x_clean.shape)
    x_noisy = x_clean + sigma_n * eps          # (N_QUERY, 4)

    t0 = time.time()
    H = jacobian_of_score(model, x_noisy, device)    # (N_QUERY, 4, 4)
    print(f"    Jacobian ({N_QUERY} pts) in {time.time()-t0:.1f}s")

    diag = np.diagonal(H, axis1=1, axis2=2)           # (N_QUERY, 4)
    mean_abs_diag = np.mean(np.abs(diag))

    constancy_eta_mu = float(np.std(H[:, 0, 3]) / (mean_abs_diag + 1e-12))
    constancy_diag   = float(
        np.mean(np.std(diag, axis=0) / (mean_abs_diag + 1e-12))
    )

    abs_H_em = np.abs(H[:, 0, 3])
    H_em_mean = float(abs_H_em.mean())
    boots = np.array([
        rng.choice(abs_H_em, size=N_QUERY, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "H_samples":        H,
        "constancy_eta_mu": constancy_eta_mu,
        "constancy_diag":   constancy_diag,
        "H_eta_mu_mean":    H_em_mean,
        "H_eta_mu_ci":      np.array([lo, hi]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng    = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\n{'='*65}")
    print(f"Phase 2 σ_n ablation  α={ALPHA}  κ={KAPPA}  σ={SIGMA}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Simulate
    # ------------------------------------------------------------------
    print(f"\n[1] Simulation: {N_STEPS:,} steps, subsample={SUBSAMPLE} → {N_SAMPLES} samples")
    t0 = time.time()
    X, n_cross = euler_maruyama(
        alpha=ALPHA, kappa=KAPPA, sigma=SIGMA,
        n_steps=N_STEPS, subsample=SUBSAMPLE, seed=SEED,
    )
    print(f"    Done in {time.time()-t0:.1f}s   basin crossings: {n_cross}")
    if n_cross < 20:
        print(f"    *** WARNING: only {n_cross} crossings — mixing may be insufficient ***")
    else:
        print(f"    Crossing check: PASS (≥ 20)")

    # ACF check
    acf = lag1_acf(X)
    state_names = ["η", "s", "a", "μ"]
    print(f"\n    Lag-1 ACF (subsampled, threshold < 0.05):")
    for j, nm in enumerate(state_names):
        flag = "PASS" if abs(acf[j]) < 0.05 else "FAIL"
        print(f"      {nm}: {acf[j]:.4f}  {flag}")
    max_acf = float(np.max(np.abs(acf)))
    acf_ok = max_acf < 0.05
    if acf_ok:
        print(f"    ACF check: PASS (max={max_acf:.4f})")
    else:
        print(f"    ACF check: FAIL (max={max_acf:.4f})")
        print(f"    Supervisor must decide on subsample before Phase 2A go-ahead.")
        print(f"    Continuing ablation with current samples.")

    # Basin balance
    mu = X[:, 3]
    frac_pos = float((mu > 0).mean())
    frac_neg = float((mu < 0).mean())
    print(f"\n    Basin fractions: μ>0={frac_pos:.3f}  μ<0={frac_neg:.3f}")

    mask_deep = np.abs(mu) > MU_THRESH
    X_basin   = X[mask_deep]
    print(f"    Samples |μ|>{MU_THRESH}: {len(X_basin)} ({100*len(X_basin)/len(X):.1f}%)")
    if len(X_basin) < N_QUERY:
        raise RuntimeError(
            f"Only {len(X_basin)} deep-basin samples but need {N_QUERY} for Hessian check."
        )

    # ------------------------------------------------------------------
    # 2. Train/val split
    # ------------------------------------------------------------------
    n_val   = int(N_SAMPLES * VAL_FRAC)
    n_train = N_SAMPLES - n_val
    perm    = rng.permutation(N_SAMPLES)
    X_train = X[perm[:n_train]]
    X_val   = X[perm[n_train:]]
    print(f"\n[2] Train/val split: {n_train} / {n_val}")

    # ------------------------------------------------------------------
    # 3. Train each network and compute diagnostics
    # ------------------------------------------------------------------
    results: dict = {}

    for i_sn, sigma_n in enumerate(SIGMA_N_LIST):
        print(f"\n{'='*65}")
        print(f"σ_n = {sigma_n}")
        print(f"{'='*65}")

        L_star = compute_L_star(X, sigma_n)
        print(f"  L* (Gaussian approx) = {L_star:.4f}")

        t0 = time.time()
        model, tr_loss, vl_loss, best_val, epochs = train_network(
            X_train, X_val, sigma_n, device, seed=SEED
        )
        print(f"  Training: {time.time()-t0:.1f}s  ({epochs} epochs)")
        print(f"  best val loss = {best_val:.5f}")
        val_ratio = best_val / L_star
        flag_conv = "PASS" if val_ratio < 1.05 else "FAIL"
        print(f"  val/L* = {val_ratio:.4f}  (target < 1.05) → {flag_conv}")

        # Hessian diagnostics with independent rng per σ_n
        hess_rng = np.random.default_rng(SEED + 200 + i_sn)
        print(f"  Hessian constancy ({N_QUERY} query pts from |μ|>{MU_THRESH} basin):")
        diag = hessian_diagnostics(model, X_basin, sigma_n, hess_rng, device)

        print(f"    std(Ĥ[0,3]) / mean|Ĥ_diag| = {diag['constancy_eta_mu']:.4f}")
        print(f"    std(Ĥ_diag) / mean|Ĥ_diag|  = {diag['constancy_diag']:.4f}")
        ci = diag["H_eta_mu_ci"]
        print(f"    |Ĥ_ημ| mean = {diag['H_eta_mu_mean']:.5f}"
              f"  95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]")

        results[sigma_n] = {
            "L_star":          L_star,
            "best_val_loss":   best_val,
            "val_over_L_star": val_ratio,
            "epochs_trained":  epochs,
            "train_losses":    np.array(tr_loss),
            "val_losses":      np.array(vl_loss),
            **diag,
        }

    # ------------------------------------------------------------------
    # 4. Summary table and selection
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    header = f"{'σ_n':>6}  {'val/L*':>8}  {'const[0,3]':>12}  "
    header += f"{'const[diag]':>12}  {'|Ĥ_ημ|':>10}  {'95% CI lo':>10}  {'95% CI hi':>10}"
    print(header)
    for sigma_n in SIGMA_N_LIST:
        r  = results[sigma_n]
        ci = r["H_eta_mu_ci"]
        print(
            f"{sigma_n:>6.2f}  {r['val_over_L_star']:>8.4f}  "
            f"{r['constancy_eta_mu']:>12.4f}  {r['constancy_diag']:>12.4f}  "
            f"{r['H_eta_mu_mean']:>10.5f}  {ci[0]:>10.5f}  {ci[1]:>10.5f}"
        )

    # Selection
    candidates = [
        (sn, results[sn]) for sn in SIGMA_N_LIST
        if results[sn]["val_over_L_star"] < 1.05
    ]
    if not candidates:
        print("\nWARNING: no σ_n achieves val/L* < 1.05 — selecting by constancy only.")
        candidates = [(sn, results[sn]) for sn in SIGMA_N_LIST]

    selected_sn = min(candidates, key=lambda x: (x[1]["constancy_eta_mu"], x[0]))[0]
    print(f"\nSelected σ_n = {selected_sn}  "
          f"(lowest within-basin Hessian constancy with val/L* < 1.05)")
    if selected_sn != 0.05:
        print(f"NOTE: differs from Phase 1B value (0.05) — log as DEVIATION before Phase 2A.")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase2_sigma_ablation.npz")

    save: dict = {
        "alpha": ALPHA, "kappa": KAPPA, "sigma": SIGMA,
        "n_samples": N_SAMPLES, "subsample": SUBSAMPLE,
        "n_crossings": n_cross,
        "acf_lag1": acf, "acf_ok": acf_ok,
        "frac_pos_basin": frac_pos, "frac_neg_basin": frac_neg,
        "n_basin_samples": len(X_basin),
        "sigma_n_list": np.array(SIGMA_N_LIST),
        "selected_sigma_n": selected_sn,
    }
    for sigma_n in SIGMA_N_LIST:
        r   = results[sigma_n]
        key = f"sn{sigma_n:.2f}".replace(".", "p")
        for field in [
            "L_star", "best_val_loss", "val_over_L_star", "epochs_trained",
            "constancy_eta_mu", "constancy_diag",
            "H_eta_mu_mean", "H_eta_mu_ci",
            "train_losses", "val_losses", "H_samples",
        ]:
            save[f"{key}_{field}"] = r[field]

    np.savez(out_path, **save)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
