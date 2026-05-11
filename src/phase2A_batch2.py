"""
Phase 2A Batch 2 — bifurcation point and early bistable regime
α ∈ {0.0, 0.25, 0.50, 0.75, 1.00}, κ=0, σ=0.5

α = 0.0  (bifurcation point)
  - Global ACF criterion — DEVIATION 012 does not apply; system is unimodal.
  - Subsample searched upward through [600, 1200, 2400, 4800] until ACF[μ] < 0.05.
    If none succeeds, largest subsample is used with a noted WARNING.
  - Large |Ĥ_ημ| from estimation noise on a flat μ marginal is NOT a failure.
  - GLasso window may narrow or close — report as-is.

α > 0  (bistable, per-basin — DEVIATION 012)
  - Pilot must confirm ≥20 basin crossings AND within-basin ACF < 0.05.
  - If per-basin ACF passes but global fails, proceed (documented expected behavior).
  - Per-basin H_emp (split by sign(μ)) is the primary estimator.
  - Global H_emp reported for continuity but not the scientific target.
  - Score network constancy check uses deep-basin query pool (|μ| > 0.3).
  - GLasso applied to global data; result may be unreliable (non-Gaussian density).

GLasso λ grid: logspace(−4, 1, 50)  — extended from Batch 1's logspace(−3, 1, 30).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.covariance import GraphicalLasso
import time

from sde import euler_maruyama

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_LIST        = [0.0, 0.25, 0.50, 0.75, 1.00]
KAPPA             = 0.0
SIGMA             = 0.5
SIGMA_N           = 0.05          # confirmed by ablation 2026-05-08
N_PILOT           = 1_000
N_FULL            = 10_000
SUBSAMPLE_DEFAULT = 600
SUBSAMPLE_SEARCH  = [600, 1200, 2400, 4800]   # α=0 pilot search only
N_EPOCHS          = 500
BATCH_SIZE        = 256
LR                = 1e-3
PATIENCE          = 50
HIDDEN            = 64
DEPTH             = 2
N_QUERY           = 500           # noisy query points for Hessian constancy
N_LAMBDA          = 50
LAMBDA_GRID       = np.logspace(-4, 1, N_LAMBDA)
ZERO_THRESH       = 1e-8
N_BOOTSTRAP       = 1_000
VAL_FRAC          = 0.2
SEED              = 42
MIN_CROSSINGS     = 20
MU_BASIN_THRESH   = 0.3           # |μ| > this defines deep-basin region
MIN_PER_BASIN     = 100           # minimum samples for reliable per-basin H_emp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_NAMES  = ["η", "s", "a", "μ"]


# ---------------------------------------------------------------------------
# Score network — SiLU MLP, task.md Phase 2 architecture
# ---------------------------------------------------------------------------
class ScoreNetSiLU(nn.Module):
    """R^4 → R^4, 2 hidden layers, 64 units, SiLU."""

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


def dsm_loss(model: ScoreNetSiLU, x_batch: torch.Tensor) -> torch.Tensor:
    sn = model.sigma_n
    eps = torch.randn_like(x_batch)
    return ((model(x_batch + sn * eps) + eps / sn) ** 2).sum(1).mean()


def train_network(
    X_tr: np.ndarray,
    X_vl: np.ndarray,
    sigma_n: float,
    device: torch.device,
    seed: int = SEED,
) -> tuple:
    torch.manual_seed(seed)
    model  = ScoreNetSiLU(sigma_n).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr    = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Xvl    = torch.tensor(X_vl, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xtr), batch_size=BATCH_SIZE, shuffle=True)

    best_val, patience_ct, best_state = float("inf"), 0, {}
    tr_hist: list[float] = []
    vl_hist: list[float] = []

    for epoch in range(N_EPOCHS):
        model.train()
        total, nb = 0.0, 0
        for (b,) in loader:
            opt.zero_grad()
            loss = dsm_loss(model, b)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vl = dsm_loss(model, Xvl).item()
        tr_hist.append(total / nb)
        vl_hist.append(vl)

        if vl < best_val:
            best_val = vl
            patience_ct = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"    early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"    ep {epoch+1:3d}  tr={tr_hist[-1]:.4f}  "
                  f"vl={vl:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, np.array(tr_hist), np.array(vl_hist), best_val, epoch + 1


# ---------------------------------------------------------------------------
# Hessian (Jacobian of score) computation
# ---------------------------------------------------------------------------
def compute_hessians(
    model: ScoreNetSiLU, X_query: np.ndarray, device: torch.device
) -> np.ndarray:
    """Returns (N, 4, 4) array of Jacobians of s_θ at each query point."""
    model.eval()
    Xq = torch.tensor(X_query, dtype=torch.float32).to(device)
    Hs = []
    for i in range(len(Xq)):
        xi = Xq[i:i + 1].requires_grad_(True)
        si = model(xi).squeeze(0)
        rows = [
            torch.autograd.grad(si[k], xi, retain_graph=(k < 3), create_graph=False)[0]
            .squeeze(0).detach().cpu()
            for k in range(4)
        ]
        Hs.append(torch.stack(rows).numpy())
    return np.array(Hs)


def hessian_diagnostics(
    model: ScoreNetSiLU,
    X_query_pool: np.ndarray,
    sigma_n: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    """Hessian constancy over N_QUERY noisy query points drawn from X_query_pool."""
    n_pool = len(X_query_pool)
    n_use  = min(N_QUERY, n_pool)
    idx    = rng.integers(0, n_pool, size=n_use)
    x_clean = X_query_pool[idx]
    x_noisy = x_clean + sigma_n * rng.standard_normal(x_clean.shape)

    t0 = time.time()
    H  = compute_hessians(model, x_noisy, device)
    print(f"    Jacobian ({n_use} pts) in {time.time()-t0:.1f}s")

    diag          = np.diagonal(H, axis1=1, axis2=2)
    mean_abs_diag = float(np.mean(np.abs(diag)))
    denom         = mean_abs_diag + 1e-12

    const_em   = float(np.std(H[:, 0, 3]) / denom)
    const_diag = float(np.mean(np.std(diag, axis=0)) / denom)

    abs_em    = np.abs(H[:, 0, 3])
    H_em_mean = float(abs_em.mean())
    boots     = np.array([
        rng.choice(abs_em, size=n_use, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    ci = np.percentile(boots, [2.5, 97.5])

    return dict(
        H_samples  = H,
        const_em   = const_em,
        const_diag = const_diag,
        H_em_mean  = H_em_mean,
        H_em_ci    = ci,
    )


# ---------------------------------------------------------------------------
# Graphical lasso sweep
# ---------------------------------------------------------------------------
def glasso_sweep(X: np.ndarray, lambda_grid: np.ndarray) -> dict:
    """
    Blanket window: H[0,3]=0 AND all four ring edges nonzero.
    Ring edges: (0,1) η–s, (0,2) η–a, (1,3) s–μ, (2,3) a–μ.
    H[1,2] (s–a) excluded — also theoretically zero (s⊥a|{η,μ}).
    """
    blanket_entry = (0, 3)
    other_offdiag = [(0, 1), (0, 2), (1, 3), (2, 3)]

    H_list    = []
    in_window = np.zeros(len(lambda_grid), dtype=bool)

    for k, lam in enumerate(lambda_grid):
        try:
            gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4, assume_centered=False)
            gl.fit(X)
            H = gl.precision_.copy()
        except Exception:
            H = None
        H_list.append(H)
        if H is not None:
            blanket_zero   = abs(H[0, 3]) < ZERO_THRESH
            others_nonzero = all(abs(H[r, c]) > ZERO_THRESH for r, c in other_offdiag)
            in_window[k]   = blanket_zero and others_nonzero

    if np.any(in_window):
        valid  = lambda_grid[in_window]
        lam_lo = float(valid.min())
        lam_hi = float(valid.max())
        width  = float(np.log10(lam_hi / lam_lo)) if lam_hi > lam_lo else 0.0
    else:
        lam_lo, lam_hi, width = None, None, 0.0

    H03_vs_lam = np.array([
        H[0, 3] if H is not None else np.nan for H in H_list
    ])
    return dict(
        in_window   = in_window,
        lam_lo      = lam_lo,
        lam_hi      = lam_hi,
        width_dec   = width,
        H03_vs_lam  = H03_vs_lam,
        n_in_window = int(np.sum(in_window)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lag1_acf(X: np.ndarray) -> np.ndarray:
    acfs = []
    for j in range(X.shape[1]):
        x   = X[:, j] - X[:, j].mean()
        var = (x ** 2).mean()
        acfs.append(float((x[:-1] * x[1:]).mean() / var) if var > 1e-12 else 0.0)
    return np.array(acfs)


def compute_H_emp(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Sigma = np.cov(X.T)
    return np.linalg.inv(Sigma), Sigma


def print_H_emp_row(label: str, n: int, H: np.ndarray, Sigma: np.ndarray) -> dict:
    h03     = float(H[0, 3])
    h03_rel = abs(h03) / float(np.max(np.abs(H)))
    cond    = float(np.linalg.cond(Sigma))
    eigs    = np.linalg.eigvalsh(Sigma)
    spd_ok  = bool(eigs.min() > 0)
    print(f"  {label} (n={n}): H[0,3]={h03:+.6f}  |/max|={h03_rel:.2e}  "
          f"cond={cond:.2f}  min_eig={eigs.min():.4e}  {'PASS' if spd_ok else '*** NOT SPD ***'}")
    return dict(h03=h03, h03_rel=h03_rel, cond=cond, min_eig=float(eigs.min()), spd_ok=spd_ok)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng    = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 2A Batch 2 — α ∈ {ALPHA_LIST}, κ={KAPPA}, σ={SIGMA}, σ_n={SIGMA_N}")
    print(f"GLasso λ grid: logspace(-4, 1, {N_LAMBDA})")

    out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    batch_rows: list[dict] = []

    for i_a, alpha in enumerate(ALPHA_LIST):
        is_bistable = alpha > 0
        sign_str    = "p" if alpha > 0 else ("z" if alpha == 0.0 else "m")
        tag         = f"alpha{sign_str}{abs(alpha):.2f}"

        print(f"\n{'='*65}")
        print(f"α = {alpha:+.2f}   κ = {KAPPA}   σ = {SIGMA}")
        regime = "bistable (per-basin)" if is_bistable else "bifurcation point"
        print(f"  Regime: {regime}")
        print(f"{'='*65}")

        # ── PILOT ────────────────────────────────────────────────────────────
        print("\n[PILOT]")

        # Defaults for values that differ by regime
        subsample_use = SUBSAMPLE_DEFAULT
        n_cross_p     = -1           # -1 = "not counted"
        cross_ok      = True
        acf_ok        = False
        max_acf_p     = np.nan
        acf_pos_p = acf_neg_p = np.full(4, np.nan)
        max_acf_pos_p = max_acf_neg_p = np.nan

        if alpha == 0.0:
            # Bifurcation: search for subsample that achieves ACF[μ] < 0.05
            print(f"  Searching for adequate subsample (ACF[μ] < 0.05)...")
            for sub in SUBSAMPLE_SEARCH:
                t0 = time.time()
                X_pilot, _ = euler_maruyama(
                    alpha=alpha, kappa=KAPPA, sigma=SIGMA,
                    n_steps=N_PILOT * sub, subsample=sub, seed=SEED,
                )
                dt    = time.time() - t0
                acf_p = lag1_acf(X_pilot)
                max_p = float(np.max(np.abs(acf_p)))
                acf_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_p))
                print(f"  sub={sub}: {N_PILOT} samples in {dt:.1f}s")
                print(f"  ACF lag-1: {acf_str}")
                print(f"  max ACF = {max_p:.4f}  "
                      + ("PASS" if max_p < 0.05 else "FAIL — trying larger"))
                if max_p < 0.05:
                    subsample_use = sub
                    max_acf_p     = max_p
                    acf_ok        = True
                    break
            else:
                subsample_use = SUBSAMPLE_SEARCH[-1]
                # recompute final pilot stats with largest subsample
                X_pilot, _ = euler_maruyama(
                    alpha=alpha, kappa=KAPPA, sigma=SIGMA,
                    n_steps=N_PILOT * subsample_use, subsample=subsample_use, seed=SEED,
                )
                acf_p     = lag1_acf(X_pilot)
                max_acf_p = float(np.max(np.abs(acf_p)))
                print(f"  *** ACF did not reach <0.05. Proceeding with sub={subsample_use}.")
            print(f"  Subsample selected: {subsample_use}")

        else:
            # Bistable: crossing count + per-basin ACF
            t0 = time.time()
            X_pilot, n_cross_p = euler_maruyama(
                alpha=alpha, kappa=KAPPA, sigma=SIGMA,
                n_steps=N_PILOT * SUBSAMPLE_DEFAULT, subsample=SUBSAMPLE_DEFAULT, seed=SEED,
            )
            print(f"  {N_PILOT} samples in {time.time()-t0:.1f}s")

            cross_ok = n_cross_p >= MIN_CROSSINGS
            print(f"  Basin crossings: {n_cross_p}"
                  + (f"  PASS (≥{MIN_CROSSINGS})" if cross_ok
                     else f"  *** FAIL (<{MIN_CROSSINGS}) ***"))

            mu_p     = X_pilot[:, 3]
            n_pos_p  = int((mu_p > 0).sum())
            n_neg_p  = int((mu_p < 0).sum())
            print(f"  Basin split (pilot): μ>0 n={n_pos_p}  μ<0 n={n_neg_p}")

            if n_pos_p >= 10 and n_neg_p >= 10:
                acf_pos_p     = lag1_acf(X_pilot[mu_p > 0])
                acf_neg_p     = lag1_acf(X_pilot[mu_p < 0])
                max_acf_pos_p = float(np.max(np.abs(acf_pos_p)))
                max_acf_neg_p = float(np.max(np.abs(acf_neg_p)))
                acf_ok        = max_acf_pos_p < 0.05 and max_acf_neg_p < 0.05
                max_acf_p     = max(max_acf_pos_p, max_acf_neg_p)
                for basin_label, acf_arr, mx in [
                    ("μ>0", acf_pos_p, max_acf_pos_p),
                    ("μ<0", acf_neg_p, max_acf_neg_p),
                ]:
                    acf_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_arr))
                    print(f"  Per-basin ACF ({basin_label}): {acf_str}")
                    print(f"  max ACF ({basin_label}) = {mx:.4f}  "
                          + ("PASS" if mx < 0.05 else "FAIL"))
            else:
                print(f"  *** Too few samples in one basin for per-basin ACF ***")

        # ── FULL RUN ─────────────────────────────────────────────────────────
        print("\n[FULL RUN]")
        t0 = time.time()
        X, n_cross_full = euler_maruyama(
            alpha=alpha, kappa=KAPPA, sigma=SIGMA,
            n_steps=N_FULL * subsample_use, subsample=subsample_use, seed=SEED + 1,
        )
        t_sim = time.time() - t0
        print(f"  N={N_FULL} samples in {t_sim:.1f}s  (subsample={subsample_use})")

        if is_bistable:
            cross_full_ok = n_cross_full >= MIN_CROSSINGS
            print(f"  Basin crossings (full): {n_cross_full}"
                  + (f"  PASS (≥{MIN_CROSSINGS})" if cross_full_ok
                     else "  *** FAIL ***"))

        # Global ACF (all α)
        acf_full    = lag1_acf(X)
        max_acf_f   = float(np.max(np.abs(acf_full)))
        acf_str_f   = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_full))
        print(f"  ACF lag-1 (full, global): {acf_str_f}  max={max_acf_f:.4f}")

        # Basin split for full run (α>0)
        if is_bistable:
            mu_f       = X[:, 3]
            mask_pos_f = mu_f > 0
            mask_neg_f = mu_f < 0
            n_pos_f    = int(mask_pos_f.sum())
            n_neg_f    = int(mask_neg_f.sum())
            X_pos_f    = X[mask_pos_f]
            X_neg_f    = X[mask_neg_f]
            print(f"  Basin split (full):   μ>0 n={n_pos_f}  μ<0 n={n_neg_f}")

            acf_pos_f     = lag1_acf(X_pos_f) if n_pos_f >= 10 else np.full(4, np.nan)
            acf_neg_f     = lag1_acf(X_neg_f) if n_neg_f >= 10 else np.full(4, np.nan)
            max_acf_pos_f = float(np.nanmax(np.abs(acf_pos_f)))
            max_acf_neg_f = float(np.nanmax(np.abs(acf_neg_f)))
            for basin_label, mx in [("μ>0", max_acf_pos_f), ("μ<0", max_acf_neg_f)]:
                print(f"  Per-basin ACF (full, {basin_label}): max={mx:.4f}  "
                      + ("PASS" if mx < 0.05 else "FAIL"))
        else:
            n_pos_f = n_neg_f = 0
            X_pos_f = X_neg_f = np.empty((0, 4))
            acf_pos_f = acf_neg_f = np.full(4, np.nan)
            max_acf_pos_f = max_acf_neg_f = np.nan

        # Train/val split (uses rng → advances per α)
        n_val   = int(N_FULL * VAL_FRAC)
        n_train = N_FULL - n_val
        perm    = rng.permutation(N_FULL)
        X_tr, X_vl = X[perm[:n_train]], X[perm[n_train:]]

        # ── H_EMP ────────────────────────────────────────────────────────────
        print("\n[H_EMP]")
        H_emp_global, Sigma_emp_global = compute_H_emp(X)
        sym_err = float(np.max(np.abs(H_emp_global - H_emp_global.T)))
        global_label = "global" if is_bistable else ""
        g = print_H_emp_row(global_label or "H_emp", N_FULL, H_emp_global, Sigma_emp_global)
        print(f"  H_emp symmetry err = {sym_err:.2e}"
              + ("  PASS" if sym_err < 1e-12 else "  WARN"))

        # Per-basin H_emp (α>0)
        pb_pos = pb_neg = {}
        H_pos  = H_neg  = np.full((4, 4), np.nan)
        if is_bistable:
            print(f"\n  [PER-BASIN H_EMP]")
            if n_pos_f >= MIN_PER_BASIN:
                H_pos, Sig_pos = compute_H_emp(X_pos_f)
                pb_pos = print_H_emp_row("μ>0", n_pos_f, H_pos, Sig_pos)
            else:
                print(f"  μ>0 (n={n_pos_f}): *** too few samples (need {MIN_PER_BASIN}) ***")
                pb_pos = dict(h03=np.nan, h03_rel=np.nan, cond=np.nan, min_eig=np.nan, spd_ok=False)
            if n_neg_f >= MIN_PER_BASIN:
                H_neg, Sig_neg = compute_H_emp(X_neg_f)
                pb_neg = print_H_emp_row("μ<0", n_neg_f, H_neg, Sig_neg)
            else:
                print(f"  μ<0 (n={n_neg_f}): *** too few samples (need {MIN_PER_BASIN}) ***")
                pb_neg = dict(h03=np.nan, h03_rel=np.nan, cond=np.nan, min_eig=np.nan, spd_ok=False)

        # ── SCORE NETWORK ────────────────────────────────────────────────────
        print(f"\n[SCORE NETWORK]  σ_n = {SIGMA_N}")
        t0 = time.time()
        model, tr_hist, vl_hist, best_val, n_ep = train_network(
            X_tr, X_vl, SIGMA_N, device, seed=SEED
        )
        print(f"  Training: {time.time()-t0:.1f}s  ({n_ep} epochs)  best_val={best_val:.5f}")

        # Query pool for constancy check
        if is_bistable:
            X_deep  = X[np.abs(X[:, 3]) > MU_BASIN_THRESH]
            n_deep  = len(X_deep)
            pool    = X_deep if n_deep >= 10 else X
            pool_lbl = f"deep-basin |μ|>{MU_BASIN_THRESH}  n={n_deep}"
        else:
            pool     = X
            pool_lbl = "all samples (unimodal)"
        print(f"  Query pool: {pool_lbl}")

        hnet_rng = np.random.default_rng(SEED + 300 + i_a)
        hdiag    = hessian_diagnostics(model, pool, SIGMA_N, hnet_rng, device)
        ci       = hdiag["H_em_ci"]
        const_ok = hdiag["const_em"] < 0.1 and hdiag["const_diag"] < 0.1
        print(f"  const[0,3]  = {hdiag['const_em']:.4f}")
        print(f"  const[diag] = {hdiag['const_diag']:.4f}")
        print(f"  |Ĥ_ημ| mean = {hdiag['H_em_mean']:.5f}   "
              f"95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]")
        print(f"  Constancy: {'PASS' if const_ok else '*** FAIL ***'}")
        if alpha == 0.0:
            print(f"  Note: large |Ĥ_ημ| expected at bifurcation (flat μ marginal).")

        # ── GLASSO ───────────────────────────────────────────────────────────
        print(f"\n[GLASSO]  {N_LAMBDA}-pt λ-sweep  logspace(-4,1)")
        if is_bistable:
            print(f"  Note: density is bimodal; GLasso Gaussian assumption violated.")
        t0  = time.time()
        gl  = glasso_sweep(X, LAMBDA_GRID)
        print(f"  Sweep in {time.time()-t0:.1f}s")
        if gl["lam_lo"] is not None:
            print(f"  Blanket window: [{gl['lam_lo']:.5f}, {gl['lam_hi']:.5f}]  "
                  f"width = {gl['width_dec']:.3f} dec")
        else:
            print(f"  *** No blanket window found ***")
        print(f"  λ values in window: {gl['n_in_window']}/{N_LAMBDA}")

        # ── KURTOSIS ─────────────────────────────────────────────────────────
        print("\n[KURTOSIS]")
        kurt = float(scipy_kurtosis(X[:, 3], fisher=True))
        print(f"  Excess kurtosis of μ = {kurt:.4f}  (Gaussian = 0)")

        # ── SAVE PER α ───────────────────────────────────────────────────────
        out_path  = os.path.join(out_dir, f"phase2A_{tag}.npz")
        np.savez(
            out_path,
            # parameters
            alpha=alpha, kappa=KAPPA, sigma=SIGMA, sigma_n=SIGMA_N,
            n_pilot=N_PILOT, n_full=N_FULL, subsample=subsample_use,
            is_bistable=is_bistable,
            # pilot diagnostics
            acf_ok=acf_ok, max_acf_pilot=max_acf_p,
            n_cross_pilot=n_cross_p,
            acf_pos_pilot=acf_pos_p, acf_neg_pilot=acf_neg_p,
            max_acf_pos_pilot=max_acf_pos_p, max_acf_neg_pilot=max_acf_neg_p,
            # full-run diagnostics
            acf_full=acf_full, max_acf_full=max_acf_f,
            n_cross_full=n_cross_full, n_pos_full=n_pos_f, n_neg_full=n_neg_f,
            acf_pos_full=acf_pos_f, acf_neg_full=acf_neg_f,
            max_acf_pos_full=max_acf_pos_f, max_acf_neg_full=max_acf_neg_f,
            # global H_emp
            H_emp_global=H_emp_global, Sigma_emp_global=Sigma_emp_global,
            H_emp_03_global=g["h03"], H_emp_03_rel_global=g["h03_rel"],
            Sigma_cond_global=g["cond"], Sigma_min_eig_global=g["min_eig"],
            H_emp_sym_err=sym_err,
            # per-basin H_emp
            H_emp_pos=H_pos, H_emp_neg=H_neg,
            H_emp_03_pos=pb_pos.get("h03", np.nan),
            H_emp_03_neg=pb_neg.get("h03", np.nan),
            H_emp_03_rel_pos=pb_pos.get("h03_rel", np.nan),
            H_emp_03_rel_neg=pb_neg.get("h03_rel", np.nan),
            # score network
            score_train_losses=tr_hist, score_val_losses=vl_hist,
            score_best_val=best_val, score_epochs=n_ep,
            score_const_em=hdiag["const_em"], score_const_diag=hdiag["const_diag"],
            score_H_em_mean=hdiag["H_em_mean"], score_H_em_ci=hdiag["H_em_ci"],
            score_H_samples=hdiag["H_samples"],
            # GLasso
            glasso_lambda_grid=LAMBDA_GRID,
            glasso_in_window=gl["in_window"],
            glasso_H03_vs_lam=gl["H03_vs_lam"],
            glasso_lam_lo=gl["lam_lo"] if gl["lam_lo"] is not None else np.nan,
            glasso_lam_hi=gl["lam_hi"] if gl["lam_hi"] is not None else np.nan,
            glasso_width_dec=gl["width_dec"],
            # kurtosis
            kurtosis_mu=kurt,
        )
        print(f"\n  Saved → {out_path}")

        batch_rows.append(dict(
            alpha          = alpha,
            is_bistable    = is_bistable,
            acf_ok         = acf_ok,
            subsample      = subsample_use,
            n_cross_full   = n_cross_full,
            H_emp_03_global= g["h03"],
            H_emp_rel_global= g["h03_rel"],
            H_emp_03_pos   = pb_pos.get("h03", np.nan),
            H_emp_03_neg   = pb_neg.get("h03", np.nan),
            H_em_mean      = hdiag["H_em_mean"],
            const_em       = hdiag["const_em"],
            const_diag     = hdiag["const_diag"],
            gl_width       = gl["width_dec"],
            kurtosis_mu    = kurt,
        ))

    # ── BATCH SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("BATCH 2 SUMMARY")
    print(f"{'='*65}")
    hdr = (f"{'α':>6}  {'ACF':>4}  {'sub':>5}  {'cross':>6}  "
           f"{'H[0,3]_gl':>10}  {'H[0,3]_pos':>11}  {'H[0,3]_neg':>11}  "
           f"{'GL_wid':>7}  {'kurt_μ':>7}")
    print(hdr)
    for r in batch_rows:
        acf_flag = "OK" if r["acf_ok"] else "FAIL"
        cross_s  = str(r["n_cross_full"]) if r["is_bistable"] else "  n/a"
        h03g     = f"{r['H_emp_03_global']:+.5f}"
        h03p     = f"{r['H_emp_03_pos']:+.5f}" if not np.isnan(r["H_emp_03_pos"]) else "    —   "
        h03n     = f"{r['H_emp_03_neg']:+.5f}" if not np.isnan(r["H_emp_03_neg"]) else "    —   "
        print(f"{r['alpha']:>6.2f}  {acf_flag:>4}  {r['subsample']:>5}  {cross_s:>6}  "
              f"{h03g:>10}  {h03p:>11}  {h03n:>11}  "
              f"{r['gl_width']:>7.3f}  {r['kurtosis_mu']:>7.4f}")

    summary_path = os.path.join(out_dir, "phase2A_batch2_summary.npz")
    np.savez(
        summary_path,
        alpha_list       = np.array(ALPHA_LIST),
        H_emp_03_global  = np.array([r["H_emp_03_global"]  for r in batch_rows]),
        H_emp_rel_global = np.array([r["H_emp_rel_global"] for r in batch_rows]),
        H_emp_03_pos     = np.array([r["H_emp_03_pos"]     for r in batch_rows]),
        H_emp_03_neg     = np.array([r["H_emp_03_neg"]     for r in batch_rows]),
        H_em_mean        = np.array([r["H_em_mean"]        for r in batch_rows]),
        const_em         = np.array([r["const_em"]         for r in batch_rows]),
        const_diag       = np.array([r["const_diag"]       for r in batch_rows]),
        gl_width         = np.array([r["gl_width"]         for r in batch_rows]),
        kurtosis_mu      = np.array([r["kurtosis_mu"]      for r in batch_rows]),
        acf_ok           = np.array([r["acf_ok"]           for r in batch_rows]),
        subsample        = np.array([r["subsample"]        for r in batch_rows]),
        n_cross_full     = np.array([r["n_cross_full"]     for r in batch_rows]),
    )
    print(f"\nBatch 2 summary saved → {summary_path}")


if __name__ == "__main__":
    main()
