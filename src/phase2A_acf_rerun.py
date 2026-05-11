"""
Phase 2A — Per-basin ACF subsample search + selective re-run
α ∈ {0.75, 1.00} — both had per-basin ACF failures in Batch 2

Stage 1 [ACF PILOT]:
  For each α, sweep global subsample ∈ {600, 800, 1000, 1200}.
  At each subsample: simulate N_PILOT=1000 samples, split by sign(μ), compute
  per-basin lag-1 ACF. Select minimum subsample where max ACF < 0.05 for BOTH
  basins. If none passes, use 1200 with a WARNING.

Stage 2 [FULL RE-RUN]:
  Run the full Phase 2A bistable analysis at each α using the determined subsample.
  Overwrites results/phase2/phase2A_alphap{α:.2f}.npz.
  Saves results/phase2/phase2A_acf_rerun_summary.npz.

Context: Batch 2 per-basin ACF failures were attributed to within-basin residual
correlation from the ring-coupling slow mode (τ_ring ≈ 2 time units). At sub=600
(6 time units), the ACF is near exp(−3) ≈ 0.05 — right at threshold. Larger
subsample provides more margin.
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
ALPHA_LIST        = [0.75, 1.00]
SUBSAMPLE_SEARCH  = [600, 800, 1000, 1200]   # per-basin ACF search
KAPPA             = 0.0
SIGMA             = 0.5
SIGMA_N           = 0.05
N_PILOT           = 1_000
N_FULL            = 10_000
N_EPOCHS          = 500
BATCH_SIZE        = 256
LR                = 1e-3
PATIENCE          = 50
HIDDEN            = 64
DEPTH             = 2
N_QUERY           = 500
N_LAMBDA          = 50
LAMBDA_GRID       = np.logspace(-4, 1, N_LAMBDA)
ZERO_THRESH       = 1e-8
N_BOOTSTRAP       = 1_000
VAL_FRAC          = 0.2
SEED              = 42
MIN_CROSSINGS     = 20
MU_BASIN_THRESH   = 0.3
MIN_PER_BASIN     = 100

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_NAMES  = ["η", "s", "a", "μ"]


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------
class ScoreNetSiLU(nn.Module):
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
    sn  = model.sigma_n
    eps = torch.randn_like(x_batch)
    return ((model(x_batch + sn * eps) + eps / sn) ** 2).sum(1).mean()


def train_network(X_tr, X_vl, sigma_n, device, seed=SEED):
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
# Hessian
# ---------------------------------------------------------------------------
def compute_hessians(model, X_query, device):
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


def hessian_diagnostics(model, X_pool, sigma_n, rng, device):
    n_use   = min(N_QUERY, len(X_pool))
    idx     = rng.integers(0, len(X_pool), size=n_use)
    x_noisy = X_pool[idx] + sigma_n * rng.standard_normal((n_use, 4))

    t0 = time.time()
    H  = compute_hessians(model, x_noisy, device)
    print(f"    Jacobian ({n_use} pts) in {time.time()-t0:.1f}s")

    diag          = np.diagonal(H, axis1=1, axis2=2)
    mean_abs_diag = float(np.mean(np.abs(diag)))
    denom         = mean_abs_diag + 1e-12
    const_em      = float(np.std(H[:, 0, 3]) / denom)
    const_diag    = float(np.mean(np.std(diag, axis=0)) / denom)

    abs_em    = np.abs(H[:, 0, 3])
    H_em_mean = float(abs_em.mean())
    boots     = np.array([
        rng.choice(abs_em, size=n_use, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    ci = np.percentile(boots, [2.5, 97.5])
    return dict(H_samples=H, const_em=const_em, const_diag=const_diag,
                H_em_mean=H_em_mean, H_em_ci=ci)


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


def print_H_emp_row(label, n, H, Sigma) -> dict:
    h03     = float(H[0, 3])
    h03_rel = abs(h03) / float(np.max(np.abs(H)))
    cond    = float(np.linalg.cond(Sigma))
    eigs    = np.linalg.eigvalsh(Sigma)
    spd_ok  = bool(eigs.min() > 0)
    print(f"  {label} (n={n}): H[0,3]={h03:+.6f}  |/max|={h03_rel:.2e}  "
          f"cond={cond:.2f}  min_eig={eigs.min():.4e}  "
          + ("PASS" if spd_ok else "*** NOT SPD ***"))
    return dict(h03=h03, h03_rel=h03_rel, cond=cond, min_eig=float(eigs.min()), spd_ok=spd_ok)


def glasso_sweep(X: np.ndarray, lambda_grid: np.ndarray) -> dict:
    """
    Blanket window: H[0,3]=0 AND ring edges (0,1),(0,2),(1,3),(2,3) nonzero.
    H[1,2] excluded — also theoretically zero.
    """
    blanket_entry = (0, 3)
    other_offdiag = [(0, 1), (0, 2), (1, 3), (2, 3)]
    H_list        = []
    in_window     = np.zeros(len(lambda_grid), dtype=bool)

    for lam in lambda_grid:
        try:
            gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4, assume_centered=False)
            gl.fit(X)
            H = gl.precision_.copy()
        except Exception:
            H = None
        H_list.append(H)
        if H is not None:
            in_window[len(H_list) - 1] = (
                abs(H[0, 3]) < ZERO_THRESH
                and all(abs(H[r, c]) > ZERO_THRESH for r, c in other_offdiag)
            )

    if np.any(in_window):
        valid  = lambda_grid[in_window]
        lam_lo = float(valid.min())
        lam_hi = float(valid.max())
        width  = float(np.log10(lam_hi / lam_lo)) if lam_hi > lam_lo else 0.0
    else:
        lam_lo = lam_hi = None
        width  = 0.0

    return dict(
        in_window   = in_window,
        lam_lo      = lam_lo,
        lam_hi      = lam_hi,
        width_dec   = width,
        H03_vs_lam  = np.array([H[0, 3] if H is not None else np.nan for H in H_list]),
        n_in_window = int(np.sum(in_window)),
    )


# ---------------------------------------------------------------------------
# Stage 1 — ACF pilot sweep
# ---------------------------------------------------------------------------
def run_acf_pilot(device) -> dict[float, int]:
    """Return {alpha: selected_subsample} for each α in ALPHA_LIST."""
    selected: dict[float, int] = {}

    print("\n" + "=" * 65)
    print("STAGE 1 — PER-BASIN ACF SUBSAMPLE SEARCH")
    print("=" * 65)

    for alpha in ALPHA_LIST:
        sign_str = f"+{alpha:.2f}"
        print(f"\nα = {sign_str}")
        print("-" * 40)

        found = False
        for sub in SUBSAMPLE_SEARCH:
            t0   = time.time()
            X, _ = euler_maruyama(
                alpha=alpha, kappa=KAPPA, sigma=SIGMA,
                n_steps=N_PILOT * sub, subsample=sub, seed=SEED,
            )
            dt  = time.time() - t0
            mu  = X[:, 3]
            n_p = int((mu > 0).sum())
            n_n = int((mu < 0).sum())

            if n_p < 10 or n_n < 10:
                print(f"  sub={sub}: {N_PILOT} samples in {dt:.1f}s  "
                      f"μ>0 n={n_p}  μ<0 n={n_n}  *** too few in one basin ***")
                continue

            acf_p = lag1_acf(X[mu > 0])
            acf_n = lag1_acf(X[mu < 0])
            max_p = float(np.max(np.abs(acf_p)))
            max_n = float(np.max(np.abs(acf_n)))
            pass_ = max_p < 0.05 and max_n < 0.05

            acf_str_p = "  ".join(f"{nm}={v:+.4f}" for nm, v in zip(STATE_NAMES, acf_p))
            acf_str_n = "  ".join(f"{nm}={v:+.4f}" for nm, v in zip(STATE_NAMES, acf_n))
            status_p  = "PASS" if max_p < 0.05 else "FAIL"
            status_n  = "PASS" if max_n < 0.05 else "FAIL"

            print(f"  sub={sub}: {N_PILOT} samples in {dt:.1f}s  "
                  f"μ>0 n={n_p}  μ<0 n={n_n}")
            print(f"    μ>0 ACF: {acf_str_p}   max={max_p:.4f}  {status_p}")
            print(f"    μ<0 ACF: {acf_str_n}   max={max_n:.4f}  {status_n}")

            if pass_:
                print(f"  → Selected subsample = {sub} for α={sign_str}")
                selected[alpha] = sub
                found = True
                break

        if not found:
            print(f"  *** No subsample in search list passed. Using {SUBSAMPLE_SEARCH[-1]}. ***")
            selected[alpha] = SUBSAMPLE_SEARCH[-1]

    print("\nACF pilot summary:")
    for alpha, sub in selected.items():
        print(f"  α={alpha:+.2f}  →  subsample = {sub}")

    return selected


# ---------------------------------------------------------------------------
# Stage 2 — Full re-run
# ---------------------------------------------------------------------------
def run_full(alpha: float, subsample: int, rng: np.random.Generator,
             i_a: int, device: torch.device, out_dir: str) -> dict:
    """Full Phase 2A bistable analysis for one α. Returns summary dict."""
    tag = f"alphap{alpha:.2f}"
    print(f"\n{'='*65}")
    print(f"α = {alpha:+.2f}   κ = {KAPPA}   σ = {SIGMA}   subsample = {subsample}")
    print(f"{'='*65}")

    # ── FULL RUN ─────────────────────────────────────────────────────────────
    print("\n[FULL RUN]")
    t0 = time.time()
    X, n_cross = euler_maruyama(
        alpha=alpha, kappa=KAPPA, sigma=SIGMA,
        n_steps=N_FULL * subsample, subsample=subsample, seed=SEED + 1,
    )
    print(f"  N={N_FULL} samples in {time.time()-t0:.1f}s  (subsample={subsample})")

    cross_ok = n_cross >= MIN_CROSSINGS
    print(f"  Basin crossings: {n_cross}"
          + (f"  PASS (≥{MIN_CROSSINGS})" if cross_ok else "  *** FAIL ***"))

    # Global ACF
    acf_full  = lag1_acf(X)
    max_acf_f = float(np.max(np.abs(acf_full)))
    print(f"  ACF lag-1 (global): "
          + "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_full))
          + f"  max={max_acf_f:.4f}")

    # Basin split
    mu_f       = X[:, 3]
    mask_pos   = mu_f > 0
    mask_neg   = mu_f < 0
    n_pos      = int(mask_pos.sum())
    n_neg      = int(mask_neg.sum())
    X_pos      = X[mask_pos]
    X_neg      = X[mask_neg]
    print(f"  Basin split: μ>0 n={n_pos}  μ<0 n={n_neg}")

    # Per-basin ACF
    acf_pos_f    = lag1_acf(X_pos) if n_pos >= 10 else np.full(4, np.nan)
    acf_neg_f    = lag1_acf(X_neg) if n_neg >= 10 else np.full(4, np.nan)
    max_acf_pos  = float(np.nanmax(np.abs(acf_pos_f)))
    max_acf_neg  = float(np.nanmax(np.abs(acf_neg_f)))
    acf_basin_ok = max_acf_pos < 0.05 and max_acf_neg < 0.05

    for lbl, acf_arr, mx in [("μ>0", acf_pos_f, max_acf_pos),
                               ("μ<0", acf_neg_f, max_acf_neg)]:
        print(f"  Per-basin ACF ({lbl}): "
              + "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_arr))
              + f"  max={mx:.4f}  {'PASS' if mx < 0.05 else 'FAIL'}")

    # Train/val split
    perm    = rng.permutation(N_FULL)
    X_tr    = X[perm[:N_FULL - int(N_FULL * VAL_FRAC)]]
    X_vl    = X[perm[N_FULL - int(N_FULL * VAL_FRAC):]]

    # ── H_EMP ────────────────────────────────────────────────────────────────
    print("\n[H_EMP]")
    H_global, Sig_global = compute_H_emp(X)
    sym_err = float(np.max(np.abs(H_global - H_global.T)))
    g = print_H_emp_row("global", N_FULL, H_global, Sig_global)
    print(f"  H_emp symmetry err = {sym_err:.2e}"
          + ("  PASS" if sym_err < 1e-12 else "  WARN"))

    print(f"\n  [PER-BASIN H_EMP]")
    H_pos_mat = np.full((4, 4), np.nan)
    H_neg_mat = np.full((4, 4), np.nan)
    pb_pos = pb_neg = dict(h03=np.nan, h03_rel=np.nan, cond=np.nan,
                           min_eig=np.nan, spd_ok=False)
    if n_pos >= MIN_PER_BASIN:
        H_pos_mat, Sig_pos = compute_H_emp(X_pos)
        pb_pos = print_H_emp_row("μ>0", n_pos, H_pos_mat, Sig_pos)
    else:
        print(f"  μ>0: too few samples (n={n_pos})")
    if n_neg >= MIN_PER_BASIN:
        H_neg_mat, Sig_neg = compute_H_emp(X_neg)
        pb_neg = print_H_emp_row("μ<0", n_neg, H_neg_mat, Sig_neg)
    else:
        print(f"  μ<0: too few samples (n={n_neg})")

    # ── SCORE NETWORK ────────────────────────────────────────────────────────
    print(f"\n[SCORE NETWORK]  σ_n = {SIGMA_N}")
    t0 = time.time()
    model, tr_hist, vl_hist, best_val, n_ep = train_network(
        X_tr, X_vl, SIGMA_N, device, seed=SEED,
    )
    print(f"  Training: {time.time()-t0:.1f}s  ({n_ep} epochs)  best_val={best_val:.5f}")

    X_deep  = X[np.abs(mu_f) > MU_BASIN_THRESH]
    n_deep  = len(X_deep)
    pool    = X_deep if n_deep >= 10 else X
    print(f"  Query pool: deep-basin |μ|>{MU_BASIN_THRESH}  n={n_deep}")

    hnet_rng = np.random.default_rng(SEED + 300 + i_a)
    hdiag    = hessian_diagnostics(model, pool, SIGMA_N, hnet_rng, device)
    ci       = hdiag["H_em_ci"]
    const_ok = hdiag["const_em"] < 0.1 and hdiag["const_diag"] < 0.1
    print(f"  const[0,3]  = {hdiag['const_em']:.4f}")
    print(f"  const[diag] = {hdiag['const_diag']:.4f}")
    print(f"  |Ĥ_ημ| mean = {hdiag['H_em_mean']:.5f}   "
          f"95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]")
    print(f"  Constancy: {'PASS' if const_ok else '*** FAIL ***'}")

    # ── GLASSO ───────────────────────────────────────────────────────────────
    print(f"\n[GLASSO]  {N_LAMBDA}-pt λ-sweep  logspace(-4,1)")
    print(f"  Note: density is bimodal; GLasso Gaussian assumption violated.")
    t0 = time.time()
    gl = glasso_sweep(X, LAMBDA_GRID)
    print(f"  Sweep in {time.time()-t0:.1f}s")
    if gl["lam_lo"] is not None:
        print(f"  Blanket window: [{gl['lam_lo']:.5f}, {gl['lam_hi']:.5f}]  "
              f"width = {gl['width_dec']:.3f} dec")
    else:
        print(f"  *** No blanket window found ***")
    print(f"  λ values in window: {gl['n_in_window']}/{N_LAMBDA}")

    # ── KURTOSIS ─────────────────────────────────────────────────────────────
    print("\n[KURTOSIS]")
    kurt = float(scipy_kurtosis(X[:, 3], fisher=True))
    print(f"  Excess kurtosis of μ = {kurt:.4f}  (Gaussian = 0)")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"phase2A_{tag}.npz")
    np.savez(
        out_path,
        alpha=alpha, kappa=KAPPA, sigma=SIGMA, sigma_n=SIGMA_N,
        n_full=N_FULL, subsample=subsample, is_bistable=True,
        acf_full=acf_full, max_acf_full=max_acf_f,
        acf_pos_full=acf_pos_f, acf_neg_full=acf_neg_f,
        max_acf_pos_full=max_acf_pos, max_acf_neg_full=max_acf_neg,
        acf_basin_ok=acf_basin_ok,
        n_cross_full=n_cross, n_pos_full=n_pos, n_neg_full=n_neg,
        H_emp_global=H_global, Sigma_emp_global=Sig_global,
        H_emp_03_global=g["h03"], H_emp_03_rel_global=g["h03_rel"],
        Sigma_cond_global=g["cond"], Sigma_min_eig_global=g["min_eig"],
        H_emp_sym_err=sym_err,
        H_emp_pos=H_pos_mat, H_emp_neg=H_neg_mat,
        H_emp_03_pos=pb_pos["h03"], H_emp_03_neg=pb_neg["h03"],
        H_emp_03_rel_pos=pb_pos["h03_rel"], H_emp_03_rel_neg=pb_neg["h03_rel"],
        score_train_losses=tr_hist, score_val_losses=vl_hist,
        score_best_val=best_val, score_epochs=n_ep,
        score_const_em=hdiag["const_em"], score_const_diag=hdiag["const_diag"],
        score_H_em_mean=hdiag["H_em_mean"], score_H_em_ci=hdiag["H_em_ci"],
        score_H_samples=hdiag["H_samples"],
        glasso_lambda_grid=LAMBDA_GRID,
        glasso_in_window=gl["in_window"],
        glasso_H03_vs_lam=gl["H03_vs_lam"],
        glasso_lam_lo=gl["lam_lo"] if gl["lam_lo"] is not None else np.nan,
        glasso_lam_hi=gl["lam_hi"] if gl["lam_hi"] is not None else np.nan,
        glasso_width_dec=gl["width_dec"],
        kurtosis_mu=kurt,
    )
    print(f"\n  Saved → {out_path}")

    return dict(
        alpha=alpha, subsample=subsample, acf_basin_ok=acf_basin_ok,
        n_cross=n_cross,
        H_emp_03_global=g["h03"], H_emp_rel_global=g["h03_rel"],
        H_emp_03_pos=pb_pos["h03"], H_emp_03_neg=pb_neg["h03"],
        H_em_mean=hdiag["H_em_mean"],
        const_em=hdiag["const_em"], const_diag=hdiag["const_diag"],
        gl_width=gl["width_dec"], kurtosis_mu=kurt,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 2A ACF pilot + re-run — α ∈ {ALPHA_LIST}")
    print(f"Subsample search: {SUBSAMPLE_SEARCH}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    selected = run_acf_pilot(device)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 2 — FULL RE-RUN WITH DETERMINED SUBSAMPLES")
    print("=" * 65)

    rows = []
    for i_a, alpha in enumerate(ALPHA_LIST):
        subsample = selected[alpha]
        rng       = np.random.default_rng(SEED + 200 + i_a)
        row       = run_full(alpha, subsample, rng, i_a, device, out_dir)
        rows.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RE-RUN SUMMARY")
    print(f"{'='*65}")
    hdr = (f"{'α':>6}  {'sub':>5}  {'ACF':>4}  {'cross':>6}  "
           f"{'H[0,3]_gl':>10}  {'H[0,3]_pos':>11}  {'H[0,3]_neg':>11}  "
           f"{'GL_wid':>7}  {'kurt_μ':>7}")
    print(hdr)
    for r in rows:
        acf_flag = "OK" if r["acf_basin_ok"] else "FAIL"
        h03p = f"{r['H_emp_03_pos']:+.5f}" if not np.isnan(r["H_emp_03_pos"]) else "    —   "
        h03n = f"{r['H_emp_03_neg']:+.5f}" if not np.isnan(r["H_emp_03_neg"]) else "    —   "
        print(f"{r['alpha']:>6.2f}  {r['subsample']:>5}  {acf_flag:>4}  "
              f"{r['n_cross']:>6}  {r['H_emp_03_global']:>+10.5f}  "
              f"{h03p:>11}  {h03n:>11}  "
              f"{r['gl_width']:>7.3f}  {r['kurtosis_mu']:>7.4f}")

    summary_path = os.path.join(out_dir, "phase2A_acf_rerun_summary.npz")
    np.savez(
        summary_path,
        alpha_list       = np.array(ALPHA_LIST),
        subsample        = np.array([r["subsample"]        for r in rows]),
        acf_basin_ok     = np.array([r["acf_basin_ok"]     for r in rows]),
        H_emp_03_global  = np.array([r["H_emp_03_global"]  for r in rows]),
        H_emp_rel_global = np.array([r["H_emp_rel_global"] for r in rows]),
        H_emp_03_pos     = np.array([r["H_emp_03_pos"]     for r in rows]),
        H_emp_03_neg     = np.array([r["H_emp_03_neg"]     for r in rows]),
        H_em_mean        = np.array([r["H_em_mean"]        for r in rows]),
        const_em         = np.array([r["const_em"]         for r in rows]),
        const_diag       = np.array([r["const_diag"]       for r in rows]),
        gl_width         = np.array([r["gl_width"]         for r in rows]),
        kurtosis_mu      = np.array([r["kurtosis_mu"]      for r in rows]),
        n_cross          = np.array([r["n_cross"]          for r in rows]),
    )
    print(f"\nRe-run summary saved → {summary_path}")


if __name__ == "__main__":
    main()
