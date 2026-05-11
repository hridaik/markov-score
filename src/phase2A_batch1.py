"""
Phase 2A Batch 1 — bifurcation sweep α ∈ {-1, -0.75, -0.5, -0.25}

All monostable (unimodal) α values. κ=0, σ=0.5 throughout.
ACF criterion: global < 0.05 at subsample=600 (DEVIATION 012 — global criterion
applies at α ≤ 0; per-basin criterion applies only for α > 0).

For each α:
  (a) Quick pilot (N_PILOT=1000 samples): global ACF check, n_crossings must be 0
  (b) Full run (N=10,000 samples, subsample=600)
  (c) Global H_emp = inv(cov(X))
  (d) Score MLP (SiLU 2×64, σ_n=0.05) + Hessian constancy check (all samples,
      monostable → single basin)
  (e) Graphical lasso 30-pt λ-sweep, blanket window
  (f) Excess kurtosis of μ marginal

Output per α:  results/phase2/phase2A_alpha{sign}{|α|:.2f}.npz
Batch summary: results/phase2/phase2A_batch1_summary.npz
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
ALPHA_LIST  = [-1.0, -0.75, -0.5, -0.25]
KAPPA       = 0.0
SIGMA       = 0.5
SIGMA_N     = 0.05          # confirmed by ablation, 2026-05-08
N_PILOT     = 1_000         # subsampled samples for ACF pilot
N_FULL      = 10_000
SUBSAMPLE   = 600
N_EPOCHS    = 500
BATCH_SIZE  = 256
LR          = 1e-3
PATIENCE    = 50
HIDDEN      = 64
DEPTH       = 2             # 2 hidden layers → task.md architecture
N_QUERY     = 500           # noisy query points for Hessian constancy
N_LAMBDA    = 30
LAMBDA_GRID = np.logspace(-3, 1, N_LAMBDA)
ZERO_THRESH = 1e-8          # GLasso: |H_ij| < this → treated as zeroed
N_BOOTSTRAP = 1_000
VAL_FRAC    = 0.2
SEED        = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_NAMES = ["η", "s", "a", "μ"]


# ---------------------------------------------------------------------------
# Score network — SiLU MLP, task.md Phase 1B architecture
# ---------------------------------------------------------------------------
class ScoreNetSiLU(nn.Module):
    """R^4 → R^4, 2 hidden layers, 64 units, SiLU activation."""

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
    """E[‖s_θ(x̃) + ε/σ_n‖²]  (Vincent 2011)."""
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
    """Jacobian of s_θ at each query point = Hessian of log p. Returns (N,4,4)."""
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
    return np.array(Hs)   # (N, 4, 4)


def hessian_diagnostics(
    model: ScoreNetSiLU,
    X: np.ndarray,
    sigma_n: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    """
    Hessian constancy over N_QUERY noisy query points drawn randomly from X.
    For α < 0 (monostable, single basin) there is no separatrix threshold —
    all samples are 'within-basin'.
    """
    idx     = rng.integers(0, len(X), size=N_QUERY)
    x_clean = X[idx]
    x_noisy = x_clean + sigma_n * rng.standard_normal(x_clean.shape)

    t0 = time.time()
    H  = compute_hessians(model, x_noisy, device)   # (N_QUERY, 4, 4)
    print(f"    Jacobian ({N_QUERY} pts) in {time.time()-t0:.1f}s")

    diag          = np.diagonal(H, axis1=1, axis2=2)    # (N_QUERY, 4)
    mean_abs_diag = float(np.mean(np.abs(diag)))
    denom         = mean_abs_diag + 1e-12

    const_em   = float(np.std(H[:, 0, 3]) / denom)
    const_diag = float(np.mean(np.std(diag, axis=0)) / denom)

    abs_em    = np.abs(H[:, 0, 3])
    H_em_mean = float(abs_em.mean())
    boots     = np.array([
        rng.choice(abs_em, size=N_QUERY, replace=True).mean()
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
    Fit GraphicalLasso at each λ.  Identify blanket window = range of λ where
    H[0,3] is zeroed but all other off-diagonal entries remain nonzero.

    Off-diagonal pairs (upper triangle):
      (0,1) η–s   (0,2) η–a   (0,3) η–μ ← blanket entry
      (1,2) s–a   (1,3) s–μ   (2,3) a–μ

    Window condition: |(0,3)| < ZERO_THRESH AND all four ring edges nonzero.
    Ring edges only: (0,1) η–s, (0,2) η–a, (1,3) s–μ, (2,3) a–μ.
    H[1,2] (s–a) is excluded: it is also theoretically zero (s⊥a|{η,μ}
    by d-separation) and is always zeroed before or with H[0,3].
    """
    blanket_entry = (0, 3)
    other_offdiag = [(0, 1), (0, 2), (1, 3), (2, 3)]   # ring edges only

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

    # Window bounds
    if np.any(in_window):
        valid   = lambda_grid[in_window]
        lam_lo  = float(valid.min())
        lam_hi  = float(valid.max())
        width   = float(np.log10(lam_hi / lam_lo)) if lam_hi > lam_lo else 0.0
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
# ACF at lag 1 (in subsampled space)
# ---------------------------------------------------------------------------
def lag1_acf(X: np.ndarray) -> np.ndarray:
    acfs = []
    for j in range(X.shape[1]):
        x   = X[:, j] - X[:, j].mean()
        var = (x ** 2).mean()
        acfs.append(float((x[:-1] * x[1:]).mean() / var) if var > 1e-12 else 0.0)
    return np.array(acfs)


# ---------------------------------------------------------------------------
# H_emp
# ---------------------------------------------------------------------------
def compute_H_emp(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Sigma = np.cov(X.T)
    return np.linalg.inv(Sigma), Sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng    = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 2A Batch 1 — α ∈ {ALPHA_LIST}, κ={KAPPA}, σ={SIGMA}, σ_n={SIGMA_N}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    batch_rows: list[dict] = []

    for i_a, alpha in enumerate(ALPHA_LIST):
        # Encode α in filename: m for minus, p for plus
        sign_str = "m" if alpha < 0 else "p"
        tag      = f"alpha{sign_str}{abs(alpha):.2f}"

        print(f"\n{'='*65}")
        print(f"α = {alpha:+.2f}   κ = {KAPPA}   σ = {SIGMA}")
        print(f"{'='*65}")

        # ── PILOT ────────────────────────────────────────────────────────────
        print("\n[PILOT]")
        t0 = time.time()
        X_pilot, n_cross_p = euler_maruyama(
            alpha=alpha, kappa=KAPPA, sigma=SIGMA,
            n_steps=N_PILOT * SUBSAMPLE, subsample=SUBSAMPLE, seed=SEED,
        )
        print(f"  {N_PILOT} samples in {time.time()-t0:.1f}s")

        # crossing check — only meaningful for α≥0 (basin-to-basin transitions)
        # at α<0, μ=0 is the fixed point and noise crossings are not basin transitions
        if alpha >= 0:
            cross_ok = n_cross_p == 0
            print(f"  Basin crossings: {n_cross_p}"
                  + ("  PASS" if cross_ok else "  *** NONZERO ***"))
        else:
            cross_ok = True

        # ACF check (global criterion applies at α ≤ 0, DEVIATION 012)
        acf_p   = lag1_acf(X_pilot)
        acf_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_p))
        max_acf_p = float(np.max(np.abs(acf_p)))
        acf_ok    = max_acf_p < 0.05
        print(f"  ACF lag-1: {acf_str}")
        flag = "PASS" if acf_ok else "*** FAIL — unexpected for monostable α ***"
        print(f"  max ACF = {max_acf_p:.4f}  {flag}")

        # ── FULL RUN ─────────────────────────────────────────────────────────
        print("\n[FULL RUN]")
        t0 = time.time()
        X, n_cross = euler_maruyama(
            alpha=alpha, kappa=KAPPA, sigma=SIGMA,
            n_steps=N_FULL * SUBSAMPLE, subsample=SUBSAMPLE, seed=SEED + 1,
        )
        t_sim = time.time() - t0
        print(f"  N={N_FULL} samples in {t_sim:.1f}s")
        if alpha >= 0:
            print(f"  Basin crossings: {n_cross}"
                  + ("  PASS" if n_cross == 0 else "  *** NONZERO ***"))

        acf_full    = lag1_acf(X)
        max_acf_f   = float(np.max(np.abs(acf_full)))
        acf_str_f   = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(STATE_NAMES, acf_full))
        print(f"  ACF lag-1 (full): {acf_str_f}  max={max_acf_f:.4f}")

        # train / val split (rng advances per α → different permutation each time)
        n_val   = int(N_FULL * VAL_FRAC)
        n_train = N_FULL - n_val
        perm    = rng.permutation(N_FULL)
        X_tr, X_vl = X[perm[:n_train]], X[perm[n_train:]]

        # ── H_EMP ────────────────────────────────────────────────────────────
        print("\n[H_EMP]")
        H_emp, Sigma_emp = compute_H_emp(X)
        h03     = float(H_emp[0, 3])
        h03_rel = abs(h03) / float(np.max(np.abs(H_emp)))
        cond    = float(np.linalg.cond(Sigma_emp))
        eigs    = np.linalg.eigvalsh(Sigma_emp)
        sym_err = float(np.max(np.abs(H_emp - H_emp.T)))
        print(f"  H_emp[0,3] = {h03:+.6f}   |H[0,3]|/max = {h03_rel:.2e}")
        print(f"  Σ cond = {cond:.2f}   min_eig = {eigs.min():.4e}"
              + ("  PASS" if eigs.min() > 0 else "  *** NOT SPD ***"))
        print(f"  H_emp symmetry err = {sym_err:.2e}"
              + ("  PASS" if sym_err < 1e-12 else "  WARN"))

        # ── SCORE NETWORK ────────────────────────────────────────────────────
        print(f"\n[SCORE NETWORK]  σ_n = {SIGMA_N}")
        t0 = time.time()
        model, tr_hist, vl_hist, best_val, n_ep = train_network(
            X_tr, X_vl, SIGMA_N, device, seed=SEED
        )
        print(f"  Training: {time.time()-t0:.1f}s  ({n_ep} epochs)  best_val={best_val:.5f}")

        # Hessian constancy — random 500 query points from X (all samples, monostable)
        hnet_rng = np.random.default_rng(SEED + 300 + i_a)
        hdiag    = hessian_diagnostics(model, X, SIGMA_N, hnet_rng, device)
        ci       = hdiag["H_em_ci"]
        print(f"  const[0,3]  = {hdiag['const_em']:.4f}")
        print(f"  const[diag] = {hdiag['const_diag']:.4f}")
        print(f"  |Ĥ_ημ| mean = {hdiag['H_em_mean']:.5f}   "
              f"95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]")

        # ── GLASSO ───────────────────────────────────────────────────────────
        print(f"\n[GLASSO]  30-pt λ-sweep  logspace({np.log10(LAMBDA_GRID[0]):.0f},"
              f"{np.log10(LAMBDA_GRID[-1]):.0f})")
        t0    = time.time()
        gl    = glasso_sweep(X, LAMBDA_GRID)
        print(f"  Sweep in {time.time()-t0:.1f}s")
        if gl["lam_lo"] is not None:
            print(f"  Blanket window: [{gl['lam_lo']:.4f}, {gl['lam_hi']:.4f}]  "
                  f"width = {gl['width_dec']:.3f} dec")
        else:
            print(f"  *** No blanket window found ***")
        print(f"  λ values in window: {gl['n_in_window']}/{N_LAMBDA}")

        # ── KURTOSIS ─────────────────────────────────────────────────────────
        print("\n[KURTOSIS]")
        kurt = float(scipy_kurtosis(X[:, 3], fisher=True))
        print(f"  Excess kurtosis of μ = {kurt:.4f}  (Gaussian = 0)")

        # ── SAVE PER α ───────────────────────────────────────────────────────
        out_path = os.path.join(out_dir, f"phase2A_{tag}.npz")
        np.savez(
            out_path,
            # parameters
            alpha=alpha, kappa=KAPPA, sigma=SIGMA, sigma_n=SIGMA_N,
            n_pilot=N_PILOT, n_full=N_FULL, subsample=SUBSAMPLE,
            # pilot diagnostics
            acf_pilot=acf_p, max_acf_pilot=max_acf_p, acf_ok=acf_ok,
            n_cross_pilot=n_cross_p, cross_ok=cross_ok,
            # full-run diagnostics
            acf_full=acf_full, max_acf_full=max_acf_f, n_cross_full=n_cross,
            # H_emp
            H_emp=H_emp, Sigma_emp=Sigma_emp,
            H_emp_03=h03, H_emp_03_rel=h03_rel,
            Sigma_cond=cond, Sigma_min_eig=float(eigs.min()), H_emp_sym_err=sym_err,
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
            alpha       = alpha,
            acf_ok      = acf_ok,
            max_acf     = max_acf_p,
            n_cross_p   = n_cross_p,
            H_emp_03    = h03,
            H_emp_rel   = h03_rel,
            H_em_mean   = hdiag["H_em_mean"],
            H_em_ci_lo  = float(ci[0]),
            H_em_ci_hi  = float(ci[1]),
            const_em    = hdiag["const_em"],
            const_diag  = hdiag["const_diag"],
            gl_width    = gl["width_dec"],
            kurtosis_mu = kurt,
        ))

    # ── BATCH SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("BATCH 1 SUMMARY")
    print(f"{'='*65}")
    hdr = (f"{'α':>6}  {'ACF':>4}  {'H_emp[0,3]':>11}  {'|/max|':>8}  "
           f"{'|Ĥ_ημ|':>8}  {'GL_wid':>7}  {'kurt_μ':>7}")
    print(hdr)
    for r in batch_rows:
        acf_flag = "OK" if r["acf_ok"] else "FAIL"
        print(f"{r['alpha']:>6.2f}  {acf_flag:>4}  "
              f"{r['H_emp_03']:>+11.6f}  {r['H_emp_rel']:>8.2e}  "
              f"{r['H_em_mean']:>8.5f}  {r['gl_width']:>7.3f}  "
              f"{r['kurtosis_mu']:>7.4f}")

    # Save batch summary
    summary_path = os.path.join(out_dir, "phase2A_batch1_summary.npz")
    np.savez(
        summary_path,
        alpha_list    = np.array(ALPHA_LIST),
        H_emp_03      = np.array([r["H_emp_03"]    for r in batch_rows]),
        H_emp_rel     = np.array([r["H_emp_rel"]   for r in batch_rows]),
        H_em_mean     = np.array([r["H_em_mean"]   for r in batch_rows]),
        H_em_ci_lo    = np.array([r["H_em_ci_lo"]  for r in batch_rows]),
        H_em_ci_hi    = np.array([r["H_em_ci_hi"]  for r in batch_rows]),
        const_em      = np.array([r["const_em"]    for r in batch_rows]),
        const_diag    = np.array([r["const_diag"]  for r in batch_rows]),
        gl_width      = np.array([r["gl_width"]    for r in batch_rows]),
        kurtosis_mu   = np.array([r["kurtosis_mu"] for r in batch_rows]),
        acf_ok        = np.array([r["acf_ok"]      for r in batch_rows]),
        max_acf       = np.array([r["max_acf"]     for r in batch_rows]),
    )
    print(f"\nBatch 1 summary saved → {summary_path}")


if __name__ == "__main__":
    main()
