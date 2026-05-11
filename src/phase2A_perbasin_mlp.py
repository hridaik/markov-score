"""
Phase 2A per-basin MLP Hessian evaluation
α ∈ {0.50, 0.75, 1.00}, κ=0, σ=0.5

Model weights were not saved by Batch 2 / ACF re-run, so this script
re-simulates X and re-trains the score network with identical hyperparameters
and seed, then evaluates the per-basin Hessian.

Subsamples match the original runs:
  α=0.50: sub=600  (Batch 2, seed=SEED+1)
  α=0.75: sub=1200 (ACF re-run, DEVIATION 012, seed=SEED+1)
  α=1.00: sub=1200 (ACF re-run, DEVIATION 012, seed=SEED+1)

For each α and each basin (μ>0 and μ<0 separately):
  - N_QUERY=500 query points from within-basin deep region (|μ|>MU_BASIN_THRESH)
  - σ_n=0.05 Gaussian noise added
  - Ĥ_prec[i,j] ≈ −∂s_θ_i/∂x̃_j  (Hessian of log p; H_prec = −Jac(s_θ))
  - Report signed mean Ĥ_prec[0,3] with 95% bootstrap CI on the mean
  - Report relative magnitude |mean Ĥ_prec[0,3]| / mean|Ĥ_prec_diag|
  - Report constancy std(Jac[0,3]) / mean|Jac_diag|  (< 0.1 threshold)

Scientific target at κ=0: H_prec[0,3] = 0 in each basin (blanket intact).
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
# Constants — must match Batch 2 / ACF re-run hyperparameters exactly
# ---------------------------------------------------------------------------
KAPPA           = 0.0
SIGMA           = 0.5
SIGMA_N         = 0.05
N_FULL          = 10_000
SEED            = 42
N_QUERY         = 500
MU_BASIN_THRESH = 0.3
VAL_FRAC        = 0.2
N_BOOTSTRAP     = 1_000
N_EPOCHS        = 500
BATCH_SIZE      = 256
LR              = 1e-3
PATIENCE        = 50
HIDDEN          = 64
DEPTH           = 2

ALPHA_LIST      = [0.50, 0.75, 1.00]
ALPHA_SUBSAMPLE = {0.50: 600, 0.75: 1200, 1.00: 1200}

BLANKET_THRESH  = 0.05   # |H_prec[0,3]| / mean|H_prec_diag| < this → PASS
CONST_THRESH    = 0.10   # std(Jac[0,3]) / mean|Jac_diag| < this → PASS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Score network — identical to Batch 2 / ACF re-run
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
    sn = model.sigma_n
    eps = torch.randn_like(x_batch)
    return ((model(x_batch + sn * eps) + eps / sn) ** 2).sum(1).mean()


def train_network(X_tr: np.ndarray, X_vl: np.ndarray, device: torch.device) -> tuple:
    torch.manual_seed(SEED)
    model  = ScoreNetSiLU(SIGMA_N).to(device)
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
    return model, best_val, epoch + 1


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------
def compute_jacobian(
    model: ScoreNetSiLU, X_query: np.ndarray, device: torch.device
) -> np.ndarray:
    """
    Returns (N, 4, 4) Jacobian of s_θ at each query point.
    H_prec[i,j] ≈ −Jac[i,j]  (for Gaussian: Jac(∇log p) = −H_prec).
    """
    model.eval()
    Xq = torch.tensor(X_query, dtype=torch.float32).to(device)
    Js = []
    for i in range(len(Xq)):
        xi = Xq[i:i + 1].requires_grad_(True)
        si = model(xi).squeeze(0)
        rows = [
            torch.autograd.grad(si[k], xi, retain_graph=(k < 3), create_graph=False)[0]
            .squeeze(0).detach().cpu()
            for k in range(4)
        ]
        Js.append(torch.stack(rows).numpy())
    return np.array(Js)


# ---------------------------------------------------------------------------
# Per-basin evaluation
# ---------------------------------------------------------------------------
def eval_basin(
    model: ScoreNetSiLU,
    X_deep: np.ndarray,
    label: str,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    """
    Evaluate score-network Hessian within one basin.
    X_deep: samples already filtered to this basin with |μ|>MU_BASIN_THRESH.
    """
    n_pool = len(X_deep)
    n_use  = min(N_QUERY, n_pool)
    if n_pool < N_QUERY:
        print(f"  {label}: only {n_pool} deep-basin points (need {N_QUERY}), using all")
    idx     = rng.integers(0, n_pool, size=n_use)
    x_clean = X_deep[idx]
    x_noisy = x_clean + SIGMA_N * rng.standard_normal(x_clean.shape)

    t0  = time.time()
    Jac = compute_jacobian(model, x_noisy, device)   # (n_use, 4, 4)
    print(f"    Jacobian ({n_use} pts) in {time.time()-t0:.1f}s")

    # H_prec[i,j] = -Jac[i,j]; report signed values for interpretation
    h03_vals = -Jac[:, 0, 3]                          # (n_use,) H_prec[0,3] per query
    jac_diag = np.diagonal(Jac, axis1=1, axis2=2)     # (n_use, 4)
    mean_abs_diag = float(np.mean(np.abs(jac_diag))) + 1e-12

    mean_h03  = float(h03_vals.mean())
    h03_rel   = float(np.abs(h03_vals).mean() / mean_abs_diag)
    const_em  = float(np.std(Jac[:, 0, 3]) / mean_abs_diag)   # uses Jac convention (same as Batch 2)

    boots = np.array([
        rng.choice(h03_vals, size=n_use, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    ci = np.percentile(boots, [2.5, 97.5])

    blanket_ok = h03_rel < BLANKET_THRESH
    const_ok   = const_em < CONST_THRESH

    print(f"  {label}  n_pool={n_pool}  n_query={n_use}")
    print(f"    mean H_prec[0,3] = {mean_h03:+.5f}  95% CI = [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(f"    |H_prec[0,3]|/mean|H_diag| = {h03_rel:.4f}  "
          f"({'PASS' if blanket_ok else 'FAIL'} < {BLANKET_THRESH})")
    print(f"    constancy std(Jac[0,3])/mean|Jac_diag| = {const_em:.4f}  "
          f"({'PASS' if const_ok else 'FAIL'} < {CONST_THRESH})")

    return dict(
        label      = label,
        n_pool     = n_pool,
        n_query    = n_use,
        mean_h03   = mean_h03,
        ci_lo      = float(ci[0]),
        ci_hi      = float(ci[1]),
        h03_rel    = h03_rel,
        const_em   = const_em,
        blanket_ok = blanket_ok,
        const_ok   = const_ok,
        h03_vals   = h03_vals,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 2A per-basin MLP Hessian — α ∈ {ALPHA_LIST}")
    print(f"κ={KAPPA}  σ={SIGMA}  σ_n={SIGMA_N}  N_query={N_QUERY}  |μ|>{MU_BASIN_THRESH}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []

    for i_a, alpha in enumerate(ALPHA_LIST):
        sub = ALPHA_SUBSAMPLE[alpha]
        print(f"\n{'='*65}")
        print(f"α = {alpha:+.2f}   subsample = {sub}")
        print(f"{'='*65}")

        # Simulate — same seed as original run
        print("\n[SIMULATE]")
        t0 = time.time()
        X, _ = euler_maruyama(
            alpha=alpha, kappa=KAPPA, sigma=SIGMA,
            n_steps=N_FULL * sub, subsample=sub, seed=SEED + 1,
        )
        print(f"  N={N_FULL} in {time.time()-t0:.1f}s  (subsample={sub}  seed={SEED+1})")

        # Train/val split — fresh rng per alpha to avoid cross-alpha dependency
        rng_split = np.random.default_rng(SEED + 200 + i_a)
        perm      = rng_split.permutation(N_FULL)
        n_val     = int(N_FULL * VAL_FRAC)
        X_tr      = X[perm[:-n_val]]
        X_vl      = X[perm[-n_val:]]

        # Train score network
        print(f"\n[SCORE NETWORK]  σ_n={SIGMA_N}")
        t0 = time.time()
        model, best_val, n_ep = train_network(X_tr, X_vl, device)
        print(f"  Training: {time.time()-t0:.1f}s  ({n_ep} epochs)  best_val={best_val:.5f}")

        # Basin split — within-basin deep region
        mu         = X[:, 3]
        deep_mask  = np.abs(mu) > MU_BASIN_THRESH
        X_pos_deep = X[(mu > 0) & deep_mask]
        X_neg_deep = X[(mu < 0) & deep_mask]
        print(f"\n[BASIN SPLIT]  |μ|>{MU_BASIN_THRESH}")
        print(f"  μ>0 deep: {len(X_pos_deep)}  μ<0 deep: {len(X_neg_deep)}")

        # Evaluate each basin
        print(f"\n[PER-BASIN HESSIAN]")
        rng_eval = np.random.default_rng(SEED + 400 + i_a)

        if len(X_pos_deep) >= 10:
            res_pos = eval_basin(model, X_pos_deep, "μ>0", rng_eval, device)
        else:
            print(f"  μ>0: *** {len(X_pos_deep)} deep points — skipping ***")
            res_pos = dict(
                label="μ>0", n_pool=len(X_pos_deep), n_query=0,
                mean_h03=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                h03_rel=np.nan, const_em=np.nan,
                blanket_ok=False, const_ok=False, h03_vals=np.array([]),
            )

        if len(X_neg_deep) >= 10:
            res_neg = eval_basin(model, X_neg_deep, "μ<0", rng_eval, device)
        else:
            print(f"  μ<0: *** {len(X_neg_deep)} deep points — skipping ***")
            res_neg = dict(
                label="μ<0", n_pool=len(X_neg_deep), n_query=0,
                mean_h03=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                h03_rel=np.nan, const_em=np.nan,
                blanket_ok=False, const_ok=False, h03_vals=np.array([]),
            )

        # Basin asymmetry check (Z₂ symmetry → |h03_pos| should ≈ |h03_neg|)
        if not (np.isnan(res_pos["mean_h03"]) or np.isnan(res_neg["mean_h03"])):
            ratio = abs(res_pos["mean_h03"]) / (abs(res_neg["mean_h03"]) + 1e-12)
            print(f"\n  Basin asymmetry: |H_pos[0,3]|/|H_neg[0,3]| = {ratio:.3f}  "
                  f"(Z₂ → expect ≈ 1)")

        results.append(dict(alpha=alpha, sub=sub, pos=res_pos, neg=res_neg))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-BASIN MLP HESSIAN SUMMARY")
    print(f"{'='*65}")
    hdr = (f"  {'α':>5}  {'basin':>5}  {'mean H[0,3]':>12}  "
           f"{'95% CI':>22}  {'|H03|/diag':>10}  {'const':>6}  {'bkt':>5}  {'cst':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        for res in (r["pos"], r["neg"]):
            def _f(v: float, fmt: str = "+.5f") -> str:
                return f"{v:{fmt}}" if not np.isnan(v) else "    —    "
            bk = "PASS" if res["blanket_ok"] else "FAIL"
            ct = "PASS" if res["const_ok"]   else "FAIL"
            ci = f"[{_f(res['ci_lo'])}, {_f(res['ci_hi'])}]" if not np.isnan(res["ci_lo"]) else "         —         "
            print(f"  {r['alpha']:>5.2f}  {res['label']:>5}  "
                  f"{_f(res['mean_h03']):>12}  {ci:>22}  "
                  f"{_f(res['h03_rel'], '.4f'):>10}  {_f(res['const_em'], '.4f'):>6}  "
                  f"{bk:>5}  {ct:>5}")

    # Save
    def _arr(key: str, basin: str) -> np.ndarray:
        return np.array([r[basin][key] for r in results])

    out_path = os.path.join(out_dir, "phase2A_perbasin_mlp.npz")
    h03_save: dict = {}
    for i_a, (r, alpha) in enumerate(zip(results, ALPHA_LIST)):
        tag = f"alpha{int(alpha*100):03d}"
        h03_save[f"h03_vals_{tag}_pos"] = r["pos"]["h03_vals"]
        h03_save[f"h03_vals_{tag}_neg"] = r["neg"]["h03_vals"]

    np.savez(
        out_path,
        alpha_list = np.array(ALPHA_LIST),
        subsamples = np.array([ALPHA_SUBSAMPLE[a] for a in ALPHA_LIST]),
        mean_h03   = np.column_stack([_arr("mean_h03", "pos"), _arr("mean_h03", "neg")]),
        ci_lo      = np.column_stack([_arr("ci_lo",    "pos"), _arr("ci_lo",    "neg")]),
        ci_hi      = np.column_stack([_arr("ci_hi",    "pos"), _arr("ci_hi",    "neg")]),
        h03_rel    = np.column_stack([_arr("h03_rel",  "pos"), _arr("h03_rel",  "neg")]),
        const_em   = np.column_stack([_arr("const_em", "pos"), _arr("const_em","neg")]),
        blanket_ok = np.column_stack([_arr("blanket_ok","pos"),_arr("blanket_ok","neg")]),
        const_ok   = np.column_stack([_arr("const_ok", "pos"), _arr("const_ok", "neg")]),
        n_query    = np.column_stack([_arr("n_query",  "pos"), _arr("n_query",  "neg")]),
        **h03_save,
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
