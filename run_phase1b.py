#!/usr/bin/env python3
"""
Phase 1B full sweep: denoising score matching over 11 κ values.

DEVIATION 007 applied:
  (a) σ_n fixed to 0.05 — raw DSM loss CV is invalid; σ_n=0.05 gives
      <3% off-diagonal attenuation (σ_n²=0.0025 ≪ Σ_min≈0.10).
  (b) Hessian and score mean evaluated at noisy query points
      x̃ = x + 0.05ε, matching the training distribution.

For each κ:
  1. Simulate 6M steps (subsample=600) → N=10,000 decorrelated samples
  2. Split 80/20 train/val
  3. Train ScoreNet (500 epochs) at σ_n=0.05
  4. Compute score mean at noisy points, Hessian at 500 noisy query points
  5. H[0,3] diagnostics vs Phase 0A Lyapunov ground truth
  6. Save per-κ .npz + final 11-point summary

κ grid matches Phase 0A: np.linspace(0, 0.5, 11)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lyapunov import DEFAULTS
from sde import euler_maruyama
from score_network import ScoreNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KAPPA_VALUES = np.linspace(0.0, 0.5, 11)   # matches Phase 0A first 11 points
SIGMA_N_FIXED = 0.05                         # DEVIATION 007a: fixed, no sweep
N_EPOCHS = 500
BATCH_SIZE = 512
LR = 1e-3
N_QUERY = 500

# Phase 0A ground truth (H[0,3] at each κ)
_p0a = np.load("results/phase0/phase0A_sweep.npz")
_p0a_kappas = _p0a["kappas"]
_p0a_H03 = _p0a["H_eta_mu"]
H_LYAP_03_GT = np.interp(KAPPA_VALUES, _p0a_kappas, _p0a_H03)

# ---------------------------------------------------------------------------
# DSM loss
# ---------------------------------------------------------------------------

def dsm_loss_gpu(model, x_batch):
    """Denoising score matching loss. x_batch on DEVICE."""
    sigma_n = model.sigma_n
    eps = torch.randn_like(x_batch)
    x_noisy = x_batch + sigma_n * eps
    score = model(x_noisy)
    target = -eps / sigma_n
    return ((score - target) ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(X_train_gpu, X_val_gpu, seed=0):
    """
    Train ScoreNet for N_EPOCHS at SIGMA_N_FIXED.
    Returns (model, val_loss).
    """
    torch.manual_seed(seed)
    model = ScoreNet(hidden=128, depth=3, sigma_n=SIGMA_N_FIXED).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS,
                                                            eta_min=1e-5)
    N = len(X_train_gpu)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, BATCH_SIZE):
            batch = X_train_gpu[perm[start:start + BATCH_SIZE]]
            opt.zero_grad()
            dsm_loss_gpu(model, batch).backward()
            opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = dsm_loss_gpu(model, X_val_gpu).item()

    return model, val_loss


# ---------------------------------------------------------------------------
# Score mean at noisy points (DEVIATION 007b)
# ---------------------------------------------------------------------------

def compute_score_mean_noisy(model, X_np, rng):
    """E_{x̃}[s_θ(x̃)] where x̃ = x + σ_n ε. Should be near zero."""
    eps = rng.standard_normal(X_np.shape).astype(np.float32)
    X_noisy = X_np + SIGMA_N_FIXED * eps
    X_noisy_gpu = torch.tensor(X_noisy, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model(X_noisy_gpu).cpu().numpy().mean(axis=0)


# ---------------------------------------------------------------------------
# Batched Jacobian of score = Hessian of log p_σ, at NOISY query points
# (DEVIATION 007b)
# ---------------------------------------------------------------------------

def compute_hessian_batch(model, X_query_noisy_np, chunk=64):
    """
    Returns H_samples (Q, 4, 4) and H_mean (4, 4).
    X_query_noisy_np: already noised x̃ = x_q + σ_n ε, shape (Q, 4).
    """
    model.eval()
    Q = len(X_query_noisy_np)
    H_all = np.zeros((Q, 4, 4), dtype=np.float32)

    for start in range(0, Q, chunk):
        end = min(start + chunk, Q)
        x = torch.tensor(X_query_noisy_np[start:end], dtype=torch.float32,
                          device=DEVICE).requires_grad_(True)
        s = model(x)          # (B, 4)
        for k in range(4):
            grads = torch.autograd.grad(
                s[:, k].sum(), x, retain_graph=(k < 3)
            )[0]              # (B, 4)
            H_all[start:end, k, :] = grads.detach().cpu().numpy()
        x = x.detach()

    return H_all, H_all.mean(axis=0)


# ---------------------------------------------------------------------------
# Per-κ run
# ---------------------------------------------------------------------------

def run_kappa(kappa, kappa_idx):
    print(f"\n[κ={kappa:.3f}  ({kappa_idx+1}/11)] Simulating...", flush=True)
    t0 = time.time()
    X, n_cross = euler_maruyama(
        kappa=kappa, alpha=DEFAULTS["alpha"], sigma=DEFAULTS["sigma"],
        n_steps=6_000_000, subsample=600, seed=42
    )
    print(f"  sim: {time.time()-t0:.1f}s  n={len(X)}  μ_crossings={n_cross}",
          flush=True)

    # Train/val split
    n_tr = int(0.8 * len(X))
    X_tr_gpu = torch.tensor(X[:n_tr],  dtype=torch.float32, device=DEVICE)
    X_va_gpu = torch.tensor(X[n_tr:],  dtype=torch.float32, device=DEVICE)

    # Train at fixed σ_n (DEVIATION 007a)
    print(f"  Training σ_n={SIGMA_N_FIXED} × {N_EPOCHS} epochs...", flush=True)
    t1 = time.time()
    model, val_loss = train_one(X_tr_gpu, X_va_gpu, seed=kappa_idx)
    print(f"  val_loss={val_loss:.4f}  {time.time()-t1:.1f}s", flush=True)

    # Theoretical minimum loss for reference
    Sigma_emp = np.cov(X.T)
    sigma_data_sq = float(np.trace(Sigma_emp) / 4)
    sn2 = SIGMA_N_FIXED ** 2
    L_star = 4 / sn2 - 4 / (sigma_data_sq + sn2)
    val_over_Lstar = val_loss / L_star if L_star > 0 else float("nan")
    print(f"  L*={L_star:.3f}  val/L*={val_over_Lstar:.4f}", flush=True)

    model.eval()
    rng = np.random.default_rng(42 + kappa_idx)

    # Score mean at noisy points (DEVIATION 007b)
    sm = compute_score_mean_noisy(model, X, rng)
    sm_norm = float(np.linalg.norm(sm))

    # Noisy query points (DEVIATION 007b)
    q_idx = rng.choice(len(X), size=N_QUERY, replace=False)
    X_query_base = X[q_idx]
    eps_q = rng.standard_normal(X_query_base.shape).astype(np.float32)
    X_query_noisy = X_query_base + SIGMA_N_FIXED * eps_q

    # Hessian at noisy query points
    t2 = time.time()
    H_samples, H_mean = compute_hessian_batch(model, X_query_noisy)
    print(f"  Hessian: {time.time()-t2:.1f}s  n_query={N_QUERY}", flush=True)

    # Diagnostics
    H03_abs = float(abs(H_mean[0, 3]))
    H03_rel = float(H03_abs / (np.abs(H_mean).max() + 1e-12))
    asym_per = (np.linalg.norm(H_samples - H_samples.transpose(0, 2, 1),
                                axis=(1, 2)) /
                (np.linalg.norm(H_samples, axis=(1, 2)) + 1e-12))
    asym_mean = float(asym_per.mean())
    H03_gt = float(H_LYAP_03_GT[kappa_idx])

    print(
        f"  |Ĥ[0,3]|={H03_abs:.4e}  |Ĥ[0,3]|/max={H03_rel:.4e}  "
        f"asym={asym_mean:.4f}  ‖E[s̃]‖={sm_norm:.4f}  "
        f"H_lyap[0,3]={H03_gt:.4e}",
        flush=True,
    )

    # Per-κ save
    os.makedirs("results/phase1", exist_ok=True)
    np.savez(
        f"results/phase1/phase1B_kappa_{kappa:.3f}.npz",
        H_mean=H_mean,
        H_samples=H_samples,
        X_query=X_query_noisy,
        H03_abs=np.float64(H03_abs),
        H03_rel=np.float64(H03_rel),
        hessian_asymmetry=np.float64(asym_mean),
        score_mean=sm,
        score_mean_norm=np.float64(sm_norm),
        sigma_n=np.float64(SIGMA_N_FIXED),
        val_loss=np.float64(val_loss),
        val_over_Lstar=np.float64(val_over_Lstar),
        H_lyap_03=np.float64(H03_gt),
        kappa=np.float64(kappa),
    )

    return dict(
        H_mean=H_mean,
        H03_abs=H03_abs,
        H03_rel=H03_rel,
        hessian_asymmetry=asym_mean,
        score_mean_norm=sm_norm,
        val_loss=val_loss,
        val_over_Lstar=val_over_Lstar,
        H_lyap_03=H03_gt,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
    print(f"κ values ({len(KAPPA_VALUES)}): {KAPPA_VALUES}", flush=True)
    print(f"σ_n (fixed): {SIGMA_N_FIXED}  [DEVIATION 007a]", flush=True)
    print(f"N_epochs={N_EPOCHS}  batch={BATCH_SIZE}  N_query={N_QUERY}  lr={LR}",
          flush=True)
    print("Hessian evaluated at noisy query points  [DEVIATION 007b]", flush=True)

    results = []
    t_total = time.time()

    for i, kappa in enumerate(KAPPA_VALUES):
        results.append(run_kappa(kappa, i))

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Summary save (11-point curves for plotting vs Phase 0A)
    np.savez(
        "results/phase1/phase1B_sweep.npz",
        kappa_values=KAPPA_VALUES,
        sigma_n=np.float64(SIGMA_N_FIXED),
        H_hat_mean=np.array([r["H_mean"] for r in results]),      # (11,4,4)
        H03_abs=np.array([r["H03_abs"] for r in results]),        # (11,)
        H03_rel=np.array([r["H03_rel"] for r in results]),        # (11,)
        hessian_asymmetry=np.array([r["hessian_asymmetry"] for r in results]),
        score_mean_norm=np.array([r["score_mean_norm"] for r in results]),
        val_loss=np.array([r["val_loss"] for r in results]),
        val_over_Lstar=np.array([r["val_over_Lstar"] for r in results]),
        H_lyap_03=np.array([r["H_lyap_03"] for r in results]),    # Phase 0A GT
    )
    print("Saved to results/phase1/phase1B_sweep.npz", flush=True)

    # Quick summary table
    print("\n--- Summary ---")
    print(f"{'κ':>6}  {'val/L*':>7}  {'|Ĥ[0,3]|':>10}  "
          f"{'|Ĥ[0,3]|/max':>13}  {'asym':>6}  "
          f"{'‖E[s̃]‖':>7}  {'H_lyap':>10}")
    for i, (kappa, r) in enumerate(zip(KAPPA_VALUES, results)):
        print(f"{kappa:6.3f}  {r['val_over_Lstar']:7.4f}  "
              f"{r['H03_abs']:10.4e}  {r['H03_rel']:13.4e}  "
              f"{r['hessian_asymmetry']:6.4f}  {r['score_mean_norm']:7.4f}  "
              f"{r['H_lyap_03']:10.4e}")
