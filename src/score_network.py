# Denoising score matching for the 4D chemosensing SDE.
# State convention: (η=0, s=1, a=2, μ=3) — fixed throughout.

import numpy as np
import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    """Small MLP: R^4 → R^4, approximating ∇log p(x)."""

    def __init__(self, hidden=128, depth=3, sigma_n=0.1):
        super().__init__()
        self.sigma_n = sigma_n
        layers = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def dsm_loss(model, x_batch):
    """
    Denoising score matching loss (Vincent 2011).
    L = E[‖s_θ(x + σ_n ε) + ε/σ_n‖²]
    """
    sigma_n = model.sigma_n
    eps = torch.randn_like(x_batch)
    x_noisy = x_batch + sigma_n * eps
    score = model(x_noisy)              # s_θ(x̃)
    target = -eps / sigma_n             # ∇log p_σ(x̃|x)
    return ((score - target) ** 2).sum(dim=1).mean()


def train_score_network(
    X_samples,
    sigma_n=0.1,
    n_epochs=500,
    batch_size=256,
    lr=1e-3,
    hidden=128,
    depth=3,
    seed=0,
    verbose_every=50,
):
    """
    Train ScoreNet on X_samples (N, 4) numpy array.

    Returns (model, loss_history).
    """
    torch.manual_seed(seed)
    model = ScoreNet(hidden=hidden, depth=depth, sigma_n=sigma_n)
    X = torch.tensor(X_samples, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(X)
    loss_history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            batch = X[idx]
            opt.zero_grad()
            loss = dsm_loss(model, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / n_batches
        loss_history.append(avg)
        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  epoch {epoch+1:4d}/{n_epochs}  loss={avg:.5f}")

    return model, loss_history


def score_mean(model, X_samples):
    """E_p[s_θ(x)] — should be near zero for a well-trained network."""
    X = torch.tensor(X_samples, dtype=torch.float32)
    with torch.no_grad():
        s = model(X).numpy()
    return s.mean(axis=0)


def hessian_at_points(model, X_query):
    """
    Jacobian of s_θ at each query point = Hessian of log p.
    Returns H_samples (Q, 4, 4).
    """
    X = torch.tensor(X_query, dtype=torch.float32)
    H_list = []
    for i in range(len(X)):
        xi = X[i].unsqueeze(0).requires_grad_(True)
        si = model(xi).squeeze(0)   # (4,)
        rows = []
        for k in range(4):
            grad_k = torch.autograd.grad(si[k], xi, retain_graph=(k < 3))[0]
            rows.append(grad_k.squeeze(0).detach())
        H_list.append(torch.stack(rows).numpy())   # (4, 4)
    return np.array(H_list)
