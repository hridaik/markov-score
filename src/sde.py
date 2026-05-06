# State vector convention (fixed for entire project):
# index 0 : η  — external (extracellular concentration)
# index 1 : s  — sensory  (receptor)
# index 2 : a  — active   (secretor)
# index 3 : μ  — internal (intracellular)

import numpy as np
from lyapunov import DEFAULTS


def euler_maruyama(
    alpha=DEFAULTS["alpha"],
    kappa=DEFAULTS["kappa"],
    sigma=DEFAULTS["sigma"],
    gamma_eta=DEFAULTS["gamma_eta"],
    gamma_s=DEFAULTS["gamma_s"],
    gamma_a=DEFAULTS["gamma_a"],
    c1=DEFAULTS["c1"],
    c2=DEFAULTS["c2"],
    c3=DEFAULTS["c3"],
    c4=DEFAULTS["c4"],
    dt=0.01,
    n_burn=100_000,
    n_steps=1_000_000,
    subsample=100,
    x0=None,
    seed=42,
):
    """
    Euler-Maruyama integrator for the 4D chemosensing SDE.

    Returns
    -------
    X : (n_steps // subsample, 4)  — subsampled trajectory after burn-in
    n_crossings : int — number of times μ changes sign (basin crossings for α>0)
    """
    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.zeros(4)

    x = np.array(x0, dtype=np.float64)
    sqrt_dt_sigma = sigma * np.sqrt(dt)
    mu_prev = x[3]
    n_crossings = 0

    # Burn-in
    dW = np.empty(4)
    for _ in range(n_burn):
        rng.standard_normal(out=dW)
        eta, s, a, mu = x[0], x[1], x[2], x[3]
        x[0] += (-gamma_eta * eta + c1 * a + kappa * mu) * dt + sqrt_dt_sigma * dW[0]
        x[1] += (-gamma_s   * s   + c2 * eta)             * dt + sqrt_dt_sigma * dW[1]
        x[2] += (-gamma_a   * a   + c3 * mu)              * dt + sqrt_dt_sigma * dW[2]
        x[3] += (alpha * mu - mu**3 + c4 * s - kappa * eta) * dt + sqrt_dt_sigma * dW[3]

    mu_prev = x[3]

    # Production run with subsampling
    n_out = n_steps // subsample
    X = np.empty((n_out, 4))
    out_idx = 0

    for step in range(n_steps):
        rng.standard_normal(out=dW)
        eta, s, a, mu = x[0], x[1], x[2], x[3]
        x[0] += (-gamma_eta * eta + c1 * a + kappa * mu) * dt + sqrt_dt_sigma * dW[0]
        x[1] += (-gamma_s   * s   + c2 * eta)             * dt + sqrt_dt_sigma * dW[1]
        x[2] += (-gamma_a   * a   + c3 * mu)              * dt + sqrt_dt_sigma * dW[2]
        x[3] += (alpha * mu - mu**3 + c4 * s - kappa * eta) * dt + sqrt_dt_sigma * dW[3]

        if x[3] * mu_prev < 0:
            n_crossings += 1
        mu_prev = x[3]

        if (step + 1) % subsample == 0:
            X[out_idx] = x
            out_idx += 1

    return X, n_crossings
