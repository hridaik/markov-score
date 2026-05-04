# State vector convention (fixed for entire project):
# index 0 : η  — external (extracellular concentration)
# index 1 : s  — sensory  (receptor)
# index 2 : a  — active   (secretor)
# index 3 : μ  — internal (intracellular)
#
# Solenoidal coupling: J[0,3] = +κ, J[3,0] = −κ
# Blanket entries to monitor: H[0,3] = H[3,0] (must be zero at κ=0)

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals

# Default parameters — c values changed from task.md (DEVIATION 001).
# task.md c=1.0 produces zero eigenvalue (ring gain = damping product exactly).
# c=0.5 gives ρ = (0.5)^4 = 0.0625, comfortably stable.
# All other values unchanged from task.md.
DEFAULTS = dict(
    gamma_eta=1.0,
    gamma_s=1.0,
    gamma_a=1.0,
    c1=0.5,
    c2=0.5,
    c3=0.5,
    c4=0.5,
    sigma=0.5,
    alpha=-1.0,
    kappa=0.0,
)


def build_jacobian(kappa, alpha=DEFAULTS["alpha"],
                   gamma_eta=DEFAULTS["gamma_eta"],
                   gamma_s=DEFAULTS["gamma_s"],
                   gamma_a=DEFAULTS["gamma_a"],
                   c1=DEFAULTS["c1"], c2=DEFAULTS["c2"],
                   c3=DEFAULTS["c3"], c4=DEFAULTS["c4"]):
    """
    Jacobian of the linearised system at x*=0 (valid for alpha < 0).
    Row i = derivative of f_i; column j = derivative w.r.t. x_j.
    State order: (eta=0, s=1, a=2, mu=3).

    At kappa > 0: J[0,3] = +kappa, J[3,0] = -kappa (antisymmetric, solenoidal).
    """
    J = np.array([
        [-gamma_eta,      0.0,   c1,       kappa],
        [       c2, -gamma_s,   0.0,         0.0],
        [      0.0,      0.0, -gamma_a,      c3 ],
        [   -kappa,       c4,   0.0,       alpha],
    ], dtype=float)
    return J


def ring_gain_ratio(alpha=DEFAULTS["alpha"],
                    gamma_eta=DEFAULTS["gamma_eta"],
                    gamma_s=DEFAULTS["gamma_s"],
                    gamma_a=DEFAULTS["gamma_a"],
                    c1=DEFAULTS["c1"], c2=DEFAULTS["c2"],
                    c3=DEFAULTS["c3"], c4=DEFAULTS["c4"]):
    """
    ρ = c₁c₂c₃c₄ / (γ_η·γ_s·γ_a·|α|).
    ρ < 1 is required for stability of the linearised ring at κ=0.
    ρ→∞ as α→0⁺; stop Lyapunov shortcut if ρ > 0.8 (use FPE instead).
    """
    return (c1 * c2 * c3 * c4) / (gamma_eta * gamma_s * gamma_a * abs(alpha))


# Maximum allowed ρ before the Lyapunov solve is considered unreliable.
RHO_LIMIT = 0.8


def solve_lyapunov(kappa, sigma=DEFAULTS["sigma"], **kwargs):
    """
    Solve JΣ + ΣJᵀ = −σ²I for steady-state covariance Σ.
    Returns (Sigma, H, eigenvalues_of_J, condition_number_of_Sigma).

    Raises ValueError if J is not Hurwitz or if ρ > RHO_LIMIT (near stability boundary).
    """
    rho = ring_gain_ratio(**{k: kwargs[k] for k in kwargs if k in
                              ("alpha", "gamma_eta", "gamma_s", "gamma_a",
                               "c1", "c2", "c3", "c4")})
    if rho > RHO_LIMIT:
        raise ValueError(
            f"ρ = {rho:.4f} > {RHO_LIMIT} at kappa={kappa}. "
            "System is too close to the stability boundary for Lyapunov solve. "
            "Use FPE numerics (Phase 0B) instead."
        )

    J = build_jacobian(kappa, **kwargs)
    ev = eigvals(J)

    if np.any(np.real(ev) >= 0):
        raise ValueError(f"J is not Hurwitz at kappa={kappa}: eigenvalues = {ev}")

    D = sigma ** 2 * np.eye(4)
    # scipy: solve_continuous_lyapunov(A, Q) solves AX + XA^H = Q
    # We want JΣ + ΣJᵀ = -D, so pass Q = -D
    Sigma = solve_continuous_lyapunov(J, -D)

    # Symmetry check
    asym = np.max(np.abs(Sigma - Sigma.T))
    if asym > 1e-12:
        raise RuntimeError(f"Sigma asymmetry {asym:.2e} > 1e-12 at kappa={kappa}")

    # Positive-definiteness check
    eigvals_sigma = np.linalg.eigvalsh(Sigma)
    if np.any(eigvals_sigma <= 0):
        raise RuntimeError(f"Sigma not positive definite at kappa={kappa}: min eig = {eigvals_sigma.min():.2e}")

    cond = np.linalg.cond(Sigma)
    H = np.linalg.inv(Sigma)

    # Symmetry check on H
    asym_H = np.max(np.abs(H - H.T))
    if asym_H > 1e-12:
        raise RuntimeError(f"H asymmetry {asym_H:.2e} > 1e-12 at kappa={kappa}")

    return Sigma, H, ev, cond


def run_phase0A(kappa_values, sigma=DEFAULTS["sigma"], seed=42,
                sigma_min_floor=1e-6, cond_ceiling=1000.0, **kwargs):
    """
    Sweep over kappa values and collect Lyapunov results.

    Safety stops (raise RuntimeError):
    - min eigenvalue of Sigma drops below sigma_min_floor
    - condition number of Sigma exceeds cond_ceiling

    Returns a dict suitable for np.savez, including:
    - H_eta_mu        : H[0,3] (signed)
    - H_eta_mu_ratio  : |H[0,3]| / max|H_ij|  (relative criterion, DEVIATION 002)
    - min_eig_sigma   : min eigenvalue of Sigma at each kappa
    - eig_real/imag   : all four J eigenvalues at each kappa
    - cond_numbers    : condition number of Sigma
    """
    kappas = np.array(kappa_values, dtype=float)
    n = len(kappas)

    H_all = np.zeros((n, 4, 4))
    Sigma_all = np.zeros((n, 4, 4))
    H_eta_mu = np.zeros(n)
    H_eta_mu_ratio = np.zeros(n)
    min_eig_sigma = np.zeros(n)
    cond_numbers = np.zeros(n)
    eig_real = np.zeros((n, 4))
    eig_imag = np.zeros((n, 4))

    for i, kappa in enumerate(kappas):
        Sigma, H, ev, cond = solve_lyapunov(kappa, sigma=sigma, **kwargs)

        min_eig = np.linalg.eigvalsh(Sigma).min()

        if min_eig < sigma_min_floor:
            raise RuntimeError(
                f"SAFETY STOP at kappa={kappa:.3f}: "
                f"min eigenvalue of Sigma = {min_eig:.2e} < {sigma_min_floor:.2e}. "
                "Report before continuing."
            )
        if cond > cond_ceiling:
            raise RuntimeError(
                f"SAFETY STOP at kappa={kappa:.3f}: "
                f"condition number of Sigma = {cond:.1f} > {cond_ceiling:.0f}. "
                "Report before continuing."
            )

        H_all[i] = H
        Sigma_all[i] = Sigma
        H_eta_mu[i] = H[0, 3]
        H_eta_mu_ratio[i] = abs(H[0, 3]) / np.max(np.abs(H))
        min_eig_sigma[i] = min_eig
        cond_numbers[i] = cond
        eig_real[i] = np.real(ev)
        eig_imag[i] = np.imag(ev)

    return dict(
        kappas=kappas,
        H_all=H_all,
        Sigma_all=Sigma_all,
        H_eta_mu=H_eta_mu,
        H_eta_mu_ratio=H_eta_mu_ratio,
        min_eig_sigma=min_eig_sigma,
        cond_numbers=cond_numbers,
        eig_real=eig_real,
        eig_imag=eig_imag,
        sigma=sigma,
        seed=42,
    )
