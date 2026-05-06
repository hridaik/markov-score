# State vector convention (fixed for entire project):
# index 0 : η  — external (extracellular concentration)
# index 1 : s  — sensory  (receptor)
# index 2 : a  — active   (secretor)
# index 3 : μ  — internal (intracellular)
#
# Solenoidal coupling: J[0,3] = +κ, J[3,0] = −κ
# Blanket entries to monitor: H[0,3] = H[3,0] (must be zero at κ=0)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lyapunov import DEFAULTS


def drift(x, alpha=DEFAULTS["alpha"], kappa=DEFAULTS["kappa"],
          gamma_eta=DEFAULTS["gamma_eta"], gamma_s=DEFAULTS["gamma_s"],
          gamma_a=DEFAULTS["gamma_a"],
          c1=DEFAULTS["c1"], c2=DEFAULTS["c2"],
          c3=DEFAULTS["c3"], c4=DEFAULTS["c4"]):
    """
    Drift vector f(x). x has shape (..., 4); returns same shape.
    State order: (eta=0, s=1, a=2, mu=3).
    """
    eta, s, a, mu = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return np.stack([
        -gamma_eta * eta + c1 * a + kappa * mu,
        -gamma_s   * s   + c2 * eta,
        -gamma_a   * a   + c3 * mu,
        alpha * mu - mu**3 + c4 * s - kappa * eta,
    ], axis=-1)


def build_fpe_matrix(grids, sigma=DEFAULTS["sigma"], **drift_kwargs):
    """
    Build sparse FPE operator A (N×N) such that A p = 0 at steady state,
    with Dirichlet p=0 boundary conditions.

    Scheme:
    - Drift: first-order upwind (guarantees M-matrix → non-negative solution)
    - Diffusion: second-order central differences

    Boundary rows: identity (Dirichlet p=0 enforced via column references in
    interior rows that reference boundary columns with value 0).

    All triplets are accumulated as numpy arrays — no Python-level loops.

    grids : list of 4 uniform 1D arrays.
    Returns (A_csr, dx, shape).
    """
    n = [len(g) for g in grids]
    N = int(np.prod(n))
    dx = float(grids[0][1] - grids[0][0])

    # Full-grid coordinates — shape (N, 4)
    mesh = np.meshgrid(*grids, indexing="ij")
    coords = np.stack([m.ravel() for m in mesh], axis=1)

    f = drift(coords, **drift_kwargs)   # (N, 4)
    D_coeff = sigma**2 / 2.0

    strides = np.array([n[1]*n[2]*n[3], n[2]*n[3], n[3], 1], dtype=np.int64)

    flat = np.arange(N, dtype=np.int64)

    # Index of each point along each dimension
    idx_along = [(flat // strides[d]) % n[d] for d in range(4)]

    # Interior mask: not on any face of the 4D hypercube
    is_interior = np.ones(N, dtype=bool)
    for d in range(4):
        is_interior &= (idx_along[d] > 0) & (idx_along[d] < n[d] - 1)

    interior = flat[is_interior]   # flat indices of interior points
    boundary = flat[~is_interior]

    # Pre-allocate lists for COO triplets
    rows_list, cols_list, vals_list = [], [], []

    for d in range(4):
        stride = int(strides[d])
        fd = f[interior, d]     # drift at interior centre points

        idx_p = interior + stride
        idx_m = interior - stride

        # --- Non-conservation upwind drift: −f_d ∂p/∂x_d ---
        # Diagonal:       −|f_d(x_i)|/dx
        # Off-diagonal at upwind neighbour: +|f_d(x_i)|/dx (≥ 0 → M-matrix)
        #
        # The non-conservation form omits the compressibility term −(∇·f)p.
        # That term is added separately as a diagonal correction below (once,
        # after all dimensions are processed) so that the full equation is:
        #   −f·∇p − (∇·f)p + D∇²p = −∇·(fp) + D∇²p = 0   ✓

        rows_list.append(interior);  cols_list.append(interior)
        vals_list.append(-np.abs(fd) / dx)

        pos = fd > 0
        if pos.any():
            rows_list.append(interior[pos]);  cols_list.append(idx_m[pos])
            vals_list.append(fd[pos] / dx)

        neg = fd < 0
        if neg.any():
            rows_list.append(interior[neg]);  cols_list.append(idx_p[neg])
            vals_list.append(-fd[neg] / dx)

        # --- Central-difference diffusion: D ∂²p/∂x_d² ---
        Ddx2 = D_coeff / dx**2
        n_int = len(interior)

        rows_list.append(interior);  cols_list.append(interior)
        vals_list.append(np.full(n_int, -2.0 * Ddx2))

        rows_list.append(interior);  cols_list.append(idx_m)
        vals_list.append(np.full(n_int, Ddx2))

        rows_list.append(interior);  cols_list.append(idx_p)
        vals_list.append(np.full(n_int, Ddx2))

    # --- Compressibility correction: −(∇·f) p added to diagonal ---
    # The non-conservation drift form computes −f·∇p but the FPE requires
    # −∇·(fp) = −f·∇p − (∇·f)p. The missing term is −(∇·f)p.
    # For our system:
    #   ∂f_η/∂η = −γ_η,  ∂f_s/∂s = −γ_s,  ∂f_a/∂a = −γ_a
    #   ∂f_μ/∂μ = α − 3μ²   (nonlinear; varies with μ)
    #   ∇·f = −γ_η − γ_s − γ_a + α − 3μ²
    # Correction = −∇·f = γ_η + γ_s + γ_a − α + 3μ²
    mu_int = coords[interior, 3]
    neg_div_f = (drift_kwargs.get("gamma_eta", DEFAULTS["gamma_eta"])
                 + drift_kwargs.get("gamma_s",   DEFAULTS["gamma_s"])
                 + drift_kwargs.get("gamma_a",   DEFAULTS["gamma_a"])
                 - drift_kwargs.get("alpha",     DEFAULTS["alpha"])
                 + 3.0 * mu_int**2)
    rows_list.append(interior);  cols_list.append(interior)
    vals_list.append(neg_div_f)

    # Boundary rows: identity (Dirichlet p = 0 at all faces)
    rows_list.append(boundary);  cols_list.append(boundary)
    vals_list.append(np.ones(len(boundary)))

    A = sp.csr_matrix(
        (np.concatenate(vals_list),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
    )
    return A, dx, tuple(n)


def solve_fpe(grids, sigma=DEFAULTS["sigma"],
              dt=0.005, tol=1e-8, max_iter=5000,
              **drift_kwargs):
    """
    Solve the stationary FPE A p = 0 with ∫p dV = 1.

    Power iteration (explicit Euler on ∂p/∂t = Ap) with renormalisation at
    each step. Avoids the normalization-row / ILU-preconditioner interaction
    that caused GMRES to find incorrect solutions. Convergence to the unique
    non-negative null vector is guaranteed for M-matrices (Perron-Frobenius).

    Stability: the iteration p ← p + dt·Ap is non-negativity-preserving when
    dt ≤ 1/D_max, where D_max = max|A[i,i]| over interior points. The info
    dict reports D_max and whether the chosen dt satisfies this bound.

    Returns (p_grid shaped as (n0,n1,n2,n3), info_dict).
    """
    A, dx, shape = build_fpe_matrix(grids, sigma=sigma, **drift_kwargs)
    N = int(np.prod(shape))
    n = shape
    dV = dx**4

    # Identify interior points (same logic as build_fpe_matrix)
    strides = np.array([n[1]*n[2]*n[3], n[2]*n[3], n[3], 1], dtype=np.int64)
    flat = np.arange(N, dtype=np.int64)
    is_interior = np.ones(N, dtype=bool)
    for d in range(4):
        idx_d = (flat // strides[d]) % n[d]
        is_interior &= (idx_d > 0) & (idx_d < n[d] - 1)
    interior = flat[is_interior]

    # Stability bound: dt must satisfy 1 + dt*A[i,i] >= 0 for all interior i,
    # i.e. dt < 1/D_max.  Violating this makes the unclipped step produce
    # negative values at corner points; clipping may then mask divergence.
    diag = A.diagonal()
    D_max = float(np.abs(diag[interior]).max())
    dt_stable = 1.0 / D_max
    if dt >= dt_stable:
        raise ValueError(
            f"dt={dt} violates stability bound dt < 1/D_max = {dt_stable:.6f} "
            f"(D_max={D_max:.4f}). Reduce dt below {dt_stable:.6f}."
        )

    # Uniform initialisation over interior; boundary stays 0 (Dirichlet BC)
    p = np.zeros(N)
    p[interior] = 1.0
    p /= p.sum() * dV

    # Power iteration
    n_iters = 0
    final_err = np.inf
    for i in range(max_iter):
        p_new = p + dt * (A @ p)
        p_new = np.clip(p_new, 0.0, None)
        total = p_new.sum() * dV
        if total > 0:
            p_new /= total
        final_err = float(
            np.max(np.abs(p_new - p)) / (np.max(np.abs(p)) + 1e-300)
        )
        p = p_new
        n_iters = i + 1
        if final_err < tol:
            break

    converged = (final_err < tol)

    # True residual of converged solution (unmodified A, no normalisation row)
    Ap = A @ p
    true_residual = float(np.abs(Ap).max())

    # Marginal density of μ: integrate p over η, s, a dimensions
    p_grid = p.reshape(shape)
    mu_marginal = p_grid.sum(axis=(0, 1, 2)) * dx**3

    integration_error = float(abs(p.sum() * dV - 1.0))

    return p_grid, dict(
        n_iters=n_iters,
        final_err=final_err,
        converged=converged,
        integration_error=integration_error,
        true_residual=true_residual,
        mu_marginal=mu_marginal,
        mu_grid=grids[3],
        D_max=D_max,
        dt_stable=dt_stable,
        dt_used=dt,
        dt_ok=(dt < dt_stable),
        dx=dx,
        dV=dV,
        shape=shape,
    )


def compute_hessian_log_p(p_grid, grids, sample_indices=None, rng_seed=42):
    """
    Hessian of log p at sampled interior grid points via central differences.

    Returns:
        H_samples : (n_pts, 4, 4)
        coords    : (n_pts, 4)
        sample_indices : flat indices used
    """
    shape = p_grid.shape
    N = int(np.prod(shape))
    dx = float(grids[0][1] - grids[0][0])
    strides = [shape[1]*shape[2]*shape[3], shape[2]*shape[3], shape[3], 1]

    log_p = np.log(np.clip(p_grid, 1e-300, None)).ravel()

    flat = np.arange(N)
    is_interior = np.ones(N, dtype=bool)
    for d in range(4):
        stride = strides[d]
        nd = shape[d]
        idx_d = (flat // stride) % nd
        # Need margin 1 for first derivatives, 1 for mixed (total margin 2? No.)
        # Mixed partial uses ±1 in each of two dimensions simultaneously,
        # so we need margin ≥ 1 in every dimension.
        is_interior &= (idx_d >= 1) & (idx_d <= nd - 2)

    interior_flat = np.where(is_interior)[0]

    if sample_indices is None:
        rng = np.random.default_rng(rng_seed)
        # Only sample points where p is significant (avoids log(near-zero) artefacts).
        # Threshold: p > p_max * 1e-4 so finite differences of log p are not
        # dominated by the 1e-300 floor. In the tails, adjacent grid points can
        # straddle the floor boundary, producing enormous spurious Hessian entries.
        p_flat_local = p_grid.ravel()
        p_threshold = p_flat_local.max() * 1e-4
        high_density = interior_flat[p_flat_local[interior_flat] > p_threshold]
        n_sample = min(1000, len(high_density))
        sample_indices = rng.choice(high_density, size=n_sample, replace=False)

    n_pts = len(sample_indices)
    H_samples = np.zeros((n_pts, 4, 4))
    coords = np.zeros((n_pts, 4))

    for i, k in enumerate(sample_indices):
        mi = np.unravel_index(k, shape)
        coords[i] = [grids[d][mi[d]] for d in range(4)]

        for d in range(4):
            s = strides[d]
            H_samples[i, d, d] = (log_p[k+s] - 2*log_p[k] + log_p[k-s]) / dx**2

        for d1 in range(4):
            for d2 in range(d1+1, 4):
                s1, s2 = strides[d1], strides[d2]
                val = (log_p[k+s1+s2] - log_p[k+s1-s2]
                       - log_p[k-s1+s2] + log_p[k-s1-s2]) / (4*dx**2)
                H_samples[i, d1, d2] = val
                H_samples[i, d2, d1] = val

    return H_samples, coords, sample_indices
