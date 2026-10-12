"""Random subspaces matched on captured variance: the control for a route's bottleneck share.

A k-dimensional route through a predictor space X (a relation's canonical
coordinates, a block's top-k scores, anything linear in X) keeps some share of
what X says about a target. Whether that share is *special* needs a control:
random k-dimensional subspaces of X. But a route's share of the target tracks
how much of X's variance the route captures, so the random subspaces must
capture the same amount, or the comparison reads variance as content.

Uniform random orthonormal k-subspaces cannot be matched: they concentrate near
k/d of the variance (l-hub, 2026-10-10: a V route at .234 against uniform draws
at .193-.196). So draws are **tilted**: Gaussian directions in X's PCA basis
scaled by ``eigenvalue ** (gamma / 2)``. gamma < 0 favours low-variance
directions and gamma > 0 high-variance ones. gamma is first bracketed by a
sweep (mean captured share per gamma on a grid) and then sampled uniformly
inside the bracket that straddles the target share. From a pool of draws, the
``n`` nearest in captured share are kept, and the match is refused if the n-th
nearest is more than ``tol`` away.

This is the sampler only. What is read through the subspaces (held-out ridge
R², a CCA, a decoding score) is the caller's, so the control is the same for
every reader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psytwill.exceptions import SpaceError


@dataclass
class MatchedSubspaces:
    bases: list[np.ndarray]
    """``n`` orthonormal (d x k) bases, nearest-in-share first."""
    shares: np.ndarray
    """Each basis's captured share of X's variance on the matching rows."""
    target_share: float
    gamma_bracket: tuple[float, float]

    @property
    def share_range(self) -> tuple[float, float]:
        return float(self.shares.min()), float(self.shares.max())


def _rng(random_state) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def captured_share(X: np.ndarray, Q: np.ndarray) -> float:
    """Share of the (centred) rows' total variance lying in the span of orthonormal ``Q``."""
    Xc = X - X.mean(0)
    return float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum())


def route_share(X: np.ndarray, route: np.ndarray) -> float:
    """Share of X's variance captured by a route linear in X.

    ``route`` holds the route's coordinates on the same rows (n x k). Its span in
    X is recovered by least squares (route ≈ X B), so a route given as
    coordinates, not as a basis, is measured exactly when it is linear in X.
    """
    Xc = X - X.mean(0)
    B, *_ = np.linalg.lstsq(Xc, route - route.mean(0), rcond=None)
    Q, _ = np.linalg.qr(B)
    return float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum())


def matched_subspaces(X: np.ndarray, k: int, target_share: float, *, n: int = 1000,
                      n_pool: int = 20000, tol: float = 0.01, gamma_grid: np.ndarray | None = None,
                      sweep_reps: int = 10, random_state=0) -> MatchedSubspaces:
    """``n`` random k-subspaces of X whose captured variance share is nearest ``target_share``.

    ``X`` holds the rows the match is made on (the training rows of the read
    that uses the subspaces); it is centred here. ``random_state`` may be a
    ``RandomState``, which is consumed in a fixed order: the gamma sweep
    (``len(gamma_grid) * sweep_reps`` draws), then the pool (one uniform gamma
    and one Gaussian block per draw).
    """
    if not 0 < k < X.shape[1]:
        raise SpaceError(f"k={k} must lie strictly between 0 and X's {X.shape[1]} columns")
    if n > n_pool:
        raise SpaceError(f"n={n} nearest cannot exceed the pool of {n_pool} draws")
    rng = _rng(random_state)
    Xc = X - X.mean(0)
    tot = (Xc ** 2).sum()
    evals, evecs = np.linalg.eigh(np.cov(Xc, rowvar=False))
    evals = np.clip(evals, evals.max() * 1e-12, None)
    d = X.shape[1]

    def draw(g):
        G = rng.standard_normal((d, k)) * (evals ** (g / 2))[:, None]
        Q, _ = np.linalg.qr(evecs @ G)
        return Q

    grid = np.linspace(-4, 4, 81) if gamma_grid is None else np.asarray(gamma_grid, dtype=float)
    mean_share = np.array([np.mean([((Xc @ draw(g)) ** 2).sum() / tot for _ in range(sweep_reps)])
                           for g in grid])
    below, above = grid[mean_share <= target_share], grid[mean_share >= target_share]
    if not len(below) or not len(above):
        raise SpaceError(
            f"no gamma in [{grid.min():g}, {grid.max():g}] brackets the target share {target_share:.4f} "
            f"(sweep spans {mean_share.min():.4f}-{mean_share.max():.4f}); widen gamma_grid")
    g_lo, g_hi = below.max() - 0.1, above.min() + 0.1
    lo, hi = min(g_lo, g_hi), max(g_lo, g_hi)
    cand = []
    for _ in range(n_pool):
        g = rng.uniform(lo, hi)
        G = rng.standard_normal((d, k)) * (evals ** (g / 2))[:, None]
        Q, _ = np.linalg.qr(evecs @ G)
        cand.append((abs(float(((Xc @ Q) ** 2).sum() / tot) - target_share), Q))
    cand.sort(key=lambda c: c[0])
    if cand[n - 1][0] > tol:
        raise SpaceError(
            f"variance matching failed: the {n}-th nearest random subspace is {cand[n - 1][0]:.4f} from the "
            f"target share {target_share:.4f} (tol {tol}); enlarge n_pool or lower n")
    bases = [Q for _, Q in cand[:n]]
    shares = np.array([float(((Xc @ Q) ** 2).sum() / tot) for Q in bases])
    return MatchedSubspaces(bases=bases, shares=shares, target_share=float(target_share),
                            gamma_bracket=(float(lo), float(hi)))
