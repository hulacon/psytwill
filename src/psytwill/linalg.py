"""Linear-algebra helpers shared by the whiteners and the block fits."""

from __future__ import annotations

import warnings

import numpy as np


def thin_svd(M: np.ndarray, compute_uv: bool = True):
    """``np.linalg.svd(M, full_matrices=False)``, retried with LAPACK ``gesvd``
    when the default divide-and-conquer driver (``gesdd``) does not converge.

    ``gesdd`` can fail on exactly rank-deficient input; MEASURED 2026-11-25
    (psytwill-space 0.3.0 scoping) on an n x p member-space view of rank k fed
    to a whitener. ``gesvd`` is slower and converges on it. The fallback warns,
    so a run log says when it was taken; the factors mean the same either way.
    """
    try:
        return np.linalg.svd(M, full_matrices=False, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        warnings.warn(f"SVD (gesdd) did not converge on a {M.shape[0]} x {M.shape[1]} matrix; "
                      "retried with gesvd", RuntimeWarning, stacklevel=2)
        return svd(M, full_matrices=False, compute_uv=compute_uv, lapack_driver="gesvd")
