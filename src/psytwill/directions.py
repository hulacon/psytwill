"""Per-direction properties of a fitted block: how its members share each direction, and how stable it is.

A block (:class:`psytwill.space.BlockMap`) is a PCA over its members'
whitened columns side by side, so each direction's loading vector splits into
one part per member. This module reads two things off each direction d:

- **Member energy** ``e_md``: the squared norm of d's loading on member m's
  columns. The loading vectors are unit-norm, so ``e_md`` sums to 1 over the
  members. It is a share of d's direction, in the block's whitened metric;
  it says nothing about how well a member predicts d.
- **Universality** ``u_d``: the participation ratio of d's member energies,
  over the member count, ``(sum_m e_md)^2 / sum_m e_md^2 / M``. It runs from
  ``1/M`` (one member carries d) to 1 (every member equally). It is symmetric:
  no member predicts another, so a one-column member cannot be trivially
  "shared". It can hold at most one direction's worth of energy.

And, from two fits of the same block on disjoint halves of the fit corpus:

- **Reliability** ``rho_d``: the correlation of d's scores between the two
  half fits, each aligned to the reference fit (:func:`direction_reliability`).
  A low-eigenvalue direction that one member carries alone may be a noisy
  estimate, and that alone would lower any score read from it.

:func:`direction_table` writes one row per direction (``component``,
``eigenvalue``, ``u``, optionally ``rho``), the per-direction table that
mmmdata's ``scripts/braintwill_encoding/universality.py read`` takes;
:func:`member_energy` keeps the per-member energies in long form, for splits
by member type.

    python -m psytwill.directions <fit manifest.json> --out <table.csv> [--energy <energy.csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from psytwill.exceptions import SpaceError

#: How each half fit is aligned to the reference before its scores are correlated.
ALIGNMENTS = ("procrustes", "sign")


def _member_slices(block_map) -> dict[str, slice]:
    out, start = {}, 0
    for m in block_map.members:
        w = int(block_map.whiteners[m].rank)
        out[m] = slice(start, start + w)
        start += w
    if start != block_map.block_components.shape[1]:
        raise SpaceError(f"the members' whitened ranks sum to {start}, but the block's components have "
                         f"{block_map.block_components.shape[1]} columns")
    return out


def member_energy(block_map, k: int | None = None) -> pd.DataFrame:
    """Each direction's energy on each member: long form, one row per (component, member).

    ``energy`` is normalised over members, so it sums to 1 per component (the
    loading vectors are unit-norm, so this is a guard against rounding, not a
    rescale). ``width`` is the member's whitened rank.
    """
    V = np.asarray(block_map.block_components, dtype=float)
    k = V.shape[0] if k is None else min(int(k), V.shape[0])
    sl = _member_slices(block_map)
    E = np.stack([(V[:k, s] ** 2).sum(axis=1) for s in sl.values()], axis=1)  # (k, M)
    tot = E.sum(axis=1, keepdims=True)
    if not (tot > 0).all():
        raise SpaceError("a block direction has zero loading on every member")
    E = E / tot
    members = list(sl)
    return pd.DataFrame({
        "component": np.repeat(np.arange(k), len(members)),
        "member": members * k,
        "width": [sl[m].stop - sl[m].start for m in members] * k,
        "energy": E.ravel(),
    })


def universality(energy: np.ndarray) -> np.ndarray:
    """``u_d`` from a (directions x members) energy matrix: participation ratio over member count."""
    E = np.asarray(energy, dtype=float)
    if E.ndim != 2 or E.shape[1] < 1:
        raise SpaceError(f"energy must be (directions x members); got shape {E.shape}")
    if (E < 0).any():
        raise SpaceError("member energies must be non-negative")
    s = E.sum(axis=1)
    q = (E ** 2).sum(axis=1)
    if not (q > 0).all():
        raise SpaceError("a direction has zero energy on every member")
    return (s ** 2) / q / E.shape[1]


def _align(S: np.ndarray, ref: np.ndarray, how: str) -> np.ndarray:
    if how == "procrustes":
        # orthogonal R minimising ||S R - ref||: S's k directions rotated onto the reference's
        U, _, Vt = np.linalg.svd(S.T @ ref, full_matrices=False)
        return S @ (U @ Vt)
    # sign: direction d stays direction d, flipped to correlate positively with the reference's d
    sgn = np.sign(np.einsum("ij,ij->j", S, ref))
    sgn[sgn == 0] = 1.0
    return S * sgn


def direction_reliability(reference, half_a, half_b, *, align: str = "procrustes") -> np.ndarray:
    """``rho_d``: correlation of each direction's scores between two half fits, after aligning both to ``reference``.

    All three are (rows x k) block scores on the **same rows** (typically every
    row, projected through each fit). Each half is centred and aligned to the
    centred reference, then direction d's correlation between the aligned
    halves is ``rho_d``. Rows with a non-finite value in any of the three are
    dropped.

    ``align``:

    - ``"procrustes"`` (default): one orthogonal rotation of the half's k
      directions onto the reference's. It absorbs sign flips and rotations
      among near-equal eigenvalues, so ``rho_d`` asks whether the half's
      k-dimensional span reproduces the reference's direction d. It is lenient
      inside the span: a direction is penalised only as far as the halves'
      spans differ from the reference's around it.
    - ``"sign"``: no rotation, each half's direction d flipped to agree with
      the reference's d. Strict: two near-equal eigenvalues that swap order
      between halves score low although the span is stable.
    """
    if align not in ALIGNMENTS:
        raise SpaceError(f"align must be one of {', '.join(ALIGNMENTS)}; got {align!r}")
    R, A, B = (np.asarray(x, dtype=float) for x in (reference, half_a, half_b))
    if R.ndim != 2 or A.shape != R.shape or B.shape != R.shape:
        raise SpaceError(f"reference and both halves must be (rows x k) of one shape; got "
                         f"{R.shape}, {A.shape}, {B.shape}")
    keep = np.isfinite(R).all(1) & np.isfinite(A).all(1) & np.isfinite(B).all(1)
    if keep.sum() < 3:
        raise SpaceError(f"only {int(keep.sum())} rows are finite in the reference and both halves")
    R, A, B = (x[keep] - x[keep].mean(0) for x in (R, A, B))
    a, b = _align(A, R, align), _align(B, R, align)
    num = np.einsum("ij,ij->j", a, b)
    den = np.sqrt(np.einsum("ij,ij->j", a, a) * np.einsum("ij,ij->j", b, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def direction_table(block_map, k: int | None = None, rho=None) -> pd.DataFrame:
    """One row per direction: ``component``, ``eigenvalue``, ``u`` and, when given, ``rho``."""
    E = member_energy(block_map, k)
    kk = int(E["component"].max()) + 1
    mat = E["energy"].to_numpy(float).reshape(kk, -1)
    out = pd.DataFrame({
        "component": np.arange(kk),
        "eigenvalue": np.asarray(block_map.block_eigenvalues, dtype=float)[:kk],
        "u": universality(mat),
    })
    if rho is not None:
        r = np.asarray(rho, dtype=float)
        if r.shape != (kk,):
            raise SpaceError(f"rho has shape {r.shape}; the table has {kk} directions")
        out["rho"] = r
    return out


def main(argv=None) -> int:
    from psytwill.space import load_fit

    ap = argparse.ArgumentParser(prog="python -m psytwill.directions",
                                 description="Per-direction table (component, eigenvalue, u) of a fitted block.")
    ap.add_argument("manifest", help="a block fit's manifest (.json beside its .npz)")
    ap.add_argument("--out", required=True, help="table path (.csv or .parquet)")
    ap.add_argument("--energy", help="also write the long per-member energies here")
    ap.add_argument("--k", type=int, help="only the first k directions (default: all the fit kept)")
    a = ap.parse_args(argv)
    fit = load_fit(a.manifest)
    t = direction_table(fit.map, a.k)
    frames = [(Path(a.out), t)] + ([(Path(a.energy), member_energy(fit.map, a.k))] if a.energy else [])
    for path, df in frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False) if path.suffix == ".parquet" else df.to_csv(path, index=False)
    print(f"wrote {a.out}: {len(t)} directions over {len(fit.members)} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
