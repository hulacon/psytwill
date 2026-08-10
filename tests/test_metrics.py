"""Metric functions: known values, NaN propagation, registry."""

import numpy as np
import pytest

from psytwill.exceptions import MetricError
from psytwill.metrics import (
    correlation,
    cosine,
    euclidean,
    get_metric,
    spearman,
)


def test_cosine_identity_and_orthogonal():
    X = np.eye(3)
    M = cosine(X, X)
    assert np.allclose(M, np.eye(3))


def test_cosine_renormalizes():
    X = np.array([[2.0, 0.0]])
    Y = np.array([[0.5, 0.0]])
    assert cosine(X, Y)[0, 0] == pytest.approx(1.0)


def test_cosine_zero_row_convention():
    # crossmodal.py convention: zero vector -> similarity 0, not NaN
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    M = cosine(X, X)
    assert M[0, 1] == pytest.approx(0.0)


def test_correlation_known_value():
    X = np.array([[1.0, 2.0, 3.0]])
    Y = np.array([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0]])
    M = correlation(X, Y)
    assert M[0, 0] == pytest.approx(1.0)
    assert M[0, 1] == pytest.approx(-1.0)


def test_correlation_is_scale_and_shift_invariant():
    rng = np.random.RandomState(0)
    X = rng.randn(3, 5)
    assert np.allclose(correlation(X, X), correlation(X * 3 + 7, X))


def test_spearman_monotone_invariance():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    Y = np.array([[1.0, 10.0, 100.0, 1000.0]])  # monotone transform of X
    assert spearman(X, Y)[0, 0] == pytest.approx(1.0)


def test_euclidean_known_values():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    M = euclidean(X, X)
    assert M[0, 0] == pytest.approx(0.0)
    assert M[0, 1] == pytest.approx(5.0)
    assert M[1, 0] == pytest.approx(5.0)


@pytest.mark.parametrize("func", [cosine, correlation, spearman, euclidean])
def test_nan_row_propagates(func):
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [0.5, 0.5]])
    M = func(X, X)
    assert np.isnan(M[1, :]).all()
    assert np.isnan(M[:, 1]).all()
    assert not np.isnan(M[0, 0]) and not np.isnan(M[2, 0])


def test_registry_forms():
    assert get_metric("cosine").form == "similarity"
    assert get_metric("euclidean").form == "distance"


def test_unknown_metric_raises():
    with pytest.raises(MetricError, match="Unknown metric"):
        get_metric("manhattan")
