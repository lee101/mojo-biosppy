"""Parity tests for the banded linear algebra behind
`detrend_smoothness_priors`.

Upstream builds the dense `t x t` matrix `I + lambda^2 D2^T D2` and inverts it.
The matrix is symmetric with bandwidth 2, so the port factors the three bands
instead.  These tests pin the band construction, the factorization, and the end
to end result against the dense reference.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye, spdiags

import mojo_biosppy as mbp
from mojo_biosppy import _lib


def _dense_system(t, smoothing_factor):
    """Upstream's matrix, dense, exactly as `detrend_smoothness_priors` builds it."""
    aux1 = np.dot(np.ones((t, 1)), np.array([[1.0, -2.0, 1.0]]))
    d2 = spdiags(aux1.T, [0, 1, 2], t - 2, t)
    aux2 = smoothing_factor ** 2 * d2.T * d2
    identity = eye(t)
    return np.asarray(csr_matrix(identity) + aux2.todense())


def _dense_d2(t):
    """Upstream's `D2`, built exactly the way `detrend_smoothness_priors` does."""
    aux1 = np.dot(np.ones((t, 1)), np.array([[1.0, -2.0, 1.0]]))
    return spdiags(aux1.T, [0, 1, 2], t - 2, t)


@pytest.mark.parametrize("t", [5, 6, 7, 8, 17, 40, 129])
def test_d2_bands_match_the_dense_second_difference_operator(t):
    bands = _lib.d2_bands(t)
    d2 = _dense_d2(t)
    full = np.asarray((d2.T @ d2).todense())
    for q in range(3):
        for r in range(t):
            c = r + q
            want = full[r, c] if c < t else 0.0
            assert bands[q, r] == want, (t, q, r)


def test_d2_bands_of_a_long_signal_have_the_convolution_stencil():
    t = 64
    bands = _lib.d2_bands(t)
    # [1,-2,1] convolved with itself, away from the truncated tail.
    # band 0 is the main diagonal, so it carries the centre of the stencil
    stencil = np.array([6.0, -4.0, 1.0])
    # The last two rows of D2 are truncated, so the interior stencil only
    # holds away from both ends: band q needs `q + 2` rows on the left and
    # stops `q + 1` rows before the end.
    for q, (lo, hi) in enumerate(((2, t - 3), (1, t - 2), (0, t - 2))):
        np.testing.assert_array_equal(bands[q, lo:hi],
                                      np.full(hi - lo, stencil[q]))
    # the truncated tail is asymmetric: row t-1 has no centre tap
    assert bands[0, -1] == 1.0 and bands[0, -2] == 5.0
    assert bands[1, -1] == 0.0 and bands[1, -2] == -2.0


def test_banded_factorisation_reproduces_the_dense_matrix():

    t = 96
    m = _dense_system(t, 10.0)
    # A symmetric positive-definite matrix with bandwidth 2 is determined by
    # its three lower bands.
    bands = np.empty((3, t), dtype=np.float64)
    bands[0] = np.diag(m)
    bands[1][: t - 1] = np.diag(m, -1)
    bands[2][: t - 2] = np.diag(m, -2)
    _lib.band_chol_factor(bands)
    factor = np.zeros((t, t))
    factor[np.arange(t), np.arange(t)] = bands[0]
    for q in (1, 2):
        rows = np.arange(t - q)
        factor[rows + q, rows] = bands[q][: t - q]
    np.testing.assert_allclose(factor @ factor.T, m, rtol=1e-11, atol=1e-12)
    # Random factors would still reconstruct something; a wrong band layout
    # would not reconstruct this specific matrix.
    assert not np.allclose(factor, np.eye(t))


@pytest.mark.parametrize("t", [5, 33, 100])
def test_banded_solve_matches_dense_inverse(t):
    rng = np.random.default_rng(404 + t)
    m = _dense_system(t, 7.0)
    rhs = rng.standard_normal(t)
    bands = np.empty((3, t), dtype=np.float64)
    bands[0] = np.diag(m)
    bands[1][: t - 1] = np.diag(m, -1)
    bands[2][: t - 2] = np.diag(m, -2)
    _lib.band_chol_factor(bands)
    got = _lib.band_chol_solve(bands, rhs)
    np.testing.assert_allclose(m @ got, rhs, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(got, np.linalg.solve(m, rhs), rtol=1e-9,
                               atol=1e-10)


def test_banded_factorisation_reports_a_bad_pivot():
    # A diagonal-dominant-breaking matrix with a zero leading pivot must be
    # reported, not silently produce NaNs.
    bands = np.zeros((3, 5), dtype=np.float64)
    bands[0, 0] = 0.0
    bands[1, 0] = 1.0
    with pytest.raises(np.linalg.LinAlgError):
        _lib.band_chol_factor(bands)


@pytest.mark.parametrize("t", [6, 25, 64, 120])
@pytest.mark.parametrize("factor", [10, 50])
def test_detrend_smoothness_priors_matches_biosppy(tools, t, factor):
    tt = np.arange(t) / t
    signal = np.sin(2 * np.pi * tt) + 0.25 * np.sin(2 * np.pi * 3 * tt) + 0.01
    got_detrended, got_trend = mbp.detrend_smoothness_priors(signal, factor)
    want_detrended, want_trend = tools.detrend_smoothness_priors(signal, factor)
    np.testing.assert_allclose(got_detrended, want_detrended, rtol=1e-8,
                               atol=1e-10)
    np.testing.assert_allclose(got_trend, want_trend, rtol=1e-8, atol=1e-10)


def test_detrend_output_sums_back_to_the_signal():
    signal = np.sin(np.linspace(0, 30, 256)) + 0.1
    detrended, trend = mbp.detrend_smoothness_priors(signal, 20.0)
    np.testing.assert_allclose(detrended + trend, signal, rtol=1e-12, atol=1e-14)


def test_detrend_removes_a_linear_trend():
    tt = np.arange(512) / 512.0
    signal = 3.0 * tt + 0.1 * np.sin(2 * np.pi * 5 * tt)
    detrended, trend = mbp.detrend_smoothness_priors(signal, 100.0)
    # The detrended part should be small compared with the original ramp.
    assert np.abs(detrended).max() < 0.05 * np.abs(signal).max()
    assert np.polyfit(tt, trend, 1)[0] > 2.5
