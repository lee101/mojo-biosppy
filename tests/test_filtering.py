"""Parity tests for the filtering kernels: lfilter, filtfilt, smoother, and the
finite-difference derivative built on the same recurrence."""

import numpy as np
import pytest
import scipy.signal as ss

import mojo_biosppy as mbp

# The recurrence is a serial dependence chain over a filter whose poles sit
# close to the unit circle, and Mojo contracts `b*x + z - a*y` into an FMA
# where SciPy rounds each product.  Measured over 4096 samples of a 4th-order
# band-pass that is a 2e-12 absolute difference on an O(1) signal, and it does
# not grow with length.  A structural error -- a wrong sign, a dropped delay, a
# transposed coefficient order -- is many orders of magnitude larger, so these
# tolerances still catch every bug they are meant to catch.
RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(20240501)
    t = np.arange(4096) / 100.0
    return (
        np.sin(2 * np.pi * 3.0 * t)
        + 0.4 * np.sin(2 * np.pi * 17.0 * t + 0.7)
        + 0.05 * rng.standard_normal(t.size)
    )


def test_lfilter_matches_scipy(signal):
    b, a = ss.butter(4, [0.05, 0.35], btype="band")
    got, zf = mbp.lfilter(b, a, signal)
    want = ss.lfilter(b, a, signal)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    assert zf.shape == (max(b.size, a.size) - 1,)


def test_lfilter_fir_is_plain_convolution(signal):
    b = np.array([0.25, 0.5, 0.25])
    a = np.array([1.0])
    got, _ = mbp.lfilter(b, a, signal)
    want = np.convolve(signal, b)[: signal.size]
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)


def test_lfilter_with_initial_state():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(512)
    b, a = ss.butter(3, 0.2)
    nstates = max(b.size, a.size) - 1
    zi = rng.standard_normal(nstates)
    got, zf = mbp.lfilter(b, a, x, zi=zi)
    want, want_zf = ss.lfilter(b, a, x, zi=zi)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(zf, want_zf, rtol=RTOL, atol=ATOL)


def test_lfilter_rejects_wrong_zi_length(signal):
    b, a = ss.butter(3, 0.2)
    with pytest.raises(ValueError):
        mbp.lfilter(b, a, signal, zi=np.zeros(max(b.size, a.size)))


def test_lfilter_state_count_matches_scipy(signal):
    b, a = ss.butter(4, 0.2)
    _, zf = mbp.lfilter(b, a, signal)
    assert zf.size == max(b.size, a.size) - 1


def test_filtfilt_matches_scipy(signal):
    b, a = ss.butter(4, [0.05, 0.35], btype="band")
    np.testing.assert_allclose(
        mbp.filtfilt(b, a, signal), ss.filtfilt(b, a, signal),
        rtol=RTOL, atol=ATOL,
    )


def test_filtfilt_is_zero_phase(signal):
    b, a = ss.butter(4, 0.1)
    y = mbp.filtfilt(b, a, signal)
    # A zero-phase filter preserves the location of a spectral peak; a
    # single forward pass would shift it by the group delay.
    peak = int(np.argmax(np.abs(np.fft.rfft(y))))
    raw_peak = int(np.argmax(np.abs(np.fft.rfft(signal))))
    assert abs(peak - raw_peak) <= 1


@pytest.mark.parametrize("kernel", ["hamming", "boxcar", "hann", "parzen",
                                    "bartlett", "blackman"])
def test_smoother_matches_biosppy(tools, signal, kernel):
    got, params = mbp.smoother(signal, kernel=kernel, size=9, mirror=True)
    want, want_params = tools.smoother(signal, kernel=kernel, size=9,
                                       mirror=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
    assert params["size"] == want_params["size"]


def test_smoother_boxzen_matches_biosppy(tools, signal):
    got, _ = mbp.smoother(signal, kernel="boxzen", size=11, mirror=True)
    want, _ = tools.smoother(signal, kernel="boxzen", size=11, mirror=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_smoother_without_mirror(tools, signal):
    got, _ = mbp.smoother(signal, kernel="hamming", size=8, mirror=False)
    want, _ = tools.smoother(signal, kernel="hamming", size=8, mirror=False)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_smoother_with_array_kernel(tools, signal):
    kernel = np.kaiser(7, 4.0)
    got, params = mbp.smoother(signal, kernel=kernel, mirror=True)
    want, _ = tools.smoother(signal, kernel=kernel, mirror=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
    assert params["size"] == 7


def test_smoother_length_is_preserved(tools, signal):
    got, _ = mbp.smoother(signal, kernel="boxcar", size=32, mirror=True)
    assert got.shape == signal.shape


def test_finite_difference_matches_biosppy(tools):
    t = np.arange(2048) / 256.0
    x = np.sin(2 * np.pi * t)
    weights = np.array([-1.0, 8.0, 0.0, -8.0, 1.0]) / (12 * (1.0 / 256.0))
    got_index, got = mbp.finite_difference(x, weights)
    want_index, want = tools.finite_difference(x, weights)
    np.testing.assert_array_equal(got_index, want_index)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    # Upstream convolves with the reversed weights, so the sign of the answer
    # follows the weight order, not the textbook formula.  A dropped or
    # unshifted delay would move the estimate by a whole sample and show up
    # here as an O(1) error rather than a sign flip.
    truth = 2 * np.pi * np.cos(2 * np.pi * t)[got_index]
    np.testing.assert_allclose(got, -truth, rtol=1e-4, atol=1e-3)
    _, flipped = mbp.finite_difference(x, -weights)
    np.testing.assert_allclose(flipped, truth, rtol=1e-4, atol=1e-3)


def test_finite_difference_requires_odd_weights():
    with pytest.raises(ValueError):
        mbp.finite_difference(np.zeros(16), np.ones(4))
