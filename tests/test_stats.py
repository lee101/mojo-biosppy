"""Parity tests for the reduction kernels: signal_stats, normalize, the
correlation and error measures, band power, and mean-wave extraction."""

import numpy as np
import pytest

import mojo_biosppy as mbp
from mojo_biosppy import _lib

RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(314159)
    t = np.arange(8192) / 512.0
    return np.sin(2 * np.pi * 5.0 * t) + 0.3 * rng.standard_normal(t.size)


def test_signal_stats_matches_biosppy(tools, signal):
    got = mbp.signal_stats(signal)
    want = tools.signal_stats(signal)
    names = ("mean", "median", "min", "max", "max_amp", "range", "var",
             "std_dev", "abs_dev", "rms")
    for name in names:
        np.testing.assert_allclose(
            got[name], want[name], rtol=RTOL, atol=ATOL, err_msg=name
        )
    # kurtosis and skewness come from SciPy in both implementations.
    np.testing.assert_allclose(got["kurtosis"], want["kurtosis"], rtol=1e-12)
    np.testing.assert_allclose(got["skew"], want["skew"], rtol=1e-12)


def test_signal_stats_var_uses_ddof_one(tools, signal):
    got = mbp.signal_stats(signal)
    n = signal.size
    np.testing.assert_allclose(
        got["var"], np.var(signal, ddof=1), rtol=RTOL, atol=ATOL
    )
    # The divisor must be exactly n - 1; a ddof=0 kernel would be off by the
    # factor n / (n - 1), which is small but far above the tolerance.
    np.testing.assert_allclose(
        got["var"] * (n - 1) / n, np.var(signal, ddof=0), rtol=1e-12
    )


def test_signal_stats_extremes_are_exact(tools):
    x = np.array([3.0, -7.0, 12.5, 0.0, 4.25])
    got = mbp.signal_stats(x)
    assert got["min"] == -7.0
    assert got["max"] == 12.5
    assert got["range"] == 19.5


def test_normalize_matches_biosppy(tools, signal):
    got, = mbp.normalize(signal)
    want, = tools.normalize(signal)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)
    assert abs(float(got.mean())) < 1e-12
    np.testing.assert_allclose(float(got.std(ddof=1)), 1.0, rtol=1e-12)


def test_normalize_ddof_changes_the_divisor(tools, signal):
    got, = mbp.normalize(signal, ddof=0)
    want, = tools.normalize(signal, ddof=0)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        float(got.std(ddof=0)), 1.0, rtol=1e-12
    )


def test_pearson_matches_biosppy(tools):
    rng = np.random.default_rng(99)
    x = rng.standard_normal(4096)
    y = 0.7 * x + 0.3 * rng.standard_normal(4096)
    np.testing.assert_allclose(
        mbp.pearson_correlation(x, y)[0], tools.pearson_correlation(x, y)[0],
        rtol=1e-9, atol=1e-12,
    )


def test_pearson_of_identical_signals_is_exactly_one():
    x = np.linspace(-3.0, 5.0, 1000)
    assert mbp.pearson_correlation(x, x)[0] == 1.0


def test_pearson_of_opposite_signals_is_exactly_minus_one():
    x = np.linspace(-3.0, 5.0, 1000)
    assert mbp.pearson_correlation(x, -x)[0] == -1.0


def test_pearson_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mbp.pearson_correlation(np.zeros(4), np.zeros(5))


def test_rmse_matches_biosppy(tools):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2048)
    y = x + 0.25 * rng.standard_normal(2048)
    np.testing.assert_allclose(
        mbp.rms_error(x, y)[0], tools.rms_error(x, y)[0],
        rtol=RTOL, atol=ATOL,
    )
    assert mbp.rms_error(x, x)[0] == 0.0


def test_rmse_known_value():
    x = np.zeros(4)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert mbp.rms_error(x, y)[0] == pytest.approx(1.0, rel=1e-15)


def test_band_power_matches_biosppy(tools):
    nbins = 1024
    freqs = np.linspace(0.0, 250.0, nbins)
    spectrum = np.abs(np.fft.rfft(np.random.default_rng(2)
                                  .standard_normal(2 * (nbins - 1)))) + 1e-9
    power = 10.0 * np.log10(spectrum)
    for band in [(0.0, 50.0), (20.0, 60.0), (5.0, 100.0), (0.0, 250.0)]:
        np.testing.assert_allclose(
            mbp.band_power(freqs, power, band)[0],
            tools.band_power(freqs, power, band)[0],
            rtol=1e-12, atol=1e-12, err_msg=str(band),
        )
        np.testing.assert_allclose(
            mbp.band_power(freqs, power, band, decibel=False)[0],
            tools.band_power(freqs, power, band, decibel=False)[0],
            rtol=1e-12, atol=1e-12, err_msg=str(band),
        )


def test_band_power_reversed_and_clipped_bands(tools):
    freqs = np.linspace(0.0, 100.0, 501)
    power = np.linspace(10.0, -10.0, 501)
    # Upstream swaps a reversed pair and clips to the frequency range; both
    # the clipping and the inclusivity of the endpoints must match.
    np.testing.assert_allclose(
        mbp.band_power(freqs, power, (80.0, 10.0))[0],
        tools.band_power(freqs, power, (80.0, 10.0))[0],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        mbp.band_power(freqs, power, (-50.0, 500.0))[0],
        tools.band_power(freqs, power, (-50.0, 500.0))[0],
        rtol=1e-12, atol=1e-12,
    )


def test_band_power_empty_band_is_nan(tools):
    # Upstream clips the band to the frequency range before selecting, so the
    # public entry point can never produce an empty selection; the kernel
    # still has to agree with numpy's mean-of-nothing.
    freqs = np.linspace(0.0, 100.0, 501)
    power = np.zeros(501)
    got = _lib.band_power(freqs, power, 200.0, 300.0, True)
    want = float(np.mean(np.asarray([])))
    assert np.isnan(got) and np.isnan(want)
    with pytest.raises(ValueError):
        _lib.band_power(freqs, power[:10], 1.0, 2.0, True)


def test_mean_waves_matches_biosppy(tools):
    rng = np.random.default_rng(17)
    data = rng.standard_normal((37, 4))
    for step in (None, 5, 1):
        got, = mbp.mean_waves(data, size=9, step=step)
        want, = tools.mean_waves(data, size=9, step=step)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)


def test_mean_waves_one_dimensional(tools):
    data = np.arange(20.0)
    got, = mbp.mean_waves(data, size=5)
    want, = tools.mean_waves(data, size=5)
    # starts 0, 5, 10, 15 -- the 20th sample cannot start a window of 5
    assert got.shape == want.shape == (4,)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(got, [data[i:i + 5].mean()
                                     for i in (0, 5, 10, 15)],
                               rtol=1e-13, atol=1e-14)


def test_mean_waves_drops_the_trailing_window(tools):
    data = np.arange(10.0)
    got, = mbp.mean_waves(data, size=4, step=3)
    want, = tools.mean_waves(data, size=4, step=3)
    assert got.shape == want.shape == (3,)  # starts 0, 3, 6; 9 would be short
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)


def test_mean_waves_rejects_impossible_size(tools):
    with pytest.raises(ValueError):
        mbp.mean_waves(np.arange(5.0), size=9)
