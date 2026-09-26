"""Parity tests for the index-producing scans: zero crossings, local extrema
and the instantaneous heart rate."""

import numpy as np
import pytest

import mojo_biosppy as mbp


def test_zero_cross_matches_biosppy(tools):
    rng = np.random.default_rng(1234)
    t = np.arange(4096) / 64.0
    x = np.sin(2 * np.pi * t) + 0.35 * rng.standard_normal(t.size)
    np.testing.assert_array_equal(
        mbp.zero_cross(x)[0], tools.zero_cross(x)[0]
    )


def test_zero_cross_detrended(tools):
    x = np.array([2.0, -1.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0])
    np.testing.assert_array_equal(
        mbp.zero_cross(x, detrend=True)[0], tools.zero_cross(x, detrend=True)[0]
    )


def test_zero_cross_reports_the_first_index_of_a_crossing():
    # sign(signal) is [1, 0, 1], so upstream's diff has two non-zero entries
    # and the crossing is reported at the earlier index both times.
    x = np.array([1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 2.0])
    np.testing.assert_array_equal(mbp.zero_cross(x)[0],
                                  np.array([0, 1, 2, 4, 5]))


def test_zero_cross_exact_on_a_staircase():
    x = np.array([1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    np.testing.assert_array_equal(mbp.zero_cross(x)[0], np.array([2, 4]))


def test_zero_cross_is_empty_for_a_constant_signal():
    assert mbp.zero_cross(np.ones(10))[0].size == 0


@pytest.mark.parametrize("mode", ["both", "max", "min"])
def test_find_extrema_matches_biosppy(tools, mode):
    rng = np.random.default_rng(88)
    t = np.arange(3000) / 100.0
    x = np.sin(2 * np.pi * 2.0 * t) + 0.5 * rng.standard_normal(t.size)
    got_idx, got_val = mbp.find_extrema(x, mode=mode)
    want_idx, want_val = tools.find_extrema(x, mode=mode)
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_array_equal(got_val, want_val)


def test_find_extrema_on_a_plateau(tools):
    # A flat run is where a naive neighbour comparison and the upstream
    # diff(sign(diff(...))) rule disagree, so it is the discriminating case.
    x = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 2.0, 0.0])
    got_idx, got_val = mbp.find_extrema(x, mode="both")
    want_idx, want_val = tools.find_extrema(x, mode="both")
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_array_equal(got_val, want_val)
    assert got_idx.size > 0


def test_find_extrema_modes_partition_both(tools):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(2000)
    both, _ = mbp.find_extrema(x, mode="both")
    maxima, _ = mbp.find_extrema(x, mode="max")
    minima, _ = mbp.find_extrema(x, mode="min")
    np.testing.assert_array_equal(np.union1d(maxima, minima), both)


def test_find_extrema_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mbp.find_extrema(np.zeros(10), mode="nope")


def test_get_heart_rate_matches_biosppy(tools):
    fs = 100.0
    beats = np.cumsum(np.random.default_rng(21)
                      .integers(int(0.4 * fs), int(1.2 * fs), size=120))
    got_ts, got_hr = mbp.get_heart_rate(beats, sampling_rate=fs)
    want_ts, want_hr = tools.get_heart_rate(beats, sampling_rate=fs)
    np.testing.assert_array_equal(got_ts, want_ts)
    np.testing.assert_allclose(got_hr, want_hr, rtol=1e-14, atol=1e-15)


def test_get_heart_rate_drops_implausible_rates(tools):
    # The 17 ms gap gives 3529 bpm and the 2300 ms gap gives 2.6 bpm; both are
    # outside the 40-200 bpm window upstream applies, and the timestamp kept
    # is the *later* beat of the pair.
    fs = 1000.0
    beats = np.array([0, 833, 850, 1700, 4000, 4860], dtype=np.int64)
    ts, hr = mbp.get_heart_rate(beats, sampling_rate=fs)
    assert np.all((hr >= 40.0) & (hr <= 200.0))
    np.testing.assert_array_equal(ts, np.array([833, 1700, 4860]))
    np.testing.assert_allclose(hr, 1000.0 * 60.0 / np.array([833, 850, 860]),
                               rtol=1e-15)
    np.testing.assert_array_equal(ts, tools.get_heart_rate(beats,
                                                           sampling_rate=fs)[0])


def test_get_heart_rate_smoothed(tools):
    fs = 100.0
    beats = np.cumsum(np.random.default_rng(6)
                      .integers(int(0.45 * fs), int(0.9 * fs), size=80))
    _, got = mbp.get_heart_rate(beats, sampling_rate=fs, smooth=True, size=5)
    _, want = tools.get_heart_rate(beats, sampling_rate=fs, smooth=True, size=5)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_get_heart_rate_needs_two_beats():
    with pytest.raises(ValueError):
        mbp.get_heart_rate(np.array([10]))
