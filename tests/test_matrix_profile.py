"""Parity tests for the windowed statistics, the distance profile and the two
matrix-profile joins."""

import numpy as np
import pytest

import mojo_biosppy as mbp
from mojo_biosppy import _lib


def _upstream_sigma(m, n, signal):
    """The prefix-sum construction from `tools._init_dist_profile`."""
    csumx = np.zeros(n + 1, dtype="float")
    csumx[1:] = np.cumsum(signal)
    sumx = csumx[m:] - csumx[:-m]
    csumx2 = np.zeros(n + 1, dtype="float")
    csumx2[1:] = np.cumsum(np.power(signal, 2))
    sumx2 = csumx2[m:] - csumx2[:-m]
    meanx = sumx / m
    return np.sqrt((sumx2 / m) - np.power(meanx, 2)), sumx


@pytest.mark.parametrize("m", [4, 17, 64, 129])
def test_moving_stats_matches_the_upstream_prefix_sums(m):
    rng = np.random.default_rng(1000 + m)
    x = rng.standard_normal(1024)
    want_sigma, want_sum = _upstream_sigma(m, x.size, x)
    got_sum, got_sigma = _lib.moving_stats(x, m)
    assert got_sigma.size == want_sigma.size == x.size - m + 1
    np.testing.assert_allclose(got_sum, want_sum, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(got_sigma, want_sigma, rtol=1e-11, atol=1e-12)


def test_moving_stats_catches_a_dropped_first_window():
    x = np.arange(32.0)
    sums, _ = _lib.moving_stats(x, 4)
    np.testing.assert_allclose(sums[:3], [6.0, 10.0, 14.0], rtol=0, atol=0)
    np.testing.assert_allclose(sums[-1], x[-4:].sum(), rtol=0, atol=0)


def test_moving_stats_uses_a_running_prefix_sum_not_a_local_sum():
    # Both a running prefix sum and a fresh local sum give the same sums, so
    # the discriminating quantity is the sigma formula: a local sum of
    # squares gives a materially different value only if the window slides
    # wrongly, which the off-by-one check above and this one together pin.
    x = np.array([1.0, 100.0, 1.0, 1.0, 1.0])
    _, sigma = _lib.moving_stats(x, 3)
    assert sigma[0] == pytest.approx(np.std(x[:3]), rel=1e-12)
    assert sigma[1] == pytest.approx(np.std(x[1:4]), rel=1e-12)
    assert sigma[2] == pytest.approx(np.std(x[2:5]), rel=1e-12)


def test_distance_profile_matches_biosppy(tools):
    rng = np.random.default_rng(2024)
    signal = rng.standard_normal(2048)
    query = signal[100:116]
    # The euclidean profile is `abs(sqrt(dist))`, i.e. a fourth root, so where
    # the true distance is ~0 -- index 100 is a verbatim copy of the query --
    # the FFT round-off is amplified into the 1e-7 range.  A tolerance of
    # 1e-6 absolute there still separates this from any structural error.
    for metric in ("euclidean", "pearson"):
        got, = mbp.distance_profile(query, signal, metric=metric)
        want, = tools.distance_profile(query, signal, metric=metric)
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-6)


def test_distance_profile_finds_the_planted_match(tools):
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(4096)
    query = signal[512:528] + 1e-9 * rng.standard_normal(16)
    dist, = mbp.distance_profile(query, signal, metric="euclidean")
    assert int(np.argmin(dist)) == 512
    assert dist[512] < 1e-3


def test_distance_profile_rejects_a_long_query(tools):
    with pytest.raises(ValueError):
        mbp.distance_profile(np.zeros(100), np.zeros(101))


def test_signal_self_join_matches_biosppy(tools):
    rng = np.random.default_rng(31337)
    t = np.arange(4096) / 32.0
    signal = np.sin(2 * np.pi * 1.5 * t) + 0.4 * rng.standard_normal(t.size)
    size = 64
    index = np.arange(0, 4096 - size + 1, 3)
    got_idx, got_prof = mbp.signal_self_join(signal, size=size, index=index)
    want_idx, want_prof = tools.signal_self_join(signal, size=size,
                                                 index=index)
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_allclose(got_prof, want_prof, rtol=1e-7, atol=1e-9)


def test_signal_self_join_exclusion_zone_is_applied(tools):
    # A self-join must never report a subsequence as its own nearest
    # neighbour; if the exclusion zone were off by one, the argmin at the
    # query's own index would leak in.
    rng = np.random.default_rng(8)
    signal = rng.standard_normal(1024)
    size = 32
    index = np.arange(0, 1024 - size + 1, 5)
    got_idx, _ = mbp.signal_self_join(signal, size=size, index=index)
    ezone = int(round(size / 4))
    # Every queried index gets its own argmin written back, and that neighbour
    # must lie outside the exclusion zone.  An off-by-one in the zone would put
    # the query's own index here.
    assert np.all(np.abs(got_idx[index] - index) > ezone)
    assert not np.any(got_idx[index] == index)


def test_signal_self_join_single_query_has_an_exact_profile(tools):
    # With one query the profile is the query's own distance profile with the
    # exclusion zone at infinity, plus the argmin write-back at the query
    # index, so the expected values can be written out directly.
    rng = np.random.default_rng(12)
    signal = rng.standard_normal(600)
    size = 32
    idx = 100
    got_idx, got_prof = mbp.signal_self_join(signal, size=size,
                                             index=np.array([idx]))
    raw, = tools.distance_profile(signal[idx:idx + size], signal,
                                   metric="euclidean")
    ezone = int(round(size / 4))
    raw[max(0, idx - ezone):idx + ezone + 1] = np.inf
    neighbor = int(np.argmin(raw))
    raw[idx] = raw[neighbor]
    np.testing.assert_allclose(got_prof, raw, rtol=1e-12, atol=1e-14)
    assert got_idx[idx] == neighbor
    assert got_idx[neighbor] == idx


def test_signal_self_join_limit_is_honoured(tools):
    rng = np.random.default_rng(9)
    signal = rng.standard_normal(1024)
    size = 32
    index = np.arange(0, 1024 - size + 1, 7)
    got_idx, got_prof = mbp.signal_self_join(signal, size=size, index=index[:5])
    want_idx, want_prof = tools.signal_self_join(signal, size=size,
                                                 index=index[:5])
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_allclose(got_prof, want_prof, rtol=1e-7, atol=1e-9)
    assert np.all(got_prof[10:20] > 0.0)


def test_signal_cross_join_matches_biosppy(tools):
    rng = np.random.default_rng(4242)
    s1 = rng.standard_normal(2048)
    s2 = rng.standard_normal(2048)
    size = 48
    index = np.arange(0, 2048 - size + 1, 5)
    got_idx, got_prof = mbp.signal_cross_join(s1, s2, size=size, index=index)
    want_idx, want_prof = tools.signal_cross_join(s1, s2, size=size,
                                                  index=index)
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_allclose(got_prof, want_prof, rtol=1e-7, atol=1e-9)


def test_profile_update_uses_less_equal_for_the_cross_join():
    # The cross join replaces on `<=`, so a later query that ties the running
    # minimum takes ownership of the row.  A `<` kernel would leave the first
    # query's index there instead, which is what this asserts.
    dist = np.array([5.0, 1.0, 1.0, 9.0])
    prof = np.full(4, np.inf)
    pidx = np.zeros(4, dtype=np.int64)
    _lib.profile_update(dist, prof, pidx, 7, 0, 1)
    assert pidx.tolist() == [7, 7, 7, 7]
    np.testing.assert_array_equal(prof, dist)
    _lib.profile_update(dist, prof, pidx, 9, 0, 1)
    assert pidx.tolist() == [9, 9, 9, 9]
    _lib.profile_update(dist, prof, pidx, 11, 0, 1)


def test_profile_update_masks_the_self_join_exclusion_zone():
    dist = np.array([5.0, 1.0, 1.0, 9.0])
    prof = np.full(4, np.inf)
    pidx = np.zeros(4, dtype=np.int64)
    neighbor = _lib.profile_update(dist, prof, pidx, 2, 1, 0)[0]
    # zone is [1, 3], so the two 1.0 entries never become candidates and 5.0
    # wins the argmin.  Upstream then writes that argmin back at the query's
    # own index, which is itself inside the zone -- reproduced here, because
    # that write-back is exactly what distinguishes the kernel from one that
    # skips the zone entirely.
    assert neighbor == 0
    np.testing.assert_allclose(prof, np.array([5.0, np.inf, 5.0, np.inf]))
    assert pidx.tolist() == [2, 0, 0, 0]


def test_signal_self_join_rejects_a_short_subsequence(tools):
    with pytest.raises(ValueError):
        mbp.signal_self_join(np.zeros(1000), size=3)
