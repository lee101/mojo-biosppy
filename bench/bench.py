"""Correctness-gated benchmark for mojo-biosppy.

Every case verifies agreement with the real `biosppy` (or, where upstream just
calls NumPy, with the same NumPy expression) before timing, so a regression in
the Mojo kernels shows up as a correctness failure rather than a suspiciously
good number.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import time
import types

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

import mojo_biosppy as mbp  # noqa: E402
from mojo_biosppy import _lib  # noqa: E402


def _load_real_biosppy():
    """`biosppy/__init__.py` needs `peakutils`; load the real tools module."""
    try:
        return importlib.import_module("biosppy.signals.tools")
    except ImportError:
        spec = importlib.util.find_spec("biosppy")
        root = pathlib.Path(list(spec.submodule_search_locations)[0])
        for name, path in (("biosppy", root),
                           ("biosppy.signals", root / "signals")):
            if name in sys.modules:
                continue
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module
        importlib.import_module("biosppy.utils")
        return importlib.import_module("biosppy.signals.tools")


tools = _load_real_biosppy()


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_lfilter(n: int = 1 << 20):
    """Forward IIR filtering: the Mojo kernel against SciPy's lfilter."""
    import scipy.signal as ss

    rng = np.random.default_rng(0)
    t = np.arange(n) / n
    x = np.sin(2 * np.pi * 40 * t) + 0.2 * rng.standard_normal(n)
    b, a = ss.butter(4, [0.02, 0.20], btype="band")
    got, _ = mbp.lfilter(b, a, x)
    want = ss.lfilter(b, a, x)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9), "lfilter mismatch"
    return (f"lfilter n={n}", _time(lambda: ss.lfilter(b, a, x)),
            _time(lambda: mbp.lfilter(b, a, x)))


def bench_smoother(n: int = 1 << 18, size: int = 31):
    """Moving-average smoothing: the Mojo convolution against biosppy's."""

    rng = np.random.default_rng(1)
    t = np.arange(n) / n
    x = np.sin(2 * np.pi * 7 * t) + 0.3 * rng.standard_normal(n)
    got, _ = mbp.smoother(x, kernel="hamming", size=size, mirror=True)
    want, _ = tools.smoother(x, kernel="hamming", size=size, mirror=True)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-14), "smoother mismatch"
    return (f"smoother n={n} k={size}", _time(lambda: tools.smoother(
        x, kernel="hamming", size=size, mirror=True)),
        _time(lambda: mbp.smoother(x, kernel="hamming", size=size,
                                   mirror=True)))


def bench_basic_stats(n: int = 1 << 21):
    """One-pass statistics: the Mojo kernel against the NumPy expressions."""
    rng = np.random.default_rng(2)
    x = np.sin(np.arange(n) / 512.0) + 0.4 * rng.standard_normal(n)
    got = _lib.basic_stats(x)
    mean = x.mean()
    want = [mean, x.var(ddof=1), np.sqrt((x * x).mean()), x.min(), x.max(),
            np.abs(x - mean).max()]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)

    def numpy_path():
        m = x.mean()
        return (m, x.var(ddof=1), np.sqrt((x * x).mean()), x.min(), x.max(),
                np.abs(x - m).max())

    return (f"basic_stats n={n}", _time(numpy_path),
            _time(lambda: _lib.basic_stats(x)))


def bench_zero_cross(n: int = 1 << 22):
    """Zero-crossing scan against the exact NumPy expression upstream uses."""
    rng = np.random.default_rng(3)
    x = np.sin(np.arange(n) / 256.0) + 0.3 * rng.standard_normal(n)
    want = np.nonzero(np.abs(np.diff(np.sign(x))) > 0)[0]
    got = mbp.zero_cross(x)[0]
    np.testing.assert_array_equal(got, want)
    return (f"zero_cross n={n}",
            _time(lambda: np.nonzero(np.abs(np.diff(np.sign(x))) > 0)[0]),
            _time(lambda: mbp.zero_cross(x)))


def bench_self_join(n: int = 4096, size: int = 64):
    """Matrix-profile self join against biosppy, full search."""
    rng = np.random.default_rng(4)
    t = np.arange(n) / 32.0
    x = np.sin(2 * np.pi * 1.5 * t) + 0.4 * rng.standard_normal(n)
    index = np.arange(0, n - size + 1)
    got_i, got_p = mbp.signal_self_join(x, size=size, index=index)
    want_i, want_p = tools.signal_self_join(x, size=size, index=index)
    np.testing.assert_array_equal(got_i, want_i)
    np.testing.assert_allclose(got_p, want_p, rtol=1e-7, atol=1e-9)
    return (f"self_join n={n} m={size}",
            _time(lambda: tools.signal_self_join(x, size=size, index=index), 3),
            _time(lambda: mbp.signal_self_join(x, size=size, index=index), 3))


def bench_detrend(t: int = 2000):
    """Smoothness-priors detrending against biosppy's dense inverse."""
    rng = np.random.default_rng(5)
    tt = np.arange(t) / t
    x = np.sin(2 * np.pi * tt) + 0.25 * rng.standard_normal(t)
    got = mbp.detrend_smoothness_priors(x, 10.0)
    want = tools.detrend_smoothness_priors(x, 10.0)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-8, atol=1e-10)
    return (f"detrend t={t}",
            _time(lambda: tools.detrend_smoothness_priors(x, 10.0), 3),
            _time(lambda: mbp.detrend_smoothness_priors(x, 10.0), 3))


def bench_detrend_scale(t: int = 100000):
    """The same detrending at a size where the dense inverse does not fit.

    Upstream materialises a `t x t` float64 matrix: at t = 100000 that is
    80 GB, so there is no reference time to report.  Only the Mojo column is
    measured and the reference column is left blank.
    """
    rng = np.random.default_rng(6)
    tt = np.arange(t) / t
    x = np.sin(2 * np.pi * tt) + 0.25 * rng.standard_normal(t)
    detrended, trend = mbp.detrend_smoothness_priors(x, 10.0)
    assert detrended.shape == x.shape
    assert np.isfinite(trend).all()
    return (f"detrend t={t} (biosppy needs {t * t * 8 / 2**30:.0f} GiB)", None,
            _time(lambda: mbp.detrend_smoothness_priors(x, 10.0)))


def main():
    print(f"{'case':<38}{'reference':>12}{'mojo-biosppy':>16}{'ratio':>9}")
    print("-" * 75)
    for fn in (bench_lfilter, bench_smoother, bench_basic_stats,
               bench_zero_cross, bench_self_join, bench_detrend,
               bench_detrend_scale):
        label, ref, got = fn()
        if ref is None:
            print(f"{label:<38}{'n/a':>12}{got * 1e3:>13.2f}ms{'-':>9}")
            continue
        ratio = ref / got if got else float("nan")
        print(f"{label:<38}{ref * 1e3:>10.2f}ms{got * 1e3:>14.2f}ms"
              f"{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
