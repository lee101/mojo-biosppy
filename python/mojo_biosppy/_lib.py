"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-biosppy.so"

_i64 = ctypes.c_int64
_f64 = ctypes.c_double

# name -> (restype, argtypes)
_SIGNATURES = {
    "bp_lfilter": (None, [_i64] * 10),
    "bp_convolve_same": (None, [_i64] * 5),
    "bp_basic_stats": (None, [_i64, _i64, _i64]),
    "bp_abs_dev": (None, [_i64, _i64, _f64, _i64]),
    "bp_pearson": (None, [_i64, _i64, _i64, _i64]),
    "bp_rmse": (None, [_i64, _i64, _i64, _i64]),
    "bp_normalize": (None, [_i64, _i64, _i64, _i64]),
    "bp_zero_cross": (_i64, [_i64, _i64, _i64, _i64]),
    "bp_find_extrema": (_i64, [_i64, _i64, _i64, _i64, _i64, _i64]),
    "bp_heart_rate": (_i64, [_i64, _i64, _f64, _i64, _i64, _i64]),
    "bp_moving_stats": (_i64, [_i64, _i64, _i64, _i64, _i64]),
    "bp_dist_profile": (_i64, [_i64, _i64, _i64, _i64, _i64]),
    "bp_profile_update": (_i64, [_i64] * 8),
    "bp_mean_waves": (_i64, [_i64] * 6),
    "bp_band_power": (_i64, [_i64, _i64, _i64, _f64, _f64, _i64, _i64]),
    "bp_d2_bands": (_i64, [_i64, _i64]),
    "bp_band_chol_factor": (_i64, [_i64, _i64]),
    "bp_band_chol_solve": (_i64, [_i64, _i64, _i64, _i64]),
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


lib = _load()


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def lfilter(b, a, x, zi=None):
    """scipy.signal.lfilter, from the Mojo direct-form II transposed kernel."""
    b = f64(b).ravel()
    a = f64(a).ravel()
    x = f64(x).ravel()
    nst = max(b.size, a.size) - 1
    if nst < 0:
        nst = 0
    state = np.zeros(nst, dtype=np.float64)
    if zi is not None:
        zi = f64(zi).ravel()
        if zi.size != nst:
            raise ValueError(
                f"zi must hold max(len(b), len(a)) - 1 = {nst} states"
            )
        state[:] = zi
    y = np.empty(x.size, dtype=np.float64)
    zf = np.empty(nst, dtype=np.float64)
    lib.bp_lfilter(
        addr(b), addr(a), b.size, a.size, nst, x.size,
        addr(x), addr(state), addr(y), addr(zf),
    )
    return y, zf


def convolve_same(w, x):
    """numpy.convolve(w, x, mode="same") from the Mojo direct-convolution kernel."""
    w = f64(w).ravel()
    x = f64(x).ravel()
    y = np.empty(max(w.size, x.size), dtype=np.float64)
    lib.bp_convolve_same(addr(w), addr(x), w.size, x.size, addr(y))
    return y


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def basic_stats(x):
    x = f64(x).ravel()
    out = np.empty(6, dtype=np.float64)
    lib.bp_basic_stats(addr(x), x.size, addr(out))
    return out


def abs_dev(x, center):
    x = f64(x).ravel()
    out = np.empty(1, dtype=np.float64)
    lib.bp_abs_dev(addr(x), x.size, ctypes.c_double(float(center)), addr(out))
    return float(out[0])


def pearson(x, y):
    x = f64(x).ravel()
    y = f64(y).ravel()
    out = np.empty(1, dtype=np.float64)
    lib.bp_pearson(addr(x), addr(y), x.size, addr(out))
    return float(out[0])


def rmse(x, y):
    x = f64(x).ravel()
    y = f64(y).ravel()
    out = np.empty(1, dtype=np.float64)
    lib.bp_rmse(addr(x), addr(y), x.size, addr(out))
    return float(out[0])


def normalize(x, ddof=1):
    x = f64(x).ravel()
    out = np.empty_like(x)
    lib.bp_normalize(addr(x), x.size, int(ddof), addr(out))
    return out


# ---------------------------------------------------------------------------
# Index-producing scans
# ---------------------------------------------------------------------------


def zero_cross(x):
    x = f64(x).ravel()
    idx = np.empty(max(x.size, 1), dtype=np.int64)
    n = lib.bp_zero_cross(addr(x), x.size, addr(idx), idx.size)
    return idx[:n]


def find_extrema(x, mode="both"):
    modes = {"both": 0, "max": 1, "min": 2}
    if mode not in modes:
        raise ValueError(f"Unknwon mode {mode!r}.")
    x = f64(x).ravel()
    cap = max(x.size, 1)
    idx = np.empty(cap, dtype=np.int64)
    val = np.empty(cap, dtype=np.float64)
    n = lib.bp_find_extrema(
        addr(x), x.size, modes[mode], addr(idx), addr(val), cap
    )
    return idx[:n], val[:n]


def heart_rate(beats, sampling_rate=1000.0):
    beats = i64(beats).ravel()
    cap = max(beats.size, 1)
    ts = np.empty(cap, dtype=np.int64)
    hr = np.empty(cap, dtype=np.float64)
    n = lib.bp_heart_rate(
        addr(beats), beats.size, ctypes.c_double(float(sampling_rate)),
        addr(ts), addr(hr), cap,
    )
    return ts[:n], hr[:n]


# ---------------------------------------------------------------------------
# Windowed statistics
# ---------------------------------------------------------------------------


def moving_stats(x, m):
    x = f64(x).ravel()
    count = x.size - m + 1
    if m < 1 or count < 1:
        raise ValueError("window size must be between 1 and len(x)")
    sums = np.empty(count, dtype=np.float64)
    sigma = np.empty(count, dtype=np.float64)
    lib.bp_moving_stats(addr(x), x.size, m, addr(sums), addr(sigma))
    return sums, sigma


def dist_profile(m, z, sigma):
    z = np.ascontiguousarray(z, dtype=np.complex128)
    sigma = f64(sigma)
    dist = np.empty(z.size, dtype=np.complex128)
    lib.bp_dist_profile(m, addr(z), addr(sigma), addr(dist), z.size)
    return dist


def profile_update(dist, prof, pidx, idx, ezone=0, mode=0):
    dist = f64(dist)
    out = np.empty(2, dtype=np.float64)
    neighbor = lib.bp_profile_update(
        addr(dist), dist.size, addr(prof), addr(pidx), int(idx), int(ezone),
        int(mode), addr(out),
    )
    return int(neighbor), out


def mean_waves(data, size, step=None):
    data = f64(data)
    if data.ndim == 1:
        data = data[:, None]
    m, nch = data.shape
    if step is None:
        step = size
    if step < 0:
        raise ValueError("The step must be a positive integer.")
    length = m - size
    if 1 + length // step <= 0:
        raise ValueError("Not enough samples for the given `size`.")
    waves = np.empty((1 + length // step, nch), dtype=np.float64)
    count = lib.bp_mean_waves(addr(data), m, nch, int(size), int(step),
                              addr(waves))
    return waves[:count]


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


def band_power(freqs, power, f1, f2, decibel=True):
    freqs = f64(freqs).ravel()
    power = f64(power).ravel()
    if freqs.size != power.size:
        raise ValueError(
            "The input 'freqs' and 'power' arrays must have the same length."
        )
    out = np.empty(1, dtype=np.float64)
    lib.bp_band_power(
        addr(freqs), addr(power), freqs.size, ctypes.c_double(f1),
        ctypes.c_double(f2), 1 if decibel else 0, addr(out),
    )
    return float(out[0])


# ---------------------------------------------------------------------------
# Smoothness-priors detrending
# ---------------------------------------------------------------------------


def d2_bands(t):
    out = np.zeros(3 * t, dtype=np.float64)
    if lib.bp_d2_bands(t, addr(out)) != 0:
        raise ValueError("signal length must be positive")
    return out.reshape(3, t)


def band_chol_factor(d):
    d = np.ascontiguousarray(d, dtype=np.float64)
    info = lib.bp_band_chol_factor(d.shape[1], addr(d))
    if info != 0:
        raise np.linalg.LinAlgError(
            f"banded Cholesky failed: non-positive pivot at row {info}"
        )
    return d


def band_chol_solve(d, rhs):
    d = np.ascontiguousarray(d, dtype=np.float64)
    rhs = f64(rhs).ravel().copy()
    x = np.empty_like(rhs)
    if lib.bp_band_chol_solve(d.shape[1], addr(d), addr(rhs), addr(x)) != 0:
        raise ValueError("empty system")
    return x
