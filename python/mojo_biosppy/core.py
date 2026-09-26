"""biosppy-shaped signal-processing API backed by the Mojo kernels.

Each function mirrors the signature and the numerical contract of the matching
`biosppy.signals.tools` function.  Where upstream spends its time in a loop
over samples the work happens in `src/kernels.mojo`; where upstream calls into
SciPy for a transform (FFT, median filter, window design, dense linear algebra)
the port keeps that call, and says so in the README.
"""

from __future__ import annotations

import collections
import keyword

import numpy as np

from . import _lib


def _real_window(kernel, size, **kwargs):
    """Window design, forwarded to biosppy's own `_get_window`."""
    from biosppy.signals import tools

    return tools._get_window(kernel, size, **kwargs)


class ReturnTuple(tuple):
    """A tuple that also accepts its element names, as in `biosppy.utils`."""

    def __new__(cls, values, names=None):
        return tuple.__new__(cls, tuple(values))

    def __init__(self, values, names=None):
        names = list(map(str, names)) if names is not None else [
            f"_{i}" for i in range(len(values))
        ]
        if len(names) != len(values):
            raise ValueError("Number of names and values mismatch.")
        seen = set()
        for name in names:
            if not all(c.isalnum() or c == "_" for c in name):
                raise ValueError(
                    f"Names can only contain alphanumeric characters and "
                    f"underscores: {name!r}."
                )
            if keyword.iskeyword(name):
                raise ValueError(f"Names cannot be a keyword: {name!r}.")
            if name[0].isdigit():
                raise ValueError(f"Names cannot start with a number: {name!r}.")
            if name in seen:
                raise ValueError(f"Encountered duplicate name: {name!r}.")
            seen.add(name)
        self._names = names

    def as_dict(self):
        return collections.OrderedDict(zip(self._names, self))

    __dict__ = property(as_dict)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._names:
                raise KeyError(f"Unknown return value: {key!r}.")
            return tuple.__getitem__(self, self._names.index(key))
        return tuple.__getitem__(self, key)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def lfilter(b, a, x, zi=None):
    """Filter a signal with the given coefficients, forward only."""
    if zi is None:
        return _lib.lfilter(b, a, x)
    return _lib.lfilter(b, a, x, zi)


def filtfilt(b, a, x, padlen=None):
    """Zero-phase filtering: forward, reverse, forward, reverse.

    The odd edge extension and the default padding length follow
    `scipy.signal.filtfilt`; only the two lfilter passes run in Mojo.
    """
    import scipy.signal as ss

    b = _lib.f64(b).ravel()
    a = _lib.f64(a).ravel()
    x = _lib.f64(x).ravel()
    ntaps = max(b.size, a.size)
    if padlen is None:
        padlen = 3 * ntaps
    if padlen > x.size - 1:
        raise ValueError("The padding length must be less than the data length")
    edge = padlen
    left = 2.0 * x[0] - x[padlen:0:-1]
    right = 2.0 * x[-1] - x[-2:-padlen - 2:-1]
    ext = np.concatenate((left, x, right))
    # The steady-state step response is a small dense solve, so it stays in
    # SciPy; the two passes over the extended signal are the Mojo kernel.
    zi = ss.lfilter_zi(b, a)
    y, zf = _lib.lfilter(b, a, ext, zi=zi * ext[0])
    y, zf = _lib.lfilter(b, a, y[::-1], zi=zi * y[-1])
    y = y[::-1]
    return y[edge:len(ext) - edge]


def smoother(signal=None, kernel="boxzen", size=10, mirror=True, **kwargs):
    """Moving-average smoothing by convolution with a normalized window.

    The window itself is designed by `scipy.signal.windows` through biosppy's
    own `_get_window`, exactly as upstream; the convolution is the Mojo
    direct-convolution kernel.  `kernel="median"` is forwarded to SciPy.
    """
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    signal = _lib.f64(signal).ravel()
    length = signal.size

    if isinstance(kernel, str):
        if size > length:
            size = length - 1
        if size < 1:
            size = 1

        if kernel == "boxzen":
            aux, _ = smoother(signal, kernel="boxcar", size=size, mirror=mirror)
            smoothed, _ = smoother(aux, kernel="parzen", size=size,
                                   mirror=mirror)
            params = {"kernel": kernel, "size": size, "mirror": mirror}
            return ReturnTuple((smoothed, params), ("signal", "params"))

        if kernel == "median":
            import warnings

            import scipy.signal as ss

            if size % 2 == 0:
                size -= 1
                warnings.warn(
                    "When the kernel is 'median', size must be odd, so the size "
                    "was decremented by one."
                )
            smoothed = ss.medfilt(signal, kernel_size=size)
            params = {"kernel": kernel, "size": size, "mirror": mirror}
            return ReturnTuple((smoothed, params), ("signal", "params"))

        win = _real_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = _lib.f64(kernel).ravel()
        size = win.size
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")
        if size < 1:
            raise ValueError("Kernel size is smaller than 1.")
    else:
        raise TypeError("Unknown kernel type.")

    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal[0] * np.ones(size), signal, signal[-1] * np.ones(size))
        )
        smoothed = _lib.convolve_same(w, aux)[size:-size]
    else:
        smoothed = _lib.convolve_same(w, signal)

    params = {"kernel": kernel, "size": size, "mirror": mirror}
    params.update(kwargs)
    return ReturnTuple((smoothed, params), ("signal", "params"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def signal_stats(signal=None):
    """Mean, spread and shape statistics of a signal.

    Mirrors `biosppy.signals.tools.signal_stats`.  The quartiles, kurtosis and
    skewness are SciPy estimators that upstream calls directly, so they are
    kept unchanged here.
    """
    from scipy.stats import stats as sp_stats

    if signal is None:
        raise TypeError("Please specify an input signal.")
    signal = np.array(signal)
    flat = signal.ravel()
    mean, var, rms, lo, hi, max_amp = _lib.basic_stats(flat)
    median = float(np.median(signal))
    abs_dev = _lib.abs_dev(flat, median)
    values = (mean, median, lo, hi, max_amp, hi - lo, var, float(np.sqrt(var)),
              abs_dev, rms, float(sp_stats.kurtosis(signal, bias=False)),
              float(sp_stats.skew(signal, bias=False)))
    names = ("mean", "median", "min", "max", "max_amp", "range", "var",
             "std_dev", "abs_dev", "rms", "kurtosis", "skew")
    return ReturnTuple(values, names)


def normalize(signal=None, ddof=1):
    """Zero mean and unit standard deviation."""
    if signal is None:
        raise TypeError("Please specify an input signal.")
    return ReturnTuple((_lib.normalize(np.array(signal).ravel(), ddof),),
                       ("signal",))


def pearson_correlation(x=None, y=None):
    """Pearson correlation coefficient, in biosppy's difference-of-sums form."""
    if x is None:
        raise TypeError("Please specify the first input signal.")
    if y is None:
        raise TypeError("Please specify the second input signal.")
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError("Input signals must have the same length.")
    return ReturnTuple((_lib.pearson(x.ravel(), y.ravel()),), ("rxy",))


def rms_error(x=None, y=None):
    """Root mean square error between two signals."""
    if x is None:
        raise TypeError("Please specify the first input signal.")
    if y is None:
        raise TypeError("Please specify the second input signal.")
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError("Input signals must have the same length.")
    return ReturnTuple((_lib.rmse(x.ravel(), y.ravel()),), ("rmse",))


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def zero_cross(signal=None, detrend=False):
    """Indices where the signal crosses zero."""
    if signal is None:
        raise TypeError("Please specify an input signal.")
    signal = np.array(signal)
    if detrend:
        signal = signal - np.mean(signal)
    return ReturnTuple((_lib.zero_cross(signal.ravel()),), ("zeros",))


def find_extrema(signal=None, mode="both"):
    """Local extrema indices and the signal values at them."""
    if signal is None:
        raise TypeError("Please specify an input signal.")
    if mode not in ["max", "min", "both"]:
        raise ValueError(f"Unknwon mode {mode!r}.")
    extrema, values = _lib.find_extrema(np.array(signal).ravel(), mode)
    return ReturnTuple((extrema, values), ("extrema", "values"))


def get_heart_rate(beats=None, sampling_rate=1000.0, smooth=False, size=3):
    """Instantaneous heart rate in bpm from beat indices."""
    if beats is None:
        raise TypeError("Please specify the input beat indices.")
    beats = np.asarray(beats)
    if beats.size < 2:
        raise ValueError("Not enough beats to compute heart rate.")
    ts, hr = _lib.heart_rate(beats.ravel().astype(np.int64), sampling_rate)
    if smooth and hr.size > 1:
        hr, _ = smoother(signal=hr, kernel="boxcar", size=size, mirror=True)
    return ReturnTuple((ts, hr), ("index", "heart_rate"))


def finite_difference(signal=None, weights=None):
    """Central finite-difference derivative with the given weights."""
    if signal is None:
        raise TypeError("Please specify a signal to differentiate.")
    if weights is None:
        raise TypeError("Please specify the weight coefficients.")
    weights = _lib.f64(weights).ravel()
    nw = weights.size
    if nw % 2 == 0:
        raise ValueError("Number of weights must be odd.")
    signal = _lib.f64(signal).ravel()
    y, _ = _lib.lfilter(weights[::-1], np.ones(1), signal)
    d = nw - 1
    d2 = d // 2
    index = np.arange(d2, signal.size - d2, dtype="int")
    return ReturnTuple((index, y[d:]), ("index", "derivative"))


# ---------------------------------------------------------------------------
# Matrix profile
# ---------------------------------------------------------------------------


def _fft_signal(m, n, signal):
    """Upstream's `X` and the moving standard deviations, the latter from Mojo."""
    sigma = _lib.moving_stats(signal, m)[1]
    x = np.concatenate((signal, np.zeros(n, dtype="float")))
    return np.fft.fft(x), sigma


def _query_profile(m, n, query, X, sigma):
    """Upstream's `_ditance_profile`: normalize, STOMP dot, Mojo combine."""
    q = _lib.normalize(_lib.f64(query).ravel(), 0)
    y = np.concatenate((q[::-1], np.zeros(2 * n - m, dtype="float")))
    z = np.fft.ifft(X * np.fft.fft(y))[m - 1:n]
    return _lib.dist_profile(m, z, sigma)


def distance_profile(query=None, signal=None, metric="euclidean"):
    """Distance of a query to every subsequence of a signal."""
    if query is None:
        raise TypeError("Please specify an input query sequence.")
    if signal is None:
        raise TypeError("Please specify an input time series signal.")
    if metric not in ["euclidean", "pearson"]:
        raise ValueError("Unknown distance metric.")
    query = _lib.f64(np.array(query)).ravel()
    signal = _lib.f64(np.array(signal)).ravel()
    m, n = query.size, signal.size
    if m > n / 2:
        raise ValueError(
            "Time series signal is too short relative to query length."
        )
    X, sigma = _fft_signal(m, n, signal)
    dist = _query_profile(m, n, query, X, sigma)
    if metric == "pearson":
        dist = 1 - np.abs(dist) / (2 * m)
    else:
        dist = np.abs(np.sqrt(dist))
    return ReturnTuple((dist,), ("dist",))


def signal_self_join(signal=None, size=None, index=None, limit=None):
    """Matrix profile of a self-similarity join."""
    if signal is None:
        raise TypeError("Please specify an input time series signal.")
    if size is None:
        raise TypeError("Please specify the sub-sequence size.")
    signal = _lib.f64(np.array(signal)).ravel()
    n = signal.size
    if size > n / 2:
        raise ValueError(
            "Time series signal is too short relative to desired"
            " sub-sequence length."
        )
    if size < 4:
        raise ValueError("Sub-sequence length must be at least 4.")
    nb = n - size + 1
    if index is None:
        index = np.random.permutation(np.arange(nb, dtype="int"))
    else:
        index = np.asarray(index)
        if not np.all(index < nb):
            raise ValueError("Provided `index` exceeds allowable sub-sequences.")
    if limit is not None:
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")
        index = index[:limit]

    ezone = int(round(size / 4))
    profile = np.inf * np.ones(nb, dtype="float")
    pidx = np.zeros(nb, dtype=np.int64)
    X, sigma = _fft_signal(size, n, signal)
    for idx in index:
        dist = np.abs(np.sqrt(
            _query_profile(size, n, signal[idx:idx + size], X, sigma)))
        _lib.profile_update(dist, profile, pidx, int(idx), ezone, 0)
    return ReturnTuple((pidx, profile), ("matrix_index", "matrix_profile"))


def signal_cross_join(signal1=None, signal2=None, size=None, index=None,
                      limit=None):
    """Matrix profile of a similarity join between two series."""
    if signal1 is None:
        raise TypeError("Please specify the first input time series signal.")
    if signal2 is None:
        raise TypeError("Please specify the second input time series signal.")
    if size is None:
        raise TypeError("Please specify the sub-sequence size.")
    signal1 = _lib.f64(np.array(signal1)).ravel()
    signal2 = _lib.f64(np.array(signal2)).ravel()
    n1, n2 = signal1.size, signal2.size
    if size > n1 / 2:
        raise ValueError(
            "First time series signal is too short relative to"
            " desired sub-sequence length."
        )
    if size > n2 / 2:
        raise ValueError(
            "Second time series signal is too short relative to"
            " desired sub-sequence length."
        )
    if size < 4:
        raise ValueError("Sub-sequence length must be at least 4.")
    nb1 = n1 - size + 1
    nb2 = n2 - size + 1
    if index is None:
        index = np.random.permutation(np.arange(nb2, dtype="int"))
    else:
        index = np.asarray(index)
        if not np.all(index < nb2):
            raise ValueError("Provided `index` exceeds allowable `signal2`"
                             " sub-sequences.")
    if limit is not None:
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")
        index = index[:limit]

    profile = np.inf * np.ones(nb1, dtype="float")
    pidx = np.zeros(nb1, dtype=np.int64)
    X, sigma = _fft_signal(size, n1, signal1)
    for idx in index:
        dist = np.abs(np.sqrt(
            _query_profile(size, n1, signal2[idx:idx + size], X, sigma)))
        _lib.profile_update(dist, profile, pidx, int(idx), 0, 1)
    return ReturnTuple((pidx, profile), ("matrix_index", "matrix_profile"))


# ---------------------------------------------------------------------------
# Wave extraction and spectra
# ---------------------------------------------------------------------------


def mean_waves(data=None, size=None, step=None):
    """Mean of each overlapping window of a data set."""
    if data is None:
        raise TypeError("Please specify an input data set.")
    if size is None:
        raise TypeError("Please specify the number of samples for the mean.")
    if step is not None and step == 0:
        raise ValueError("The step must be a positive integer.")
    data = np.array(data)
    flat = data.ndim == 1
    waves = _lib.mean_waves(data, size, step)
    if flat:
        waves = waves.ravel()
    return ReturnTuple((waves,), ("waves",))


def median_waves(data=None, size=None, step=None):
    """Median of each overlapping window; forwarded to NumPy.

    A median needs a sort per window, which NumPy already does with a tuned
    introselect, so there is nothing here for a compiled kernel to win.
    """
    if data is None:
        raise TypeError("Please specify an input data set.")
    if size is None:
        raise TypeError("Please specify the number of samples for the median.")
    data = np.array(data)
    if step is None:
        step = size
    if step < 0:
        raise ValueError("The step must be a positive integer.")
    length = len(data) - size
    if 1 + length // step <= 0:
        raise ValueError("Not enough samples for the given `size`.")
    waves = np.array([np.median(data[i:i + size], axis=0)
                      for i in range(0, length + 1, step)])
    return ReturnTuple((waves,), ("waves",))


def band_power(freqs=None, power=None, frequency=None, decibel=True):
    """Average power in a frequency band."""
    if freqs is None:
        raise TypeError("Please specify the 'freqs' array.")
    if power is None:
        raise TypeError("Please specify the input power spectrum.")
    if len(freqs) != len(power):
        raise ValueError(
            "The input 'freqs' and 'power' arrays must have the same length."
        )
    if frequency is None:
        raise TypeError("Please specify the band frequencies.")
    try:
        f1, f2 = frequency
    except ValueError:
        raise ValueError("Input 'frequency' must be a pair of frequencies.")
    if f1 > f2:
        f1, f2 = f2, f1
    if f1 < freqs[0]:
        f1 = freqs[0]
    if f2 > freqs[-1]:
        f2 = freqs[-1]
    avg = _lib.band_power(np.asarray(freqs), np.asarray(power), f1, f2, decibel)
    return ReturnTuple((avg,), ("avg_power",))


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------


def detrend_smoothness_priors(signal, smoothing_factor=10):
    """Smoothness-priors detrending of Tarvainen et al.

    Upstream forms the dense `t x t` matrix `I + lambda^2 D2^T D2` and inverts
    it.  That matrix is symmetric with bandwidth 2, so this port builds its
    three bands, runs a banded Cholesky -- O(t) work instead of O(t^3) -- and
    solves against it.
    """
    signal = _lib.f64(np.asarray(signal)).ravel()
    t = signal.size
    if t < 1:
        raise ValueError("Please specify a signal to detrend.")
    bands = _lib.d2_bands(t)
    lam2 = float(smoothing_factor) ** 2
    m = np.empty((3, t), dtype=np.float64)
    m[0] = bands[0] * lam2 + 1.0
    m[1] = bands[1] * lam2
    m[2] = bands[2] * lam2
    _lib.band_chol_factor(m)
    z_trend = _lib.band_chol_solve(m, signal)
    z_detrended = signal - z_trend
    return ReturnTuple((z_detrended, z_trend), ("detrended", "trend"))
