"""mojo-biosppy: the numeric core of biosppy.signals.tools in Mojo.

Installable alongside the real `biosppy` package, which it is tested against
for parity.  Forward-looking pieces that upstream delegates to SciPy -- the
FFT, the median filter, window design, kurtosis and skewness -- stay in SciPy.
"""

from .core import (
    ReturnTuple,
    band_power,
    detrend_smoothness_priors,
    distance_profile,
    filtfilt,
    find_extrema,
    finite_difference,
    get_heart_rate,
    lfilter,
    mean_waves,
    median_waves,
    normalize,
    pearson_correlation,
    rms_error,
    signal_cross_join,
    signal_self_join,
    signal_stats,
    smoother,
    zero_cross,
)

__all__ = [
    "ReturnTuple",
    "band_power",
    "detrend_smoothness_priors",
    "distance_profile",
    "filtfilt",
    "find_extrema",
    "finite_difference",
    "get_heart_rate",
    "lfilter",
    "mean_waves",
    "median_waves",
    "normalize",
    "pearson_correlation",
    "rms_error",
    "signal_cross_join",
    "signal_self_join",
    "signal_stats",
    "smoother",
    "zero_cross",
]
__version__ = "0.1.0"
