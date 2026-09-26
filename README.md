# mojo-biosppy

`mojo-biosppy` is the compute-oriented subset of
[biosppy](https://github.com/circus-tent/biosppy) with the sample loops of
`biosppy.signals.tools` implemented in Mojo and callable from Python.  It keeps
the upstream function shapes, return ordering and numerical contracts; the
inner loops live in one compiled shared library.

The Python package is named `mojo_biosppy`, so it installs alongside the real
`biosppy` and the tests compare the two directly.

```python
import numpy as np
import mojo_biosppy as mbp

x = np.sin(np.linspace(0, 40, 100_000))
mbp.normalize(x)                       # z-scored signal
mbp.zero_cross(x)                      # crossing indices
mbp.find_extrema(x, mode="max")        # local maxima
mbp.detrend_smoothness_priors(x, 10.0) # banded Cholesky, not a dense inverse
```

## Covered subset

| area | biosppy function | what runs in Mojo |
| --- | --- | --- |
| Filtering | `lfilter` | direct-form II transposed recurrence |
| Filtering | `filtfilt` | both passes (padding and `lfilter_zi` stay in SciPy) |
| Smoothing | `smoother` | the direct convolution (window design stays in SciPy) |
| Differencing | `finite_difference` | the FIR recurrence and delay trim |
| Statistics | `signal_stats` | mean, variance, rms, min, max, max-amplitude, absolute deviation |
| Statistics | `normalize`, `pearson_correlation`, `rms_error` | the whole reduction |
| Events | `zero_cross`, `find_extrema` | the scan and the index/value output |
| Events | `get_heart_rate` | the rate computation and the 40-200 bpm filter |
| Matrix profile | `distance_profile` | moving sum/sigma and the complex distance combine |
| Matrix profile | `signal_self_join`, `signal_cross_join` | the per-query exclusion zone, incumbent update and argmin |
| Waves | `mean_waves` | the strided windowed mean |
| Spectral | `band_power` | the in-band reduction, linear and decibel |
| Detrending | `detrend_smoothness_priors` | band construction, banded Cholesky, banded solve |

Eighteen exported kernels back that list, all in `src/kernels.mojo`.

### What is not ported, and why

| not ported | reason |
| --- | --- |
| `get_filter`, `_filter_init`, `_filter_resp`, `OnlineFilter` | coefficient design, `lfilter_zi` and `freqz` are one-off small linear algebra, not sample loops; `get_filter` returns `b`/`a` and the shim feeds them straight to the ported `lfilter` |
| `analytic_signal`, `power_spectrum`, `welch_spectrum` | FFT work. `pocketfft` in NumPy is already a heavily tuned transform; a hand-written Mojo FFT would lose, not win. |
| `smoother(kernel="median")`, `median_waves` | per-window sorts. NumPy's introselect is already optimal and a sort has nothing to vectorise. Both are forwarded and tested for shape. |
| `windower` | its whole purpose is to call a user-supplied Python callable per window, so the loop cannot move into a C ABI without a callback mechanism upstream does not have. |
| `synchronize` | `np.correlate` on the full cross-correlation; an FFT correlate would be a different algorithm, not a faster version of this one. |
| `find_intersection`, `_pdiff` | root finding on interpolants, i.e. a `scipy.optimize` policy, not a sample loop. |
| `detrend_smoothness_priors`'s dense inverse | deliberately replaced: the matrix is symmetric with bandwidth 2, so the port factors the three bands instead of inverting a dense `t x t`. This is a different, better algorithm, and it is verified against the dense one. |
| ECG/EDA/EMG/resp/PPG/HRV signal classes, `synthesizers`, `storage`, `plotting`, `clustering`, `quality`, `inter_plotting`, `biometrics` | signal-specific pipelines built on SciPy transforms and file IO, not numeric cores of their own. Use real `biosppy` for them. |

The FFT inside `distance_profile` and the two joins stays in NumPy for the same
reason as `analytic_signal`: the STOMP dot products are cross-correlations.  The
O(n) statistics around them and the O(nb·n) profile update are the parts that
benefit from a compiled kernel, and those are ported.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-biosppy.so`.  Set `PYTHONPATH=python`
when using the package outside a Pixi task.  The shared toolchain in
`/nvme0n1-disk/mojo-toolchain` works without Pixi:

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
```

## Performance

Best-of-five wall clock, same process, against the real `biosppy` or against
the same NumPy expression upstream uses.  Every case verifies numerical
agreement before timing.  Numbers below are one run on a loaded shared box;
the shape of the result, not the third digit, is the claim.

| case | reference | mojo-biosppy | result |
| --- | ---: | ---: | ---: |
| `lfilter` n=1048576, 4th-order band-pass | 23.01 ms | 42.10 ms | 0.55x (slower) |
| `smoother` n=262144, k=31 | 15.94 ms | 18.68 ms | 0.85x (slower) |
| basic statistics n=2097152 | 57.67 ms | 7.82 ms | 7.37x faster |
| `zero_cross` n=4194304 | 103.07 ms | 54.81 ms | 1.88x faster |
| `signal_self_join` n=4096, m=64, all 4033 queries | 7207.46 ms | 6864.08 ms | 1.05x, parity |
| `detrend_smoothness_priors` t=2000 | 26745.49 ms | 0.30 ms | 89413x faster |
| `detrend_smoothness_priors` t=100000 | needs 75 GiB | 34.68 ms | no reference exists |

The two losses are real and worth naming.  `lfilter` is a strictly serial
dependence chain -- every output feeds the next state -- so there is nothing to
vectorise or thread, and SciPy's hand-unrolled C loop with no bounds checks
per tap beats a scalar Mojo loop that has to guard three array ends per tap.
`smoother` is the same story one level up: it is a bandwidth-bound direct
convolution and `np.convolve` is already at memory speed.

The detrending row is the interesting one.  It is not a constant-factor win: it
is a change of complexity.  Upstream builds a `t x t` dense matrix and calls
`np.linalg.inv` on it, which is O(t^3) time and O(t^2) memory; the port builds
the three bands of the same matrix, factorises it with a bandwidth-2 Cholesky
and solves, all O(t).  At t=100000 the upstream call needs 75 GiB for the
matrix alone and cannot be run at all, while the port finishes in 35 ms.

Reproduce with:

```bash
pixi run bench
```

## How it works

All eighteen kernels live in `src/kernels.mojo`, one compilation unit, because
shared-library build cost is largely fixed.  `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-biosppy.so`.

`python/mojo_biosppy` owns every array and normalises inputs to contiguous
`float64` (or `int64` for beat indices).  Buffers cross the C ABI as 64-bit
addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

Two design points are worth spelling out.

**`bp_band_chol_factor`.**  `I + lambda^2 D2^T D2` is symmetric with lower
bandwidth 2, so its lower triangle lives in three rows: the diagonal, and the
first and second sub-diagonals.  The Cholesky recurrence for a bandwidth-2
symmetric matrix is

```
L(j,j)   = sqrt(A(j,j) - L(j,j-1)^2 - L(j,j-2)^2)
L(j+1,j) = (A(j+1,j) - L(j+1,j-1) L(j,j-1)) / L(j,j)
L(j+2,j) = A(j+2,j) / L(j,j)
```

which touches five stored values per row.  The factor is stored in place in the
same three rows, and `bp_band_chol_solve` runs a banded forward and backward
substitution.  `bp_d2_bands` builds the bands from the definition
`D2[i, i+q] = u[q]` with `u = (1, -2, 1)`, matching upstream's
`spdiags(ones @ [1,-2,1], [0,1,2], t-2, t)` exactly, including the truncated
tail rows.

**`bp_profile_update`.**  Both joins fold one query at a time into a running
matrix profile.  The self join masks an exclusion zone of `size/4` samples
around the query, replaces on strict `<`, and writes the first `argmin` back at
the query index; the cross join has no zone and replaces on `<=`, so a later
query that ties the running minimum takes ownership of the row.  Both rules are
one kernel with a `mode` selector, and both are pinned by tests that would fail
on `<` vs `<=` or an off-by-one zone.

Mojo emits FMA, so float results match SciPy and biosppy only to a tolerance,
never bit-for-bit.  The IIR recurrence is the extreme case: over 4096 samples of
a 4th-order band-pass whose poles sit near the unit circle, the difference from
SciPy reaches 2e-12 absolute on an O(1) signal and does not grow with length.
Index outputs -- zero crossings, extrema, wave starts, matrix indices -- are
compared exactly.

## Tests

`tests/` is 92 parity tests against the real `biosppy.signals.tools` and
`scipy.signal`, all passing.  Every test names the bug it would catch: a
transposed coefficient order, a dropped delay, a plateau in `find_extrema`, a
window that slides one sample off, a band index that transposes the five
stencil taps, a `<` where upstream has `<=`.

One environment note, stated plainly.  `import biosppy` fails in the test venv
because `biosppy/signals/ecg.py` imports `peakutils`, which is not installed
there.  `tests/conftest.py` therefore registers the real `biosppy` package
directories as namespace packages so the import machinery loads the genuine
`utils.py` and `signals/tools.py` from site-packages.  No upstream code is
copied, patched or reimplemented; the fixture `biosppy_load_path` reports which
route was taken.  In an environment with a complete `biosppy` install the
conftest uses the plain import.

## License

MIT
