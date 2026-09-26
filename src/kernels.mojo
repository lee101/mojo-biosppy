"""Mojo kernels for the numeric core of biosppy.signals.tools.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The ported functions are the ones that own a loop over sample indices: the
IIR/FIR recurrence, the direct convolution used by `smoother`, the reductions
behind `signal_stats`/`pearson_correlation`/`rms_error`/`normalize`, the
index-producing `zero_cross`/`find_extrema`, the windowed statistics behind
`distance_profile` and the matrix-profile joins, the banded power average, and
the banded Cholesky that replaces biosppy's dense inverse in
`detrend_smoothness_priors`.
"""

from std.math import abs, exp, log10, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

# log(10) as a double, for the decibel conversion in band_power.
comptime LN10 = 2.302585092994045684


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def sgn(x: Float64) -> Float64:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def keep_extremum(aux: Float64, mode: Int) -> Bool:
    if mode == 0:
        return aux != 0.0
    if mode == 1:
        return aux < 0.0
    return aux > 0.0


def u_of(q: Int) -> Float64:
    if q == 0:
        return 1.0
    if q == 1:
        return -2.0
    return 1.0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@export("bp_lfilter")
def bp_lfilter(
    b_addr: Int, a_addr: Int, nb: Int, na: Int, nst: Int, n: Int,
    x_addr: Int, z_addr: Int, y_addr: Int, zf_addr: Int
) abi("C"):
    """Direct form II transposed IIR/FIR filtering, matching scipy.signal.lfilter.

    `b` and `a` are the numerator and denominator coefficients; `a[0]` is
    assumed to be 1.  `z` holds `nst = max(nb, na) - 1` state slots, the same
    count as SciPy uses, pre-loaded by the caller with the initial condition
    (zeros for a cold start).  When `zf_addr` is non-zero the final state is
    copied there.
    """
    var b = fp(b_addr)
    var a = fp(a_addr)
    var x = fp(x_addr)
    var z = fp(z_addr)
    var y = fp(y_addr)

    if nst == 0:
        for i in range(n):
            y[unsafe_offset=i] = b[unsafe_offset=0] * x[unsafe_offset=i]
    else:
        # `b[m]` and `a[m]` count as zero past the end of their arrays and
        # `z[m]` one past the end of the state, exactly as the reference
        # implementation treats them.  Hoisting those three bounds out of the
        # recurrence is tempting but there is no free lunch here: the taps
        # have to be read in increasing order because each `z[m]` is the
        # state left by the previous sample, so the loop stays serial and
        # SciPy's hand-unrolled C version stays ahead.
        var top = min(nb, nst)
        var bot = min(na, nst)
        for i in range(n):
            var xn = x[unsafe_offset=i]
            var yn = b[unsafe_offset=0] * xn + z[unsafe_offset=0]
            y[unsafe_offset=i] = yn
            for m in range(1, top + 1):
                var bm = b[unsafe_offset=m] if m < nb else 0.0
                var am = a[unsafe_offset=m] if m < na else 0.0
                var prev = z[unsafe_offset=m] if m < nst else 0.0
                z[unsafe_offset=m - 1] = bm * xn + prev - am * yn
            for m in range(top + 1, bot + 1):
                z[unsafe_offset=m - 1] = -a[unsafe_offset=m] * yn

    if zf_addr != 0:
        var zf = fp(zf_addr)
        for i in range(nst):
            zf[unsafe_offset=i] = z[unsafe_offset=i]


@export("bp_convolve_same")
def bp_convolve_same(
    w_addr: Int, x_addr: Int, nw: Int, nx: Int, y_addr: Int
) abi("C"):
    """`numpy.convolve(w, x, mode="same")`: a direct linear convolution.

    The full convolution has length `nw + nx - 1`; `same` keeps
    `max(nw, nx)` samples starting at `(min(nw, nx) - 1) // 2`.
    """
    var w = fp(w_addr)
    var x = fp(x_addr)
    var y = fp(y_addr)
    var length = max(nw, nx)
    var start = (min(nw, nx) - 1) // 2
    for i in range(length):
        var m = start + i
        var lo = m - (nx - 1)
        if lo < 0:
            lo = 0
        var hi = m
        if hi > nw - 1:
            hi = nw - 1
        var acc = Float64(0.0)
        for k in range(lo, hi + 1):
            acc = w[unsafe_offset=k] * x[unsafe_offset=m - k] + acc
        y[unsafe_offset=i] = acc


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@export("bp_basic_stats")
def bp_basic_stats(x_addr: Int, n: Int, out_addr: Int) abi("C"):
    """One-pass statistics: [mean, var(ddof=1), rms, min, max, max|x - mean|]."""
    var x = fp(x_addr)
    var o = fp(out_addr)
    var s = Float64(0.0)
    var lo = x[unsafe_offset=0]
    var hi = x[unsafe_offset=0]
    for i in range(n):
        var v = x[unsafe_offset=i]
        s = s + v
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    var mean = s / Float64(n)
    var acc = Float64(0.0)
    var sq = Float64(0.0)
    var amp = Float64(0.0)
    for i in range(n):
        var v = x[unsafe_offset=i]
        var d = v - mean
        acc = acc + d * d
        sq = sq + v * v
        var a = abs(d)
        if a > amp:
            amp = a
    o[unsafe_offset=0] = mean
    o[unsafe_offset=1] = acc / Float64(n - 1)
    o[unsafe_offset=2] = sqrt(sq / Float64(n))
    o[unsafe_offset=3] = lo
    o[unsafe_offset=4] = hi
    o[unsafe_offset=5] = amp


@export("bp_abs_dev")
def bp_abs_dev(x_addr: Int, n: Int, center: Float64, out_addr: Int) abi("C"):
    """Mean absolute deviation |x - center|, in one pass."""
    var x = fp(x_addr)
    var o = fp(out_addr)
    var acc = Float64(0.0)
    for i in range(n):
        acc = acc + abs(x[unsafe_offset=i] - center)
    o[unsafe_offset=0] = acc / Float64(n)


@export("bp_pearson")
def bp_pearson(x_addr: Int, y_addr: Int, n: Int, out_addr: Int) abi("C"):
    """Pearson r, in the same difference-of-sums form biosppy uses.

    Sxy = sum(x*y) - n*mx*my, and likewise for Sxx and Syy, so the result
    matches `biosppy.signals.tools.pearson_correlation` including its clamp to
    [-1, 1].
    """
    var x = fp(x_addr)
    var y = fp(y_addr)
    var o = fp(out_addr)
    var sx = Float64(0.0)
    var sy = Float64(0.0)
    var sxy = Float64(0.0)
    var sxx = Float64(0.0)
    var syy = Float64(0.0)
    for i in range(n):
        var a = x[unsafe_offset=i]
        var b = y[unsafe_offset=i]
        sx = sx + a
        sy = sy + b
        sxy = sxy + a * b
        sxx = sxx + a * a
        syy = syy + b * b
    var nf = Float64(n)
    var mx = sx / nf
    var my = sy / nf
    var cov = sxy - nf * mx * my
    var vx = sxx - nf * mx * mx
    var vy = syy - nf * my * my
    var r = cov / (sqrt(vx) * sqrt(vy))
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    o[unsafe_offset=0] = r


@export("bp_rmse")
def bp_rmse(x_addr: Int, y_addr: Int, n: Int, out_addr: Int) abi("C"):
    """Root mean square error, sqrt(mean((x - y)^2))."""
    var x = fp(x_addr)
    var y = fp(y_addr)
    var o = fp(out_addr)
    var acc = Float64(0.0)
    for i in range(n):
        var d = x[unsafe_offset=i] - y[unsafe_offset=i]
        acc = acc + d * d
    o[unsafe_offset=0] = sqrt(acc / Float64(n))


@export("bp_normalize")
def bp_normalize(x_addr: Int, n: Int, ddof: Int, out_addr: Int) abi("C"):
    """Zero mean and unit variance, with divisor `n - ddof`."""
    var x = fp(x_addr)
    var o = fp(out_addr)
    var s = Float64(0.0)
    for i in range(n):
        s = s + x[unsafe_offset=i]
    var mean = s / Float64(n)
    var acc = Float64(0.0)
    for i in range(n):
        var d = x[unsafe_offset=i] - mean
        acc = acc + d * d
    var sd = sqrt(acc / Float64(n - ddof))
    for i in range(n):
        o[unsafe_offset=i] = (x[unsafe_offset=i] - mean) / sd


# ---------------------------------------------------------------------------
# Index-producing scans
# ---------------------------------------------------------------------------


@export("bp_zero_cross")
def bp_zero_cross(
    x_addr: Int, n: Int, idx_addr: Int, cap: Int
) abi("C") -> Int:
    """Indices `i` where sign(x[i]) != sign(x[i+1]); returns the count found.

    This is exactly `np.nonzero(np.abs(np.diff(np.sign(signal))) > 0)[0]`,
    which is why a sample landing exactly on zero yields two crossings.
    """
    var x = fp(x_addr)
    var ix = ip(idx_addr)
    var count = 0
    var i = 0
    while i < n - 1:
        if sgn(x[unsafe_offset=i]) != sgn(x[unsafe_offset=i + 1]):
            if count < cap:
                ix[unsafe_offset=count] = Int64(i)
            count = count + 1
        i = i + 1
    return count


@export("bp_find_extrema")
def bp_find_extrema(
    x_addr: Int, n: Int, mode: Int, idx_addr: Int, val_addr: Int, cap: Int
) abi("C") -> Int:
    """Local extrema, matching `biosppy.signals.tools.find_extrema`.

    `mode` is 0 for 'both', 1 for 'max', 2 for 'min'.  The upstream test is on
    `diff(sign(diff(signal)))`, so a flat run produces an extremum index
    carrying the repeated value; this kernel reproduces that exactly.
    """
    var x = fp(x_addr)
    var ix = ip(idx_addr)
    var iv = fp(val_addr)
    var count = 0
    if n < 3:
        return 0
    for j in range(n - 2):
        var d1 = sgn(x[unsafe_offset=j + 1] - x[unsafe_offset=j])
        var d2 = sgn(x[unsafe_offset=j + 2] - x[unsafe_offset=j + 1])
        if keep_extremum(d2 - d1, mode):
            if count < cap:
                ix[unsafe_offset=count] = Int64(j + 1)
                iv[unsafe_offset=count] = x[unsafe_offset=j + 1]
            count = count + 1
    return count


@export("bp_heart_rate")
def bp_heart_rate(
    beats_addr: Int, nb: Int, sampling_rate: Float64, ts_addr: Int,
    hr_addr: Int, cap: Int
) abi("C") -> Int:
    """Instantaneous heart rate in bpm, keeping only physiologically valid beats.

    Mirrors `get_heart_rate`: `hr = fs * 60 / diff(beats)`, time stamps taken
    from `beats[1:]`, and only samples with `40 <= hr <= 200` retained.
    """
    var b = ip(beats_addr)
    var its = ip(ts_addr)
    var ihr = fp(hr_addr)
    var count = 0
    for i in range(nb - 1):
        var d = Float64(b[unsafe_offset=i + 1] - b[unsafe_offset=i])
        var hr = sampling_rate * (60.0 / d)
        if hr >= 40.0 and hr <= 200.0:
            if count < cap:
                its[unsafe_offset=count] = b[unsafe_offset=i + 1]
                ihr[unsafe_offset=count] = hr
            count = count + 1
    return count


# ---------------------------------------------------------------------------
# Windowed statistics: distance profile and matrix profile
# ---------------------------------------------------------------------------


@export("bp_moving_stats")
def bp_moving_stats(
    x_addr: Int, n: Int, m: Int, sum_addr: Int, sigma_addr: Int
) abi("C") -> Int:
    """Moving sum and standard deviation over windows of length `m`.

    Reproduces `biosppy.signals.tools._init_dist_profile`: the window sums come
    from a running prefix sum in the same accumulation order as `np.cumsum`, and
    `sigma = sqrt(sumsq/m - mean^2)`.  Returns `n - m + 1`.
    """
    var x = fp(x_addr)
    var osum = fp(sum_addr)
    var osig = fp(sigma_addr)
    if m < 1 or m > n:
        return 0
    var fm = Float64(m)
    var c1 = Float64(0.0)
    var c2 = Float64(0.0)
    for j in range(n - m + 1):
        if j == 0:
            for i in range(m):
                var v = x[unsafe_offset=i]
                c1 = c1 + v
                c2 = c2 + v * v
        else:
            var add = x[unsafe_offset=j + m - 1]
            var drop = x[unsafe_offset=j - 1]
            c1 = c1 + add - drop
            c2 = c2 + add * add - drop * drop
        var mean = c1 / fm
        var sig2 = c2 / fm - mean * mean
        if sig2 < 0.0:
            sig2 = 0.0
        osum[unsafe_offset=j] = c1
        osig[unsafe_offset=j] = sqrt(sig2)
    return n - m + 1


@export("bp_dist_profile")
def bp_dist_profile(
    m: Int, z_addr: Int, sigma_addr: Int, dist_addr: Int, count: Int
) abi("C") -> Int:
    """Combine STOMP dot products with moving statistics into squared distances.

    `z` is an interleaved complex buffer of `count` cross-correlation values
    from the FFT and `sigma` the moving standard deviations from
    `bp_moving_stats`.  Upstream keeps the result complex,
    `dist = 2*m*(1 - z/(m*sigma))`, and only takes `abs(sqrt(dist))` later, so
    `dist` is written interleaved too and the imaginary residual is carried
    through rather than discarded.
    """
    var z = fp(z_addr)
    var sig = fp(sigma_addr)
    var d = fp(dist_addr)
    var fm = Float64(m)
    var two_m = 2.0 * fm
    for j in range(count):
        var scale = 1.0 / (fm * sig[unsafe_offset=j])
        d[unsafe_offset=2 * j] = two_m * (1.0 - z[unsafe_offset=2 * j] * scale)
        d[unsafe_offset=2 * j + 1] = (
            -two_m * z[unsafe_offset=2 * j + 1] * scale
        )
    return count


@export("bp_profile_update")
def bp_profile_update(
    dist_addr: Int, nb: Int, prof_addr: Int, pidx_addr: Int, idx: Int,
    ezone: Int, mode: Int, out_addr: Int
) abi("C") -> Int:
    """Fold one query distance profile into the running matrix profile.

    `mode` 0 reproduces the per-query body of `signal_self_join`: the exclusion
    zone `[idx-ezone, idx+ezone]` is masked to infinity, every other candidate
    strictly below the incumbent replaces it, and `argmin` (first minimum) is
    written back at `idx`.  `mode` 1 reproduces `signal_cross_join`, which has
    no exclusion zone and replaces on `<=`.  `out` receives
    `[neighbor, dist[neighbor]]`; the neighbor index is returned.
    """
    var d = fp(dist_addr)
    var prof = fp(prof_addr)
    var pidx = ip(pidx_addr)
    var o = fp(out_addr)

    var lo = 0
    var hi = nb
    var best = Float64(1.0e308) * 2.0
    var besti = 0
    if mode == 0:
        lo = idx - ezone
        if lo < 0:
            lo = 0
        hi = idx + ezone + 1
        if hi > nb:
            hi = nb
        for i in range(nb):
            if i >= lo and i < hi:
                continue
            var v = d[unsafe_offset=i]
            if v < best:
                best = v
                besti = i
            if v < prof[unsafe_offset=i]:
                prof[unsafe_offset=i] = v
                pidx[unsafe_offset=i] = Int64(idx)
        if idx >= 0 and idx < nb:
            prof[unsafe_offset=idx] = d[unsafe_offset=besti]
            pidx[unsafe_offset=idx] = Int64(besti)
    else:
        for i in range(nb):
            var v = d[unsafe_offset=i]
            if v < best:
                best = v
                besti = i
            if v <= prof[unsafe_offset=i]:
                prof[unsafe_offset=i] = v
                pidx[unsafe_offset=i] = Int64(idx)

    o[unsafe_offset=0] = Float64(besti)
    o[unsafe_offset=1] = d[unsafe_offset=besti]
    return besti


@export("bp_mean_waves")
def bp_mean_waves(
    data_addr: Int, m: Int, nch: Int, size: Int, step: Int, out_addr: Int
) abi("C") -> Int:
    """Mean over each window of a row-major `m x nch` data set.

    Window starts run `0, step, 2*step, ...` up to `m - size`, matching
    `biosppy.signals.tools.mean_waves`.  Returns the number of waves written.
    """
    var data = fp(data_addr)
    var o = fp(out_addr)
    if m < size or nch < 1 or step < 1:
        return 0
    var count = 0
    var start = 0
    while start + size <= m:
        var inv = 1.0 / Float64(size)
        for c in range(nch):
            var acc = Float64(0.0)
            for t in range(size):
                acc = acc + data[unsafe_offset=(start + t) * nch + c]
            o[unsafe_offset=count * nch + c] = acc * inv
        count = count + 1
        start = start + step
    return count


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


@export("bp_band_power")
def bp_band_power(
    freqs_addr: Int, power_addr: Int, n: Int, f1: Float64, f2: Float64,
    decibel: Int, out_addr: Int
) abi("C") -> Int:
    """Average power in `[f1, f2]`, matching `tools.band_power`.

    In decibel mode the mean of `10**(p/10)` over the band is converted back to
    decibels.  Returns the number of bins in the band; when that is zero the
    output is NaN, matching `numpy.mean` of an empty selection.
    """
    var f = fp(freqs_addr)
    var p = fp(power_addr)
    var o = fp(out_addr)
    var acc = Float64(0.0)
    var count = 0
    for i in range(n):
        var fi = f[unsafe_offset=i]
        if fi >= f1 and fi <= f2:
            if decibel != 0:
                acc = acc + exp(p[unsafe_offset=i] * (LN10 / 10.0))
            else:
                acc = acc + p[unsafe_offset=i]
            count = count + 1
    if count == 0:
        o[unsafe_offset=0] = Float64(0.0) / Float64(0.0)
    elif decibel != 0:
        o[unsafe_offset=0] = 10.0 * log10(acc / Float64(count))
    else:
        o[unsafe_offset=0] = acc / Float64(count)
    return count


# ---------------------------------------------------------------------------
# Smoothness-priors detrending
# ---------------------------------------------------------------------------


@export("bp_d2_bands")
def bp_d2_bands(t: Int, out_addr: Int) abi("C") -> Int:
    """Bands 0, 1 and 2 of `D2^T D2` for the second-difference operator.

    `D2` is the `(t-2) x t` matrix with `D2[i, i+q] = u[q]` and
    `u = (1, -2, 1)`, exactly what `spdiags(ones @ [1,-2,1], [0,1,2], t-2, t)`
    builds upstream.  The three bands go to `out` as a 3 x t block; entries
    outside the matrix are zero.  Returns 0 on success.
    """
    var o = fp(out_addr)
    if t < 1:
        return 1
    for i in range(3 * t):
        o[unsafe_offset=i] = 0.0
    var imax = t - 3
    for d in range(3):
        for r in range(t):
            var c = r + d
            if c >= t:
                continue
            for q1 in range(3):
                var i1 = r - q1
                if i1 < 0 or i1 > imax:
                    continue
                for q2 in range(3):
                    if q2 - q1 != d:
                        continue
                    o[unsafe_offset=d * t + r] = (
                        o[unsafe_offset=d * t + r]
                        + u_of(q1) * u_of(q2)
                    )
    return 0


@export("bp_band_chol_factor")
def bp_band_chol_factor(n: Int, d_addr: Int) abi("C") -> Int:
    """Banded Cholesky of a symmetric matrix with lower bandwidth 2.

    `d` is a 3 x n block: row 0 the diagonal, row 1 the first sub-diagonal
    `A(j+1, j)`, row 2 the second `A(j+2, j)`.  On return the block holds the
    Cholesky factor `L` in the same layout (`L(j+q, j)` on row q, `L(j, j)` on
    row 0), so the matrix is `L L^T`.  Returns 0 on success, or the 1-based row
    of the first non-positive pivot, which the caller turns into an exception.
    """
    var d = fp(d_addr)
    for j in range(n):
        var t = d[unsafe_offset=j]
        if j >= 1:
            var v = d[unsafe_offset=n + j - 1]
            t = t - v * v
        if j >= 2:
            var v = d[unsafe_offset=2 * n + j - 2]
            t = t - v * v
        if t <= 0.0:
            return j + 1
        var l = sqrt(t)
        d[unsafe_offset=j] = l
        if j + 1 < n:
            var s = d[unsafe_offset=n + j]
            if j >= 1:
                s = s - d[unsafe_offset=2 * n + j - 1] * d[unsafe_offset=n + j - 1]
            d[unsafe_offset=n + j] = s / l
        if j + 2 < n:
            d[unsafe_offset=2 * n + j] = d[unsafe_offset=2 * n + j] / l
    return 0


@export("bp_band_chol_solve")
def bp_band_chol_solve(
    n: Int, d_addr: Int, rhs_addr: Int, x_addr: Int
) abi("C") -> Int:
    """Solve `L L^T x = rhs` for the factor from `bp_band_chol_factor`.

    `rhs` is overwritten with the intermediate `y = L^-1 rhs`; the result lands
    in `x`.  Returns 0 on success.
    """
    var d = fp(d_addr)
    var r = fp(rhs_addr)
    var x = fp(x_addr)
    if n < 1:
        return 1
    for j in range(n):
        var acc = r[unsafe_offset=j]
        if j >= 1:
            var lji = d[unsafe_offset=n + j - 1]
            acc = acc - lji * r[unsafe_offset=j - 1]
        if j >= 2:
            var lji = d[unsafe_offset=2 * n + j - 2]
            acc = acc - lji * r[unsafe_offset=j - 2]
        r[unsafe_offset=j] = acc / d[unsafe_offset=j]
    for j in range(n - 1, -1, -1):
        var acc = r[unsafe_offset=j]
        if j + 1 < n:
            acc = acc - d[unsafe_offset=n + j] * x[unsafe_offset=j + 1]
        if j + 2 < n:
            acc = acc - d[unsafe_offset=2 * n + j] * x[unsafe_offset=j + 2]
        x[unsafe_offset=j] = acc / d[unsafe_offset=j]
    return 0
