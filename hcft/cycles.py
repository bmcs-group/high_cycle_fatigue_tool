"""The cycle table: one cycling step in, one row per load cycle out.

This is hcft's method without its GUI (concept 01, ``dev/concepts/01_THE_CYCLE_TABLE_FOR_REPROXIS.md``):
the load's turning points, every channel read at them, and per cycle the secant stiffness and the loop's
opening. It imports numpy alone. It reads no files, plots nothing and knows no caller's schema.

**Units.** The core is unit-free. The columns are named for reproxis' call, which hands in the stress σ [MPa]
as the load and the strains ε [‰] as the channels, both compression positive. Then ``E_sec`` is in GPa,
``w_loop`` in MPa·‰ = kJ/m³, and the times are in s.

**A cycle** runs from an upper turning point of the load over the lower one to the next upper one: unloading,
then reloading. "Upper" is the load's maximum, so the load is handed in with the sign that makes the peak load
the maximum.

**One step at a time.** The caller hands in the samples of one cycling step. The band that separates a cycle from
noise is set per step, so a step's dominant block is read on its own scale. A head or a tail of another block is
kept in the table, marked as not ``regular``.

**What the core never does:** read files, convert force to stress, average over channels, or smooth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import ClassVar

import numpy as np

from .version import __version__

__all__ = [
    'BranchRate',
    'CycleTable',
    'branch_rate',
    'channel_quality',
    'cycle_table',
    'loss_angle',
    'sample_times',
    'turning_points',
]


# ---------------------------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CycleTable:
    """The cycles of one step: one row per cycle, ``m`` channels, ``k`` extra channels.

    The arrays are read-only. ``N = len(table)`` is the number of cycles.
    """

    # the channels' and the extra channels' names
    names: tuple[str, ...]
    extra_names: tuple[str, ...]
    # one row per cycle, (N,)
    cycle: np.ndarray      # 1 … N within the step
    t_start: np.ndarray    # the upper turning point that opens the cycle (refined)
    period: np.ndarray     # the time to the next upper turning point
    sigma_up: np.ndarray   # the load at the opening upper turning point
    sigma_lo: np.ndarray   # the load at the lower turning point
    regular: np.ndarray    # bool: the false-cycle criterion (see `cycle_table`)
    i_up: np.ndarray       # sample index (into the arrays handed in) of the opening upper turning point
    i_lo: np.ndarray       # … of the lower turning point
    i_end: np.ndarray      # … of the closing upper turning point
    # per channel and cycle, (m, N)
    eps_up: np.ndarray     # each channel at the opening upper turning point
    eps_lo: np.ndarray     # each channel at the lower turning point
    E_sec: np.ndarray      # (σ̄_up − σ_lo)/(ε̄_up − ε_lo), upper values averaged over the cycle's two uppers
    w_loop: np.ndarray     # ∮σ dε over the cycle's samples, closed by the chord
    eps_width: np.ndarray  # unloading branch minus reloading branch at the cycle's mean load
    # per extra channel and cycle, (k, N)
    extra_up: np.ndarray   # the extra channels at the opening upper turning point
    # per channel, (m,)
    follow: np.ndarray       # the channel's median amplitude over the step's largest
    still_share: np.ndarray  # the share of samples that move less than `still_rel` of a following channel
    h3: np.ndarray           # the third harmonic over the first, within the cycles
    jumps: tuple[np.ndarray, ...]  # per channel (n_j, 2): time and size of steps between consecutive cycles
    # per step
    t_first: float   # the step's first sample time
    band: float      # the hysteresis band of the turning points, in load units
    f: float         # 1 / the regular cycles' median period
    n_regular: int
    head: float      # the first regular cycle's start, after t_first
    n_dropped: int   # samples dropped because their time or load was not finite
    version: str     # hcft's version
    params: dict

    COLUMNS: ClassVar[tuple[str, ...]] = (
        'cycle', 't_start', 'period', 'sigma_up', 'sigma_lo', 'regular', 'i_up', 'i_lo', 'i_end')
    CHANNEL_COLUMNS: ClassVar[tuple[str, ...]] = ('eps_up', 'eps_lo', 'E_sec', 'w_loop', 'eps_width')
    EXTRA_COLUMNS: ClassVar[tuple[str, ...]] = ('extra_up',)
    CHANNEL_ATTRS: ClassVar[tuple[str, ...]] = ('follow', 'still_share', 'h3', 'jumps')
    STEP_ATTRS: ClassVar[tuple[str, ...]] = (
        't_first', 'band', 'f', 'n_regular', 'head', 'n_dropped', 'version', 'params')

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
        for v in self.jumps:
            v.flags.writeable = False

    def __len__(self):
        return int(self.cycle.size)

    def to_dict(self) -> dict:
        """All columns and attributes by name; ``params`` copied and ``jumps`` as a list."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d['params'] = dict(self.params)
        d['jumps'] = list(self.jumps)
        return d


# ---------------------------------------------------------------------------------------------------------------
# The building blocks
# ---------------------------------------------------------------------------------------------------------------
def sample_times(time) -> np.ndarray:
    """The samples' times with repeated stamps spread out.

    A logger export may round the time stamp (CYCRE-W: 0.05 s up to 1 000 s, 0.1 s up to 10 000 s, 1 s beyond), so
    that several samples share one stamp. The samples of one stamp are spread evenly from their stamp towards the
    next. A stamp before a gap, and the last stamp, keep the typical spacing.
    """
    t = np.asarray(time, dtype=float).ravel()
    if t.size < 2:
        return t.copy()
    dt = np.diff(t)
    if np.any(dt < 0):
        raise ValueError('time must not decrease')
    starts = np.concatenate(([0], np.flatnonzero(dt != 0) + 1))
    if starts.size in (1, t.size):
        return t.copy()
    counts = np.diff(np.append(starts, t.size))
    stamps = t[starts]
    step = np.empty(starts.size)
    step[:-1] = np.diff(stamps) / counts[:-1]
    dt0 = np.median(step[:-1])
    step[-1] = dt0
    step[step > 1.5 * dt0] = dt0
    k = np.arange(t.size) - np.repeat(starts, counts)
    return np.repeat(stamps, counts) + k * np.repeat(step, counts)


def turning_points(load, band: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices and kinds (+1 upper, −1 lower) of the load's turning points.

    A reversal counts once the load has come back from its running extreme by more than ``band``: a hysteresis
    band. This is hcft's force-range criterion applied to the running extreme instead of to maxima and minima
    paired by position, so a noise extremum never shifts the pairing. The turning points alternate. A peak of
    equal samples is taken at its first sample. Samples without a load (NaN) are skipped. Python runs over the
    candidate extrema only, not the samples.
    """
    if not band > 0:
        raise ValueError(f'band must be positive, got {band!r}')
    x = np.asarray(load, dtype=float).ravel()
    finite = np.isfinite(x)
    if not finite.all():                      # samples without a load are skipped; the indices stay the caller's
        keep = np.flatnonzero(finite)
        i, k = turning_points(x[keep], band)
        return keep[i], k
    d = np.diff(x)
    nz = np.flatnonzero(d != 0)
    if nz.size < 2:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    rising = d[nz] > 0
    chg = np.flatnonzero(rising[1:] != rising[:-1])
    cand = np.concatenate(([0], nz[chg] + 1, [x.size - 1]))
    idx, kind = [], []
    direction = 0
    hi = lo = float(x[0])
    hi_i = lo_i = 0
    ext, ext_i = hi, 0
    for i, v in zip(cand.tolist(), x[cand].tolist(), strict=True):
        if direction == 0:
            if v > hi:
                hi, hi_i = v, i
            if v < lo:
                lo, lo_i = v, i
            if hi - lo > band:
                if hi_i < lo_i:
                    idx.append(hi_i)
                    kind.append(1)
                    direction, ext, ext_i = -1, lo, lo_i
                else:
                    idx.append(lo_i)
                    kind.append(-1)
                    direction, ext, ext_i = 1, hi, hi_i
        elif direction == 1:
            if v > ext:
                ext, ext_i = v, i
            elif ext - v > band:
                idx.append(ext_i)
                kind.append(1)
                direction, ext, ext_i = -1, v, i
        else:
            if v < ext:
                ext, ext_i = v, i
            elif v - ext > band:
                idx.append(ext_i)
                kind.append(-1)
                direction, ext, ext_i = 1, v, i
    return np.array(idx, dtype=np.intp), np.array(kind, dtype=np.intp)


def _parabola(x, i):
    """Offset (in samples, −½ … ½) and value of the vertex of the parabola through i−1, i, i+1.

    At the record's first and last sample the sample itself is returned.
    """
    n = x.shape[-1]
    ic = np.clip(i, 1, n - 2)
    a, b, c = x[ic - 1], x[ic], x[ic + 1]
    den = a - 2 * b + c
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    val = b - 0.25 * (a - c) * off
    edge = ic != i
    off[edge] = 0.0
    val[edge] = x[i[edge]]
    return off, val


def _at(y, i, off):
    """y at the fractional sample i + off, on the parabola through i−1, i, i+1 (last axis)."""
    n = y.shape[-1]
    ic = np.clip(i, 1, n - 2)
    a, b, c = y[..., ic - 1], y[..., ic], y[..., ic + 1]
    val = b + 0.5 * (c - a) * off + 0.5 * (a - 2 * b + c) * off ** 2
    edge = ic != i
    if np.any(edge):
        val[..., edge] = y[..., i[edge]]
    return val


def _nanmedian(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _rows(a, n, what):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n:
        raise ValueError(f'{what} must have shape (n,) or (rows, n) with n = {n}, got {arr.shape}')
    return arr


def _names(names, m, prefix):
    if names is None:
        return tuple(f'{prefix}{i}' for i in range(m))
    names = tuple(str(s) for s in names)
    if len(names) != m:
        raise ValueError(f'{len(names)} names for {m} rows')
    return names


def _touched(bad, U0, U1):
    """(m, N): a bad sample lies in a cycle's samples or their neighbours, U0−1 … U1+1."""
    n = bad.shape[1]
    NB = np.concatenate((np.zeros((bad.shape[0], 1), dtype=np.intp), np.cumsum(bad, axis=1)), axis=1)
    a = np.maximum(U0 - 1, 0)
    b = np.minimum(U1 + 1, n - 1)
    return (NB[:, b + 1] - NB[:, a]) > 0


def _loop_area(x, y, U0, U1):
    """∮x dy per cycle: the polygon through the cycle's samples, closed by the chord from its last to its first.

    The polygon inscribed in a loop of N samples reads N sin(2π/N)/2π of its area (0.990 at 25.5 samples).
    A missing sample voids the cycles around it only (the caller voids them), not the later ones.
    """
    x0 = x - np.median(x)
    bad = ~np.isfinite(y)
    yz = np.where(bad, 0.0, y - np.nan_to_num(_nanmedian(y, axis=1))[:, None])
    seg = 0.5 * (x0[1:] + x0[:-1]) * np.diff(yz, axis=1)
    seg[bad[:, 1:] | bad[:, :-1]] = 0.0
    C = np.concatenate((np.zeros((y.shape[0], 1)), np.cumsum(seg, axis=1)), axis=1)
    return C[:, U1] - C[:, U0] - 0.5 * (x0[U0] + x0[U1]) * (yz[:, U1] - yz[:, U0])


def _loop_width(x, y, U0, L, U1, x_mean):
    """y on the unloading branch minus y on the reloading branch where the load passes its cycle's mean."""
    nc = U0.size
    bounds = np.empty(2 * nc, dtype=np.intp)
    bounds[0::2] = U0
    bounds[1::2] = L
    lens = np.diff(np.append(bounds, U1[-1]))
    msamp = np.repeat(np.repeat(x_mean, 2), lens)
    above = (x[U0[0]:U1[-1]] - msamp) > 0
    cnt = np.add.reduceat(above.astype(np.intp), bounds - U0[0])
    j_un = np.clip(U0 + cnt[0::2], 1, x.size - 1)               # first sample at or below the mean, unloading
    j_re = np.clip(L + (lens[1::2] - cnt[1::2]), 1, x.size - 1)  # first sample above the mean, reloading

    def cross(j):
        xa, xb = x[j - 1], x[j]
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.where(xb != xa, (x_mean - xa) / (xb - xa), 0.5)
        w = np.clip(w, 0.0, 1.0)
        return y[:, j - 1] + w * (y[:, j] - y[:, j - 1])

    return cross(j_un) - cross(j_re)


def _sample_quality(t, y, U0, U1, t_ua, period, ok, amp_ref, still_rel):
    """still_share and h3 per channel, over the samples of the channel's good cycles."""
    m = y.shape[0]
    still = np.full(m, np.nan)
    h3 = np.full(m, np.nan)
    lens = U1 - U0
    cyc = np.repeat(np.arange(U0.size), lens)
    a, b = U0[0], U1[-1]                               # the cycles' samples are contiguous: a … b−1
    reg_any = ok.any(axis=0)
    if not reg_any.any():
        return still, h3
    d_ref = 2.0 * amp_ref / np.median(lens[reg_any])   # a following channel's mean motion per sample
    thr = still_rel * d_ref
    phase = (t[a:b] - t_ua[cyc]) / period[cyc]
    e1 = np.exp(-2j * np.pi * phase)
    e3 = np.exp(-6j * np.pi * phase)
    nc = U0.size
    for c in range(m):
        sel = ok[c][cyc]
        if not sel.any():
            continue
        yc = y[c, a:b]
        if np.isfinite(thr) and thr > 0:
            still[c] = np.count_nonzero((np.abs(y[c, a + 1:b + 1] - yc) < thr) & sel) / np.count_nonzero(sel)
        cnt = np.bincount(cyc, weights=sel, minlength=nc)
        mean = np.bincount(cyc, weights=np.where(sel, yc, 0.0), minlength=nc) / np.maximum(cnt, 1)
        r = np.where(sel, yc - mean[cyc], 0.0)
        Y1 = np.abs(r @ e1)
        Y3 = np.abs(r @ e3)
        h3[c] = Y3 / Y1 if Y1 > 0 else np.nan
    return still, h3


def _jumps(tt, v, factor, floor, w=5, block=200):
    """Steps between consecutive cycles: (n_j, 2) of time and size.

    The detector compares the medians of the w cycles after and before each position, so a single outlier is no
    jump. A step counts beyond ``factor`` times the scatter of one value and beyond ``floor``. The scatter is
    taken from the mean absolute cycle-to-cycle difference in blocks of ``block`` cycles (σ = 0.886 mean|Δ| for
    white noise): unlike the median, it does not vanish on a channel that rests in quanta of its resolution.
    The jump's time is the start of the first cycle after it; its size is the level after minus the level before.
    """
    out = np.empty((0, 2))
    N = v.size
    if N < 2 * w + 2:
        return out
    d = np.diff(v)
    nb = max(1, d.size // block)
    edges = np.linspace(0, d.size, nb + 1).astype(np.intp)
    sig_b = [0.5 * np.sqrt(np.pi) * np.mean(np.abs(d[a:b])) for a, b in pairwise(edges)]
    sig_d = np.repeat(sig_b, np.diff(edges))
    med = np.median(np.lib.stride_tricks.sliding_window_view(v, w), axis=1)
    s = med[w:] - med[:-w]                    # s[j]: a step between cycles j+w−1 and j+w
    thr = np.fmax(factor * sig_d[np.arange(s.size) + w - 1], floor)
    ex = np.abs(s) > thr
    if not ex.any():
        return out
    e = np.concatenate(([False], ex, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(e) == 1)
    ends = np.flatnonzero(np.diff(e) == -1) - 1
    rows = []
    for ja, jb in zip(starts, ends, strict=True):
        jp = ja + int(np.argmax(np.abs(s[ja:jb + 1])))
        qa, qb = max(ja + w - 3, 0), min(jb + w + 1, d.size - 1)
        q = qa + int(np.argmax(np.sign(s[jp]) * d[qa:qb + 1]))
        rows.append((tt[q + 1], s[jp]))
    return np.array(rows, dtype=float)


def _empty(m, k, **attrs):
    z = np.empty(0)
    zi = np.empty(0, dtype=np.intp)
    zm = np.empty((m, 0))
    return CycleTable(
        cycle=zi.copy(), t_start=z.copy(), period=z.copy(), sigma_up=z.copy(), sigma_lo=z.copy(),
        regular=np.empty(0, dtype=bool), i_up=zi.copy(), i_lo=zi.copy(), i_end=zi.copy(),
        eps_up=zm.copy(), eps_lo=zm.copy(), E_sec=zm.copy(), w_loop=zm.copy(), eps_width=zm.copy(),
        extra_up=np.empty((k, 0)), follow=np.full(m, np.nan), still_share=np.full(m, np.nan),
        h3=np.full(m, np.nan), jumps=tuple(np.empty((0, 2)) for _ in range(m)),
        f=np.nan, n_regular=0, head=np.nan, version=__version__, **attrs)


# ---------------------------------------------------------------------------------------------------------------
# The function
# ---------------------------------------------------------------------------------------------------------------
def cycle_table(time, load, channels, *, names=None, extra=None, extra_names=None, band=None, band_rel=0.5,
                regular_tol=0.15, period_tol=0.25, refine=True, still_rel=0.15,
                jump_factor=8.0) -> CycleTable:
    """The per-cycle table of one cycling step.

    Parameters
    ----------
    time : (n,) monotonic sample times [s]. Repeated stamps are spread within their stamp (`sample_times`).
    load : (n,) the controlled quantity (reproxis: σ [MPa], compression positive).
    channels : (m, n) or (n,) the load's work-conjugate deformations (reproxis: ε [‰] per LVDT). NaN in a
        channel voids that channel's values in the cycles it touches, and nothing else.
    names : m channel names (default ``ch0`` …).
    extra : (k, n) or (n,), optional: channels read at the upper turning points only (T_surface, stroke).
    extra_names : k names (default ``extra0`` …).
    band : the hysteresis band of the turning points, in load units. Default: ``band_rel`` times the step's
        load range between its 0.5 and 99.5 percentiles.
    regular_tol, period_tol : a cycle is ``regular`` when its three turning loads lie within
        ``regular_tol`` × range of the step's median turning loads and its period within ``period_tol`` of the
        median period. This is hcft's false-cycle criterion, made robust. The regular cycles are the step's
        dominant block, so they must be the majority; otherwise none is regular.
    refine : turning instants refined by the parabola through three load samples; every channel is read there.
    still_rel : a sample is still when it moves less than ``still_rel`` times a following channel's mean motion
        per sample (CYCRE-W at 0.78 Hz: 0.15 → 0.1 µm).
    jump_factor : a step between consecutive cycles counts as a jump beyond this many times the scatter.

    Samples whose time or load is not finite are dropped first (``n_dropped``); the index columns refer to the
    arrays handed in.
    """
    t_in = np.asarray(time, dtype=float).ravel()
    x_in = np.asarray(load, dtype=float).ravel()
    n_in = t_in.size
    if x_in.size != n_in:
        raise ValueError(f'load has {x_in.size} samples, time {n_in}')
    y_in = _rows(channels, n_in, 'channels')
    e_in = np.empty((0, n_in)) if extra is None else _rows(extra, n_in, 'extra')
    m, k = y_in.shape[0], e_in.shape[0]
    names = _names(names, m, 'ch')
    extra_names = _names(extra_names, k, 'extra')

    keep = np.isfinite(t_in) & np.isfinite(x_in)
    kept = np.flatnonzero(keep)
    t = sample_times(t_in[keep])
    x = x_in[keep]
    y = y_in[:, keep]
    e = e_in[:, keep]
    n = t.size
    params = dict(band_rel=None if band is not None else band_rel, regular_tol=regular_tol,
                  period_tol=period_tol, refine=bool(refine), still_rel=still_rel, jump_factor=jump_factor)
    if band is None:
        band = band_rel * float(np.subtract(*np.percentile(x, [99.5, 0.5]))) if n >= 3 else 0.0
    band = float(band)
    attrs = dict(names=names, extra_names=extra_names, t_first=float(t[0]) if n else np.nan, band=band,
                 n_dropped=int(n_in - n), params=params)
    if n < 3 or not band > 0:
        return _empty(m, k, **attrs)

    idx, kind = turning_points(x, band)
    up, lo = idx[kind == 1], idx[kind == -1]
    if up.size:
        lo = lo[lo > up[0]]
    nc = min(up.size - 1, lo.size)
    if nc < 1:
        return _empty(m, k, **attrs)
    ua, L = up[:nc + 1], lo[:nc]
    U0, U1 = ua[:-1], ua[1:]

    if refine:
        o_u, x_ua = _parabola(x, ua)
        o_l, x_lo = _parabola(x, L)
    else:
        o_u, o_l = np.zeros(ua.size), np.zeros(nc)
        x_ua, x_lo = x[ua], x[L]
    t_ua = _at(t, ua, o_u)
    y_ua = _at(y, ua, o_u)
    y_lo = _at(y, L, o_l)
    e_up = _at(e, U0, o_u[:-1])

    x_up = x_ua[:-1]
    x_upm = 0.5 * (x_ua[:-1] + x_ua[1:])
    y_up = y_ua[:, :-1].copy()
    y_upm = 0.5 * (y_ua[:, :-1] + y_ua[:, 1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        E = (x_upm - x_lo) / (y_upm - y_lo)
    period = np.diff(t_ua)
    w_loop = _loop_area(x, y, U0, U1)
    width = _loop_width(x, y, U0, L, U1, 0.5 * (x_upm + x_lo))
    void = _touched(~np.isfinite(y), U0, U1)
    amp = np.abs(y_upm - y_lo)
    for a in (y_up, y_lo, E, w_loop, width, amp):
        a[void] = np.nan

    med_up, med_lo = np.median(x_up), np.median(x_lo)
    tol = regular_tol * (med_up - med_lo)
    med_p = np.median(period)
    regular = ((np.abs(x_up - med_up) < tol) & (np.abs(x_lo - med_lo) < tol)
               & (np.abs(x_ua[1:] - med_up) < tol) & (np.abs(period - med_p) < period_tol * med_p))
    if 2 * regular.sum() < nc:   # no dominant block (a hold read with a noise-sized band): the medians stand for none
        regular[:] = False

    ok = regular[None, :] & ~void
    med_amp = _nanmedian(np.where(ok, amp, np.nan), axis=1)
    amp_ref = np.nanmax(med_amp) if np.isfinite(med_amp).any() else np.nan
    follow = med_amp / amp_ref
    still, h3 = _sample_quality(t, y, U0, U1, t_ua, period, ok, amp_ref, still_rel)
    jumps = tuple(_jumps(t_ua[:-1][ok[c]], y_up[c][ok[c]], jump_factor, 1e-3 * med_amp[c]) for c in range(m))

    n_reg = int(regular.sum())
    return CycleTable(
        cycle=np.arange(1, nc + 1), t_start=t_ua[:-1], period=period, sigma_up=x_up, sigma_lo=x_lo,
        regular=regular, i_up=kept[U0], i_lo=kept[L], i_end=kept[U1],
        eps_up=y_up, eps_lo=y_lo, E_sec=E, w_loop=w_loop, eps_width=width, extra_up=e_up,
        follow=follow, still_share=still, h3=h3, jumps=jumps,
        f=float(1.0 / np.median(period[regular])) if n_reg else np.nan, n_regular=n_reg,
        head=float(t_ua[np.argmax(regular)] - t[0]) if n_reg else np.nan, version=__version__, **attrs)


# ---------------------------------------------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------------------------------------------
def loss_angle(table: CycleTable) -> np.ndarray:
    """(m, N) the loss angle [°] of each cycle: arctan(4 w / (π Δσ Δε)), Δ from the opening upper to the lower.

    Exact for an elliptic loop; the inscribed polygon reads it low by its N sin(2π/N)/2π.
    """
    ds = table.sigma_up - table.sigma_lo
    de = table.eps_up - table.eps_lo
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.degrees(np.arctan(4.0 * table.w_loop / (np.pi * ds * de)))


def _robust_rel(a):
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.nan
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)) / abs(med)) if med != 0 else np.nan


def channel_quality(table: CycleTable, block: int = 100) -> dict:
    """Per channel, over the regular cycles: the quality measures and the typical values.

    ``E_scatter`` is the robust relative scatter (MAD) of one cycle's secant; ``w_scatter`` that of the mean loop
    area over ``block`` consecutive cycles (CYCRE-W at 0.78 Hz: 100 cycles, about 2 min).
    """
    reg = table.regular
    phi = loss_angle(table)
    out = dict(name=np.array(table.names), follow=np.asarray(table.follow), still_share=np.asarray(table.still_share),
               h3=np.asarray(table.h3), n_jumps=np.array([j.shape[0] for j in table.jumps]))
    m = len(table.names)
    cols = {key: np.full(m, np.nan) for key in ('E_sec', 'E_scatter', 'w_loop', 'w_scatter', 'loss_angle')}
    for c in range(m):
        E = table.E_sec[c][reg]
        w = table.w_loop[c][reg]
        cols['E_sec'][c] = _nanmedian(E) if E.size else np.nan
        cols['E_scatter'][c] = _robust_rel(E)
        cols['w_loop'][c] = _nanmedian(w) if w.size else np.nan
        cols['loss_angle'][c] = _nanmedian(phi[c][reg]) if w.size else np.nan
        nb = w.size // block
        if nb >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                wm = np.nanmean(w[:nb * block].reshape(nb, block), axis=1)
            cols['w_scatter'][c] = _robust_rel(wm)
    out.update(cols)
    return out


@dataclass(frozen=True, eq=False)
class BranchRate:
    """The rates of the creep curve's two branches at given times into the step.

    Arrays are (a, m): one row per time in ``at``, one column per channel. Rates are in deformation units per
    time unit (reproxis: ‰/s; × 3600 for ‰/h).
    """

    at: np.ndarray
    half_window: float
    names: tuple[str, ...]
    up: np.ndarray
    lo: np.ndarray
    se_up: np.ndarray
    se_lo: np.ndarray
    n: np.ndarray
    covariate: str | None
    cov_up: np.ndarray        # the covariate's coefficient in the fit of the upper branch
    cov_lo: np.ndarray
    cov_range: np.ndarray     # (a,) the covariate's max − min in the window
    jump: np.ndarray          # (a, m) a jump of that channel lies in the window


def _fit(tt, yy, cc):
    """The slope of yy over tt (with cc as a further regressor if given), its standard error, cc's coefficient."""
    cols = [np.ones_like(tt), tt - tt.mean()]
    if cc is not None:
        cols.append(cc - cc.mean())
    A = np.column_stack(cols)
    p = A.shape[1]
    if tt.size < p + 2:
        return np.nan, np.nan, np.nan
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    res = yy - A @ coef
    s2 = res @ res / (tt.size - p)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    return coef[1], float(np.sqrt(cov[1, 1])), (coef[2] if cc is not None else np.nan)


def branch_rate(table: CycleTable, at, half_window: float, *, channels=None, covariate=None) -> BranchRate:
    """The rates of ``eps_up`` and ``eps_lo`` at the times ``at`` into the step, by a local linear fit.

    The window is ``at ± half_window`` (times after ``t_first``), over the regular cycles. ``channels``
    selects channels by name or index (default: all). ``covariate`` is the name of an extra channel (T_surface)
    or an (N,) array: it enters the fit as a further regressor, so that the rate is read at a constant covariate.
    The caller masks the covariate's spikes (NaN) before; cycles where it is NaN leave the fit.
    """
    at = np.atleast_1d(np.asarray(at, dtype=float))
    if channels is None:
        sel_c = list(range(len(table.names)))
    else:
        sel_c = [table.names.index(c) if isinstance(c, str) else int(c) for c in channels]
    names = tuple(table.names[c] for c in sel_c)
    cov_name = None
    if covariate is None:
        cv = None
    elif isinstance(covariate, str):
        cov_name = covariate
        cv = np.asarray(table.extra_up[table.extra_names.index(covariate)], dtype=float)
    else:
        cov_name = 'array'
        cv = np.asarray(covariate, dtype=float)
        if cv.shape != table.t_start.shape:
            raise ValueError('covariate must have one value per cycle')
    tt = table.t_start - table.t_first
    shape = (at.size, len(sel_c))
    out = {key: np.full(shape, np.nan) for key in ('up', 'lo', 'se_up', 'se_lo', 'cov_up', 'cov_lo')}
    nn = np.zeros(shape, dtype=np.intp)
    jump = np.zeros(shape, dtype=bool)
    cov_range = np.full(at.size, np.nan)
    for i, a in enumerate(at):
        win = table.regular & (np.abs(tt - a) <= half_window)
        if cv is not None:
            cw = cv[win & np.isfinite(cv)]
            cov_range[i] = cw.max() - cw.min() if cw.size else np.nan
            win = win & np.isfinite(cv)
        for j, c in enumerate(sel_c):
            jt = table.jumps[c][:, 0] - table.t_first if table.jumps[c].size else np.empty(0)
            jump[i, j] = bool(np.any(np.abs(jt - a) <= half_window))
            for key, Y in (('up', table.eps_up[c]), ('lo', table.eps_lo[c])):
                s = win & np.isfinite(Y)
                r, se, cc = _fit(tt[s], Y[s], cv[s] if cv is not None else None)
                out[key][i, j], out['se_' + key][i, j], out['cov_' + key][i, j] = r, se, cc
            nn[i, j] = int((win & np.isfinite(table.eps_up[c])).sum())
    return BranchRate(at=at, half_window=float(half_window), names=names, n=nn, covariate=cov_name,
                      cov_range=cov_range, jump=jump, **out)
