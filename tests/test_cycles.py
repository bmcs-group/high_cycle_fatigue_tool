"""The core on synthetic signals (concept 01, gates G1, G2, G5; G3 on a synthetic step).

Units as in reproxis' call: σ [MPa], ε [‰], t [s], sampled at 20 Hz as CYCRE-W.
"""

import os
import subprocess
import sys
import time as clock

import hcft1_reference as hcft1
import numpy as np
import pytest

from hcft import cycles
from hcft.cycles import branch_rate, channel_quality, cycle_table, loss_angle, sample_times, turning_points

FS = 20.0
SIG_M, DSIG = 15.0, 1.0   # mean and amplitude of the stress [MPa]
EPS_M, DEPS = 0.4, 0.028  # … of the strain [‰]


def inscribed(n):
    """The share of an ellipse's area that a polygon of n samples per loop encloses."""
    return n * np.sin(2 * np.pi / n) / (2 * np.pi)


def loops(spl, n_cycles, phi_deg=0.0, lag=0.0, t0=0.0):
    """σ and ε of an elliptic loop: ε lags σ by the angle phi and by the time lag."""
    f = FS / spl
    t = t0 + np.arange(round(n_cycles * spl)) / FS
    w = 2 * np.pi * f
    s = SIG_M + DSIG * np.cos(w * (t - t0))
    e = EPS_M + DEPS * np.cos(w * (t - t0 - lag) - np.radians(phi_deg))
    return t, s, e, f


def play(u, r):
    """A play (backlash) operator of half-width r: the channel stands still until the load has taken up the play."""
    out = np.empty_like(u)
    out[0] = u[0]
    for i in range(1, u.size):
        out[i] = min(max(out[i - 1], u[i] - r), u[i] + r)
    return out


# ---------------------------------------------------------------------------------------------------------------
# G1: the core imports numpy alone
# ---------------------------------------------------------------------------------------------------------------
def test_import_pulls_numpy_only():
    code = ('import sys, hcft.cycles; '
            'print(sorted({m.split(".")[0] for m in sys.modules} & '
            '{"traits", "traitsui", "pyface", "PyQt5", "pandas", "scipy", "matplotlib"}))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                         env=os.environ.copy())
    assert out.stdout.strip() == '[]'


# ---------------------------------------------------------------------------------------------------------------
# G2: the loop's area, phase and secant
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('spl', [25.5, 102.0, 179.0])
def test_elliptic_loop(spl):
    phi = 1.5
    t, s, e, f = loops(spl, 300, phi_deg=phi)
    tab = cycle_table(t, s, e)
    reg = tab.regular
    assert tab.n_regular >= len(tab) - 1 >= 297
    assert tab.f == pytest.approx(f, rel=1e-6)
    area = np.pi * DSIG * DEPS * np.sin(np.radians(phi))
    assert np.mean(tab.w_loop[0][reg]) / area == pytest.approx(inscribed(spl), abs=1e-4)
    E = DSIG / (DEPS * np.cos(np.radians(phi)))
    assert np.median(tab.E_sec[0][reg]) == pytest.approx(E, rel=4e-4)
    assert np.median(tab.sigma_up[reg]) == pytest.approx(SIG_M + DSIG, abs=1e-4)
    assert np.median(tab.sigma_lo[reg]) == pytest.approx(SIG_M - DSIG, abs=1e-4)
    width = 2 * DEPS * np.sin(np.radians(phi))
    assert np.median(tab.eps_width[0][reg]) == pytest.approx(width, rel=0.02)
    read = np.median(loss_angle(tab)[0][reg])
    assert read == pytest.approx(phi, abs=0.02)
    expected = np.degrees(np.arctan(np.tan(np.radians(phi)) * inscribed(spl)))
    assert read == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize('spl', [25.5, 102.0, 179.0])
@pytest.mark.parametrize('lag', [0.0025, 0.05])
def test_time_offset_reads_as_angle(spl, lag):
    """A pure time offset Δt opens the loop by 360° f Δt."""
    t, s, e, f = loops(spl, 200, lag=lag)
    tab = cycle_table(t, s, e)
    phi = 2 * np.pi * f * lag
    expected = np.degrees(np.arctan(np.tan(phi) * inscribed(spl)))
    assert np.median(loss_angle(tab)[0][tab.regular]) == pytest.approx(expected, abs=0.01)


def test_unrefined_turning_points():
    t, s, e, _ = loops(25.5, 100, phi_deg=1.5)
    tab = cycle_table(t, s, e, refine=False)
    assert tab.params['refine'] is False
    assert np.all(tab.sigma_up == s[tab.i_up])
    assert np.all(tab.t_start == t[tab.i_up])
    assert np.median(tab.E_sec[0][tab.regular]) == pytest.approx(DSIG / (DEPS * np.cos(np.radians(1.5))), rel=0.01)


# ---------------------------------------------------------------------------------------------------------------
# G2: noise, plateaus, a ramp, a hold and two blocks
# ---------------------------------------------------------------------------------------------------------------
def test_noise_extrema_and_plateaus_give_no_false_cycle():
    t, s, e, _ = loops(25.5, 300, phi_deg=1.5)
    clean = cycle_table(t, s, e)
    rng = np.random.default_rng(1)
    noisy = np.round((s + rng.normal(0, 0.025 * 2 * DSIG, s.size)) / 0.01) * 0.01   # noise and plateaus
    raw_max = hcft1.get_max_indices(noisy).size
    tab = cycle_table(t, noisy, e)
    assert raw_max > len(clean) + 100                  # the noise makes extrema that are no turning points
    assert len(tab) == len(clean)
    assert tab.n_regular == clean.n_regular
    assert np.max(np.abs(tab.t_start - clean.t_start)) < 3 / FS


def test_hcft1_equivalence_on_plateaus():
    """G3 on a synthetic step: hcft 1's extrema with its range cut are the core's turning points, except that a
    peak of equal samples is taken at its last sample by hcft 1 and at its first by the core, at equal load."""
    _, s, _, _ = loops(25.5, 300)
    q = np.round(s / 0.02) * 0.02                    # quantised: plateau peaks of two or more samples
    band = 0.5 * np.subtract(*np.percentile(q, [99.5, 0.5]))
    idx, kind = turning_points(q, band)
    mx, mn = hcft1.get_array_max_and_min_indices(q)
    mx, mn = hcft1.cut_indices_of_defined_range(q, mx, mn, band)
    n_doubles = 0
    for core, ref in ((idx[kind == 1], mx), (idx[kind == -1], mn)):
        core = core[core > 0]                          # hcft 1 never takes the first sample
        assert abs(core.size - ref.size) <= 1
        n = min(core.size, ref.size)
        core, ref = core[:n], ref[:n]
        assert np.all(ref >= core)
        assert np.all(q[ref] == q[core])
        doubles = ref != core
        n_doubles += doubles.sum()
        for a, b in zip(core[doubles], ref[doubles], strict=True):
            assert np.all(q[a:b + 1] == q[a])          # one plateau
    assert n_doubles > 100                             # the quantisation made plateau peaks


@pytest.mark.parametrize('head_range', [2.0, 5.0])
def test_ramp_hold_and_two_blocks(head_range):
    """A step that starts with a ramp, a hold and the tail of the previous block (below or above the band): the
    dominant block is regular, the rest is marked or not counted, and nothing is dropped silently."""
    ramp = np.linspace(12.0, 14.0, int(10 * FS))
    hold = np.full(int(60 * FS), 14.0)
    tA = np.arange(int(150 * FS)) / FS
    fA = 0.78
    a_lo = 18.0 - head_range                           # block A: the upper load is B's, the range smaller
    A = a_lo + head_range / 2 * (1 + np.cos(2 * np.pi * fA * tA))
    fB = FS / 102.0
    nB = 400
    tB = np.arange(int(nB / fB * FS)) / FS
    B = 14.0 + 4.0 * np.cos(2 * np.pi * fB * tB)
    s = np.concatenate((ramp, hold, A, B))
    t = np.arange(s.size) / FS
    e = s / 36.0
    tab = cycle_table(t, s, e)
    t_B = (ramp.size + hold.size + A.size) / FS
    inB = tab.t_start >= t_B - 0.5
    assert tab.n_regular == pytest.approx(nB - 1, abs=1)
    assert np.all(tab.regular[inB][:-1])
    assert not np.any(tab.regular[~inB])
    if head_range > tab.band:
        assert (~inB).sum() >= 100                     # A's loops are counted, and marked
    else:
        assert (~inB).sum() <= 1                       # A's loops lie within the band
    assert tab.head == pytest.approx(t_B, abs=1 / fB + 1)
    assert tab.f == pytest.approx(fB, rel=1e-4)


def test_a_hold_has_no_regular_cycle():
    """A hold read with the default band: the band is noise-sized, so noise makes "cycles", but they form no
    dominant block and none is regular."""
    t = np.arange(20000) / FS
    s = 14.0 + np.random.default_rng(2).normal(0, 0.001, t.size)
    tab = cycle_table(t, s, [s / 36, s / 37], names=['a', 'b'])
    assert len(tab) > 0
    assert tab.n_regular == 0 and not tab.regular.any()
    assert np.isnan(tab.f) and np.isnan(tab.head)
    assert np.all(np.isnan(tab.follow))


def test_empty_step():
    t = np.arange(2000) / FS
    s = np.full(t.size, 14.0)
    tab = cycle_table(t, s, [s / 36, s / 37], names=['a', 'b'])
    assert len(tab) == 0 and tab.n_regular == 0
    assert tab.eps_up.shape == (2, 0)
    assert np.isnan(tab.f)
    tab = cycle_table(t[:2], s[:2], s[:2])
    assert len(tab) == 0
    q = channel_quality(tab)
    assert q['n_jumps'].tolist() == [0]


# ---------------------------------------------------------------------------------------------------------------
# G2: NaN, a sticking channel, a clipped channel, a jump
# ---------------------------------------------------------------------------------------------------------------
def test_nan_voids_only_the_cycles_it_touches():
    t, s, e, _ = loops(25.5, 200, phi_deg=1.5)
    e1 = e.copy()
    e1[1000:1033] = np.nan
    tab = cycle_table(t, s, [e, e1], names=['clean', 'gap'])
    ref = cycle_table(t, s, [e, e])
    assert len(tab) == len(ref)
    touched = (tab.i_up - 1 <= 1032) & (tab.i_end + 1 >= 1000)
    assert 2 <= touched.sum() <= 3
    for col in ('eps_up', 'eps_lo', 'E_sec', 'w_loop', 'eps_width'):
        v = getattr(tab, col)
        assert np.all(np.isfinite(v[0]))
        assert np.all(np.isnan(v[1][touched]))
        assert np.all(np.isfinite(v[1][~touched]))
        assert np.allclose(v[1][~touched], getattr(ref, col)[1][~touched])


def test_sticking_and_clipped_channels():
    t, s, e, _ = loops(25.5, 300, phi_deg=1.5)
    stick = play(e, 0.65 * DEPS)
    clip = np.clip(e, EPS_M - 0.4 * DEPS, EPS_M + 0.4 * DEPS)
    tab = cycle_table(t, s, [e, stick, clip], names=['clean', 'stick', 'clip'])
    q = channel_quality(tab)
    assert q['follow'][0] == pytest.approx(1.0)
    assert q['h3'][0] < 0.01
    assert q['still_share'][0] < 0.12
    for c in (1, 2):
        assert q['follow'][c] < 0.5
        assert q['h3'][c] > 0.10
        assert q['still_share'][c] > 0.3
    assert q['E_sec'][1] > 2 * q['E_sec'][0]           # the stiff false secant of a sticking channel
    assert q['loss_angle'][1] > 20                     # and its wide false loop


def test_jump_and_not_a_spike():
    t, s, e, _ = loops(25.5, 3000, phi_deg=1.5)
    rng = np.random.default_rng(3)
    ch = np.vstack([e, e, e]) + rng.normal(0, 3e-4, (3, e.size))
    t_jump = 1234.56
    ch[2, t >= t_jump] += 0.007
    one = (t >= 2000.0) & (t < 2000.0 + 25.5 / FS)     # one cycle off by 0.02: an outlier, not a jump
    ch[0, one] += 0.02
    tab = cycle_table(t, s, ch)
    assert tab.jumps[0].shape == (0, 2)
    assert tab.jumps[1].shape == (0, 2)
    assert tab.jumps[2].shape == (1, 2)
    tj, size = tab.jumps[2][0]
    assert 0 <= tj - t_jump < 25.5 / FS + 0.1
    assert size == pytest.approx(0.007, rel=0.2)
    assert channel_quality(tab)['n_jumps'].tolist() == [0, 0, 1]


def test_a_channel_resting_in_quanta_has_no_jumps():
    """CYCRE-W's LVDTs rest in quanta of 0.1 µm (0.0007 ‰ over 150 mm). The median of the cycle-to-cycle
    differences is then zero; the scatter must not vanish with it, or every quantum reads as a jump."""
    t, s, e, _ = loops(25.5, 3000, phi_deg=1.5)
    q = 0.1e-3 / 150 * 1e3
    rng = np.random.default_rng(5)
    sticky = np.round((play(e, 0.3 * DEPS) + rng.normal(0, 0.2 * q, e.size)) / q) * q
    tab = cycle_table(t, s, sticky, refine=False)
    d = np.diff(tab.eps_up[0][tab.regular])
    assert np.median(np.abs(d)) == 0                   # the case: most cycles repeat the same quantum
    assert tab.jumps[0].shape == (0, 2)


# ---------------------------------------------------------------------------------------------------------------
# G2: rounded time stamps
# ---------------------------------------------------------------------------------------------------------------
def test_rounded_time_stamps():
    t, s, e, f = loops(25.5, 1500, phi_deg=1.5, t0=9000.0)
    stamp = np.where(t < 10000.0, np.floor(t * 10 + 1e-6) / 10, np.floor(t + 1e-6))
    assert np.allclose(sample_times(stamp), t, atol=1e-6)
    tab = cycle_table(stamp, s, e)
    ref = cycle_table(t, s, e)
    assert len(tab) == len(ref)
    assert np.max(np.abs(tab.t_start - ref.t_start)) < 1e-6
    assert tab.f == pytest.approx(f, rel=1e-6)


def test_time_must_not_decrease():
    with pytest.raises(ValueError):
        sample_times([0.0, 1.0, 0.5])


# ---------------------------------------------------------------------------------------------------------------
# The reductions
# ---------------------------------------------------------------------------------------------------------------
def test_branch_rate_with_temperature_as_covariate():
    spl = 25.5
    t, s, _, f = loops(spl, int(4 * 3600 * FS / spl))
    rate = 10e-3 / 3600                                 # ‰/s
    kT = 8e-3                                           # ‰/K
    T = 23.0 - 2.0 / (1 + np.exp(-(t - 2 * 3600) / 600))
    drift = rate * t + kT * (T - 23.0)
    e = EPS_M + drift + DEPS * np.cos(2 * np.pi * f * t - np.radians(1.5))
    tab = cycle_table(t, s, [e, e], names=['a', 'b'], extra=T, extra_names=['T_surface'])
    plain = branch_rate(tab, [2 * 3600], 3600)
    withT = branch_rate(tab, [2 * 3600], 3600, covariate='T_surface')
    assert withT.up[0, 0] == pytest.approx(rate, rel=0.02)
    assert withT.lo[0, 1] == pytest.approx(rate, rel=0.02)
    assert withT.cov_up[0, 0] == pytest.approx(kT, rel=0.02)
    assert abs(plain.up[0, 0] - rate) > 0.5 * rate     # without T the drop of 2 K swamps the rate
    assert withT.cov_range[0] == pytest.approx(2.0, abs=0.1)
    assert not withT.jump.any()
    assert withT.n[0, 0] == pytest.approx(2 * 3600 * f, rel=0.01)
    only_b = branch_rate(tab, [3600, 3 * 3600], 1800, channels=['b'])
    assert only_b.up.shape == (2, 1) and only_b.names == ('b',)


def test_table_is_frozen_and_complete():
    t, s, e, _ = loops(25.5, 50)
    tab = cycle_table(t, s, e, names=['u1'])
    with pytest.raises(AttributeError):
        tab.band = 1.0
    with pytest.raises(ValueError):
        tab.E_sec[0, 0] = 1.0
    d = tab.to_dict()
    for key in (*tab.COLUMNS, *tab.CHANNEL_COLUMNS, *tab.EXTRA_COLUMNS, *tab.CHANNEL_ATTRS, *tab.STEP_ATTRS):
        assert key in d
    assert d['version'] == cycles.__version__
    assert tab.extra_up.shape == (0, len(tab))


# ---------------------------------------------------------------------------------------------------------------
# G5 (with margin; the 1 s of G5 is checked on CYCRE-W by dev/tools/check_cycrew.py)
# ---------------------------------------------------------------------------------------------------------------
def test_an_eleven_hour_leg_runs_in_seconds():
    t, s, e, _ = loops(25.5, int(11 * 3600 * FS / 25.5), phi_deg=1.5)
    ch = np.vstack([e, 1.1 * e, 0.9 * e, e]) + np.random.default_rng(4).normal(0, 3e-4, (4, e.size))
    t0 = clock.perf_counter()
    tab = cycle_table(t, s, ch)
    dt = clock.perf_counter() - t0
    assert len(tab) > 30_000
    assert dt < 3.0
