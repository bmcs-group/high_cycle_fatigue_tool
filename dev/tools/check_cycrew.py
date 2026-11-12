"""H3: concept 01's numbers from CYCRE-W's consolidated files, read-only (gates G3, G4, G5).

Run outside CI, with netCDF4 beside the core:

    uv run --with netcdf4 python dev/tools/check_cycrew.py [DIR] [--quick]

DIR defaults to the share's CYCRE-W/Consolidated folder (``/mnt/reproxis``, mounted read-only). The script writes
nothing. ``--quick`` reads SWP-002 and the two legs of G4's counts only.

CYCRE-W's geometry (Ø150 mm, gauge length 150 mm) and its variable names live here, in the caller, as they will
in reproxis: the core knows neither.
"""

from __future__ import annotations

import sys
import time as clock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import hcft1_reference as hcft1  # noqa: E402

from hcft.cycles import (  # noqa: E402
    branch_rate,
    channel_quality,
    cycle_table,
    turning_points,
)

SHARE = Path('/mnt/reproxis/processes/research_projects/2025-2028_ERC_FatResCon-P15/CYCRE-W/Consolidated')
AREA = np.pi * 150.0 ** 2 / 4.0   # mm²
L0 = 150.0                        # mm
QUICK = ('CYCREW-SWP-002', 'CYCREW-SW-H-eswitch_creep_to_cyclic-r1', 'CYCREW-SW-L-eswitch_cyclic_to_creep-r1',
         'CYCREW-SW-L-eswitch_creep_to_cyclic-r1')

# §2's numbers, keyed by (file, n-th FATIGUE step of the file)
COUNTS = {  # regular, all
    ('CYCREW-SW-H-eswitch_creep_to_cyclic-r1', 0): (30984, 30985),
    ('CYCREW-SW-L-eswitch_cyclic_to_creep-r1', 0): (30961, 30963),
    ('CYCREW-SWP-002', 0): (15374, 15375),
    ('CYCREW-SWP-002', 1): (3839, 3840),
    ('CYCREW-SWP-002', 2): (2191, 2230),
}
SWP002_HEAD = (176.0, 195.0, 214.0)              # s, first regular cycle after the step's start
SWP002_UPPER = (0.7507, 0.7836, 0.8354)          # ‰, mean of the channels, last ten minutes
SWP002_E_MEAN = (45.6, 36.5, 34.9)               # GPa, secant of the mean strain, last ten minutes
SWP002_E = ((102.0, 35.9, 42.4, 37.0), (42.4, 33.0, 39.0, 33.1), (38.4, 31.2, 38.3, 31.2))  # GPa per channel
SWP002_FOLLOW_1 = (0.35, 1.00, 0.85, 0.97)       # block 1, scratch estimator
SWP002_STILL_1 = (0.40, 0.07, 0.16, 0.08)
SWP002_H3_1 = (0.18, 0.009, 0.033, 0.013)        # scratch: least-squares harmonics over 200 s windows
ANGLES = {  # loss angle [°] over SWP's three blocks
    ('CYCREW-SWP-002', 'u_axial_4'): (2.13, 1.52, 1.52),
    ('CYCREW-SWP-002', 'u_axial_2'): (1.82, 1.58, 1.74),
    ('CYCREW-SWP-001', 'u_axial_4'): (1.61, 1.42, 1.51),
    ('CYCREW-SWP-003', 'u_axial_4'): (2.26, 1.92, 1.96),
}
# §2.5's jumps. u_axial_2 steps +0.032 ‰ from one cycle to the next and slides on for about six cycles: the core
# reports the level shift, +0.051 ‰.
JUMPS = {('CYCREW-SWP-002', 2): (('u_axial_2', 73776.0, 0.051), ('u_axial_3', 60937.0, 0.007))}

RESULTS: list[tuple[str, bool, str]] = []


def check(gate, ok, detail):
    RESULTS.append((gate, bool(ok), detail))
    print(f'    [{"PASS" if ok else "FAIL"}] {gate}: {detail}')


def load(path):
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        d = {k: np.asarray(ds[k][:]) for k in ds.variables
             if ds[k].dimensions == ('time',) and k != 'datetime'}
        d['step_phase'] = [str(s) for s in ds['step_phase'][:]]
    return d


def step_arrays(d, step):
    """t [s], F [kN], σ [MPa, compression +], ε [‰] (m, n), the LVDTs' names, T_surface [°C] with spikes out."""
    sel = d['step_index'] == step
    names = sorted(k for k in d if k.startswith('u_axial'))
    F = d['force'][sel].astype(float)
    sig = -F * 1e3 / AREA
    eps = np.array([d[k][sel].astype(float) / L0 * 1e3 for k in names])
    T = d['T_surface'][sel].astype(float) if 'T_surface' in d else np.full(sel.sum(), np.nan)
    T[(T < 10) | (T > 40)] = np.nan
    return d['time'][sel].astype(float), F, sig, eps, names, T


def compare_hcft1(F, sig, band, tab):
    """G3: the core's turning points against hcft 1.1.0's extrema with its range cut at the same threshold."""
    idx, kind = turning_points(sig, band)
    rng = 0.5 * np.subtract(*np.percentile(F, [99.5, 0.5]))
    mx, mn = hcft1.get_array_max_and_min_indices(F)
    mx, mn = hcft1.cut_indices_of_defined_range(F, mx, mn, rng)
    out = {}
    first = tab.i_up[np.argmax(tab.regular)] if tab.n_regular else 0
    last = tab.i_end[len(tab.regular) - 1 - np.argmax(tab.regular[::-1])] if tab.n_regular else F.size
    for label, core, ref in (('upper', idx[kind == 1], np.asarray(mx)), ('lower', idx[kind == -1], np.asarray(mn))):
        c_only = np.setdiff1d(core, ref)
        r_only = np.setdiff1d(ref, core)
        doubles = 0
        used = set()
        for a in c_only:
            j = np.searchsorted(r_only, a)
            if j < r_only.size and r_only[j] not in used and np.all(F[a:r_only[j] + 1] == F[a]):
                doubles += 1
                used.add(r_only[j])
        rest_c = [a for a in c_only if not any(b > a and np.all(F[a:b + 1] == F[a]) for b in used)]
        rest_r = [b for b in r_only if b not in used]
        inner = [i for i in rest_c + rest_r if first < i < last]
        out[label] = dict(core=core.size, hcft=ref.size, doubles=doubles, rest_core=len(rest_c),
                          rest_hcft=len(rest_r), inner=len(inner))
    # hcft's positional pairing: kept pairs whose range is far from the cycles'
    n = min(len(mx), len(mn))
    pr = np.abs(F[np.asarray(mx[:n])] - F[np.asarray(mn[:n])])
    med = np.median(pr) if n else np.nan
    out['odd_pairs'] = [round(float(r), 1) for r in pr[(pr > 1.3 * med) | (pr < 0.7 * med)]]
    return out


def main(argv):
    quick = '--quick' in argv
    args = [a for a in argv if not a.startswith('--')]
    folder = Path(args[0]) if args else SHARE
    files = sorted(folder.glob('*.nc'))
    if quick:
        files = [f for f in files if f.stem in QUICK]
    print(f'CYCRE-W from {folder}: {len(files)} files')
    swp: dict[str, list] = {}
    clean_078: list[tuple[str, str, float, float, float, float]] = []
    timings = []
    n_jumps_total = 0
    for path in files:
        name = path.stem
        t0 = clock.perf_counter()
        d = load(path)
        t_read = clock.perf_counter() - t0
        fatigue = [i for i, ph in enumerate(d['step_phase']) if ph == 'FATIGUE']
        for k, step in enumerate(fatigue):
            t, F, sig, eps, names, T = step_arrays(d, step)
            ch = np.vstack([eps, np.nanmean(eps, axis=0)])
            t1 = clock.perf_counter()
            tab = cycle_table(t, sig, ch, names=[*names, 'mean'], extra=T, extra_names=['T_surface'])
            dt = clock.perf_counter() - t1
            timings.append((name, step, t.size, len(names), dt))
            q = channel_quality(tab)
            reg = tab.regular
            hours = (t[-1] - t[0]) / 3600
            print(f'\n=== {name} FATIGUE step {step} (#{k}): {t.size} samples ({hours:.1f} h, read {t_read:.1f} s), '
                  f'{len(tab)} cycles, {tab.n_regular} regular, head {tab.head:.0f} s, f {tab.f:.4f} Hz, '
                  f'σ {np.median(tab.sigma_lo[reg]):.2f} … {np.median(tab.sigma_up[reg]):.2f} MPa, '
                  f'table {dt * 1e3:.0f} ms')
            print('    channel    follow  still   h3     E_sec [GPa] ±1 cycle   w [J/m³]  ±100   angle [°]  jumps')
            for c, cn in enumerate(tab.names):
                print(f'    {cn:10s} {q["follow"][c]:5.2f}  {100 * q["still_share"][c]:4.0f} %  '
                      f'{100 * q["h3"][c]:5.1f} %  {q["E_sec"][c]:7.2f}  {100 * q["E_scatter"][c]:5.2f} %  '
                      f'{1e3 * q["w_loop"][c]:8.2f}  {100 * q["w_scatter"][c]:4.1f} %  {q["loss_angle"][c]:6.2f}  '
                      f'{" ".join(f"{tj:.0f}:{sz:+.4f}" for tj, sz in tab.jumps[c])}')
            n_jumps_total += sum(j.shape[0] for j in tab.jumps[:len(names)])

            g3 = compare_hcft1(F, sig, tab.band, tab)
            for label in ('upper', 'lower'):
                r = g3[label]
                check('G3', r['inner'] == 0,
                      f'{label}: core {r["core"]}, hcft 1 {r["hcft"]}, plateau doubles {r["doubles"]}, '
                      f'other differences {r["rest_core"]} / {r["rest_hcft"]} (inside the regular cycles {r["inner"]})')
            if g3['odd_pairs']:
                print(f'    hcft 1 keeps pairs of odd range [kN]: {g3["odd_pairs"]}')

            if (name, k) in COUNTS:
                n_reg, n_all = COUNTS[(name, k)]
                check('G4 counts', (tab.n_regular, len(tab)) == (n_reg, n_all),
                      f'{tab.n_regular} / {len(tab)} against §2.1 {n_reg} / {n_all}')
            if (name, k) in JUMPS:
                for cn, tj, size in JUMPS[(name, k)]:
                    j = tab.jumps[tab.names.index(cn)]
                    hit = j[np.abs(j[:, 0] - tj) < 3 / tab.f] if j.size else j
                    check('G4 jumps', hit.shape[0] == 1 and abs(hit[0, 1] - size) < 0.25 * size,
                          f'{cn} at {tj:.0f} s, {size:+.3f} ‰: found {hit.tolist()}')

            if name.startswith('CYCREW-SWP'):
                swp.setdefault(name, []).append((tab, q))
            if abs(tab.f - 0.78) < 0.05:
                # §2.4's cleanest channel: the lowest opening among the channels that do not rest (still < 30 %;
                # a resting or quantised channel reads a small false loop)
                moving = [c for c in range(len(names)) if q['still_share'][c] < 0.3]
                if moving:
                    c = min(moving, key=lambda c: q['w_loop'][c])
                    clean_078.append((name, tab.names[c], 1e3 * q['w_loop'][c], q['loss_angle'][c],
                                      q['w_scatter'][c], q['still_share'][c], q['follow'][c]))

            if name in ('CYCREW-SW-L-eswitch_creep_to_cyclic-r1', 'CYCREW-SW-H-eswitch_cyclic_to_creep-r1'):
                at = np.array([1.0, 5.0, 10.0]) * 3600
                plain = branch_rate(tab, at, 1800)
                withT = branch_rate(tab, at, 1800, covariate='T_surface')
                print('    branch_rate of eps_up at 1 / 5 / 10 h (± 0.5 h) [1e-3 ‰/h], plain | with T as covariate:')
                for c, cn in enumerate(tab.names):
                    p = ' '.join(f'{3.6e6 * v:+6.2f}' for v in plain.up[:, c])
                    w = ' '.join(f'{3.6e6 * v:+6.2f}' for v in withT.up[:, c])
                    print(f'      {cn:10s} {p} | {w}')
                print(f'      ΔT in the windows [K]: {" ".join(f"{v:.2f}" for v in withT.cov_range)}')

    print('\n=== SWP: over the three blocks')
    for name, blocks in sorted(swp.items()):
        if len(blocks) != 3:
            continue
        tabs = [b[0] for b in blocks]
        qs = [b[1] for b in blocks]
        print(f'  {name}: f {" / ".join(f"{tb.f:.4f}" for tb in tabs)} Hz, '
              f'head {" / ".join(f"{tb.head:.0f}" for tb in tabs)} s')
        for (nm, cn), ref in ANGLES.items():
            if nm != name:
                continue
            c = tabs[0].names.index(cn)
            got = [qq['loss_angle'][c] for qq in qs]
            check('G4 angle', all(abs(g - r) < 0.05 for g, r in zip(got, ref, strict=True)) and got[0] / got[2] < 2,
                  f'{name} {cn}: {" / ".join(f"{g:.2f}" for g in got)}° against {" / ".join(map(str, ref))}° '
                  f'(a time offset would give 7 : 1.75 : 1; read {got[0] / got[2]:.2f} : 1)')
        if name != 'CYCREW-SWP-002':
            continue
        for i, (tb, qq) in enumerate(blocks):
            last = tb.regular & (tb.t_start > tb.t_start[tb.regular][-1] - 600)
            m = tb.names.index('mean')
            up = float(np.mean(tb.eps_up[m][last]))
            E_mean = float(np.median(tb.E_sec[m][last]))
            check('G4 upper strain', abs(up - SWP002_UPPER[i]) < 5e-4,
                  f'block {i + 1}: {up:.4f} ‰ against {SWP002_UPPER[i]} ‰ (mean of the channels, last 10 min)')
            check('G4 E of the mean', abs(E_mean - SWP002_E_MEAN[i]) < 0.15,
                  f'block {i + 1}: {E_mean:.2f} GPa against {SWP002_E_MEAN[i]} GPa (last 10 min)')
            got = qq['E_sec'][:4]
            check('G4 E per channel', np.all(np.abs(got - SWP002_E[i]) < 0.15),
                  f'block {i + 1}: {" / ".join(f"{g:.1f}" for g in got)} GPa against '
                  f'{" / ".join(map(str, SWP002_E[i]))}')
            check('G4 head', abs(tb.head - SWP002_HEAD[i]) < 2,
                  f'block {i + 1}: first regular cycle at +{tb.head:.0f} s against +{SWP002_HEAD[i]:.0f} s')
        q1 = blocks[0][1]
        print('  SWP-002 block 1, channel quality: core | scratch')
        for c in range(4):
            print(f'    {q1["name"][c]}: follow {q1["follow"][c]:.2f} | {SWP002_FOLLOW_1[c]:.2f}   '
                  f'still {100 * q1["still_share"][c]:.0f} % | {100 * SWP002_STILL_1[c]:.0f} %   '
                  f'h3 {100 * q1["h3"][c]:.1f} % | {100 * SWP002_H3_1[c]:.1f} %')
        sticking = q1['follow'][0] < 0.5 and q1['h3'][0] > 0.10 and np.all(q1['follow'][1:4] >= 0.8)
        check('G4 sticking', sticking, 'SWP-002 block 1: u_axial_1 flagged by follow < 0.5 and h3 > 10 %, the '
              'others follow ≥ 0.8')

    print('\n=== 0.78 Hz: the cleanest channel per step (still < 30 %, lowest opening)')
    for name, cn, w, ang, ws, st, fo in clean_078:
        print(f'  {name:42s} {cn}: w {w:5.2f} J/m³, loss angle {ang:4.2f}°, 100-cycle mean ±{100 * ws:4.1f} %, '
              f'still {100 * st:3.0f} %, follow {fo:.2f}')
    if clean_078:
        ws = np.array([r[4] for r in clean_078])
        check('G4 opening', np.all(ws <= 0.20),
              f'100-cycle-mean scatter of w on the cleanest channel ≤ 20 % in {np.sum(ws <= 0.2)} of {ws.size} steps '
              f'(max {100 * np.nanmax(ws):.1f} %)')
        angles = np.array([r[3] for r in clean_078])
        print(f'  loss angle ≤ 2.8° in {np.sum(angles <= 2.8)} of {angles.size} steps; '
              f'the rest {", ".join(f"{a:.1f}" for a in angles[angles > 2.8])}°')

    print(f'\n=== jumps on the LVDT channels: {n_jumps_total} in all steps')
    long = [tm for tm in timings if tm[2] > 700_000]
    if long:
        worst = max(long, key=lambda tm: tm[4])
        check('G5', worst[4] < 1.0, f'slowest leg > 700 000 samples: {worst[0]} step {worst[1]}, {worst[2]} samples, '
              f'{worst[3]} + 1 channels, {worst[4]:.2f} s')

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f'\n=== {len(RESULTS)} checks, {n_fail} failed')
    for gate, ok, detail in RESULTS:
        if not ok:
            print(f'  FAIL {gate}: {detail}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
