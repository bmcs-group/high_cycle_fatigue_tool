"""hcft-table: the cycle table of a NetCDF file, written as CSV (concept 01 §5).

    hcft-table FILE.nc --load force --channels u_axial_1,u_axial_2 [options]

The command knows no schema: the variable names are arguments. The load is handed in with the sign that makes the
peak load the maximum and the channels in the same sign convention, so ``--load-factor -1 --channel-factor -1``
turn a force and displacements that are negative in compression; the factors also convert units (``--load-factor`` −10³/A turns kN into MPa for A in mm², ``--channel-factor`` 10³/L turns mm into ‰
for L in mm). With ``--step-var`` the record is read step by step, one table per step, as the core expects. Each
step's channel quality goes to stderr.

Needs the extra ``nc`` (netCDF4): ``pip install 'hcft[nc]'``.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .cycles import CycleTable, channel_quality, cycle_table
from .version import __version__


def _list(text):
    return [s.strip() for s in text.split(',') if s.strip()] if text else []


def read_nc(path, names):
    """The named variables of a NetCDF file as float arrays, flattened, masked values as NaN."""
    try:
        import netCDF4
    except ImportError as err:
        raise SystemExit("hcft-table needs netCDF4: pip install 'hcft[nc]'") from err
    out = {}
    with netCDF4.Dataset(path) as ds:
        for name in names:
            if name not in ds.variables:
                raise SystemExit(f'{path}: no variable {name!r}; it has {", ".join(ds.variables)}')
            v = np.ma.asanyarray(ds[name][:]).astype(float)
            out[name] = np.ma.filled(v, np.nan).ravel()
    return out


def _label(v):
    return str(int(v)) if float(v).is_integer() else f'{v:g}'


def header(tab: CycleTable):
    cols = ['step', *CycleTable.COLUMNS]
    cols += [f'{c}[{n}]' for n in tab.names for c in CycleTable.CHANNEL_COLUMNS]
    cols += [f'extra_up[{n}]' for n in tab.extra_names]
    return cols


def rows(step, tab: CycleTable, regular_only=False):
    per_cycle = [getattr(tab, c) for c in CycleTable.COLUMNS]
    per_channel = [getattr(tab, c)[i] for i in range(len(tab.names)) for c in CycleTable.CHANNEL_COLUMNS]
    per_extra = [tab.extra_up[i] for i in range(len(tab.extra_names))]
    for j in range(len(tab)):
        if regular_only and not tab.regular[j]:
            continue
        values = [a[j] for a in (*per_cycle, *per_channel, *per_extra)]
        yield [step, *(int(v) if isinstance(v, (bool, np.bool_)) else f'{v:.9g}' for v in values)]


def summary(step, tab: CycleTable, out):
    print(f'step {step}: {len(tab)} cycles, {tab.n_regular} regular, f {tab.f:.4f}, head {tab.head:.0f}, '
          f'band {tab.band:.4g}', file=out)
    q = channel_quality(tab)
    for c, name in enumerate(tab.names):
        print(f'  {name}: follow {q["follow"][c]:.2f}, still {100 * q["still_share"][c]:.0f} %, '
              f'h3 {100 * q["h3"][c]:.1f} %, E_sec {q["E_sec"][c]:.4g} ±{100 * q["E_scatter"][c]:.1f} %, '
              f'w_loop {q["w_loop"][c]:.4g}, loss angle {q["loss_angle"][c]:.2f}°, jumps {q["n_jumps"][c]}',
              file=out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='hcft-table', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('file', help='the NetCDF file')
    p.add_argument('--time', default='time', help='the time variable [s] (default: time)')
    p.add_argument('--load', required=True, help='the load variable (the controlled quantity)')
    p.add_argument('--channels', required=True, help='the deformation variables, comma-separated')
    p.add_argument('--extra', default='', help='variables read at the upper turning points, comma-separated')
    p.add_argument('--step-var', default=None, help='a per-sample step variable: one table per step')
    p.add_argument('--steps', default='', help='the steps to read, comma-separated (default: all)')
    p.add_argument('--load-factor', type=float, default=1.0, help='the load is multiplied by this (default 1)')
    p.add_argument('--channel-factor', type=float, default=1.0, help='the channels are multiplied by this')
    p.add_argument('--band', type=float, default=None, help='the hysteresis band in (scaled) load units')
    p.add_argument('--band-rel', type=float, default=0.5, help="the band's share of the step's range (0.5)")
    p.add_argument('--regular-tol', type=float, default=0.15)
    p.add_argument('--period-tol', type=float, default=0.25)
    p.add_argument('--regular-only', action='store_true', help='write the regular cycles only')
    p.add_argument('-o', '--output', default='-', help='the CSV file (default: stdout)')
    p.add_argument('--version', action='version', version=f'hcft {__version__}')
    a = p.parse_args(argv)

    channels, extra = _list(a.channels), _list(a.extra)
    names = [a.time, a.load, *channels, *extra] + ([a.step_var] if a.step_var else [])
    d = read_nc(a.file, list(dict.fromkeys(names)))
    if a.step_var:
        s = d[a.step_var]
        wanted = [float(v) for v in _list(a.steps)] if a.steps else [v for v in np.unique(s) if np.isfinite(v)]
        steps = [(_label(v), s == v) for v in wanted]
    else:
        steps = [('', np.ones(d[a.time].size, dtype=bool))]

    out = sys.stdout if a.output == '-' else open(a.output, 'w', newline='')
    try:
        w = csv.writer(out)
        first = True
        for label, sel in steps:
            tab = cycle_table(d[a.time][sel], a.load_factor * d[a.load][sel],
                              np.array([a.channel_factor * d[c][sel] for c in channels]), names=channels,
                              extra=np.array([d[e][sel] for e in extra]) if extra else None, extra_names=extra,
                              band=a.band, band_rel=a.band_rel, regular_tol=a.regular_tol, period_tol=a.period_tol)
            summary(label or 'all', tab, sys.stderr)
            if first:
                w.writerow(header(tab))
                first = False
            w.writerows(rows(label, tab, a.regular_only))
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
