"""hcft-table on a small NetCDF file (concept 01 §5, H4)."""

import csv
import shutil
import subprocess

import numpy as np
import pytest

from hcft.cli import main
from hcft.cycles import cycle_table

netCDF4 = pytest.importorskip('netCDF4')

FS = 20.0
AREA = np.pi * 150.0 ** 2 / 4.0
L0 = 150.0


@pytest.fixture
def nc_file(tmp_path):
    """A hold (step 0) and 300 loops at 0.78 Hz (step 1): force negative in compression [kN], two LVDTs [mm]."""
    rng = np.random.default_rng(7)
    t0 = np.arange(int(200 * FS)) / FS
    t1 = t0[-1] + 1 / FS + np.arange(int(300 * 25.5)) / FS
    f = FS / 25.5
    F = np.concatenate((np.full(t0.size, -250.0), -265.0 - 35.0 * np.cos(2 * np.pi * f * (t1 - t1[0]))))
    F += rng.normal(0, 0.2, F.size)
    u1 = F / 1.8e3 * L0 / AREA * 1e3 * 0.025      # a linear channel, scaled to about 8 µm per loop
    u2 = 1.1 * u1
    path = tmp_path / 'test.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('time', F.size)
        for name, values in (('time', np.concatenate((t0, t1))), ('force', F), ('u_axial_1', u1),
                             ('u_axial_2', u2), ('T_surface', np.full(F.size, 23.0)),
                             ('step_index', np.repeat([0, 1], [t0.size, t1.size]))):
            ds.createVariable(name, 'f8', ('time',))[:] = values
    return path


ARGS = ['--load', 'force', '--channels', 'u_axial_1,u_axial_2', '--extra', 'T_surface',
        '--load-factor', str(-1e3 / AREA), '--channel-factor', str(-1e3 / L0)]


def read(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def test_one_table_per_step(nc_file, tmp_path, capsys):
    out = tmp_path / 'table.csv'
    assert main([str(nc_file), *ARGS, '--step-var', 'step_index', '--steps', '1', '-o', str(out)]) == 0
    rows = read(out)
    head, body = rows[0], rows[1:]
    assert head[:3] == ['step', 'cycle', 't_start']
    assert 'E_sec[u_axial_2]' in head and 'extra_up[T_surface]' in head
    with netCDF4.Dataset(nc_file) as ds:
        sel = ds['step_index'][:] == 1
        tab = cycle_table(ds['time'][:][sel], -1e3 / AREA * ds['force'][:][sel],
                          [-1e3 / L0 * ds[c][:][sel] for c in ('u_axial_1', 'u_axial_2')])
    assert len(body) == len(tab)
    assert {r[0] for r in body} == {'1'}
    j = head.index('E_sec[u_axial_1]')
    assert float(body[10][j]) == pytest.approx(tab.E_sec[0][10], rel=1e-8)
    assert 'step 1:' in capsys.readouterr().err


def test_regular_only_and_all_steps(nc_file, tmp_path):
    out = tmp_path / 'table.csv'
    main([str(nc_file), *ARGS, '--step-var', 'step_index', '--regular-only', '-o', str(out)])
    body = read(out)[1:]
    regular = read(out)[0].index('regular')
    assert all(r[regular] == '1' for r in body)
    assert {r[0] for r in body} <= {'0', '1'}


def test_a_missing_variable_is_named(nc_file):
    with pytest.raises(SystemExit, match="no variable 'stroke'"):
        main([str(nc_file), '--load', 'force', '--channels', 'stroke'])


def test_the_console_script():
    exe = shutil.which('hcft-table')
    if exe is None:
        pytest.skip('hcft-table is not installed in this environment')
    out = subprocess.run([exe, '--version'], capture_output=True, text=True, check=True)
    assert out.stdout.startswith('hcft ')
