"""The GUI's "Generate filtered and creep npy" on the core (concept 01 §5, H4), headless.

A CSV is parsed and cut as the GUI does it, without opening a window: an ascending branch, the tail of a smaller
block below the cycle range, and the cycles, with noise extrema and plateaus at the peaks.
"""

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for module in ('traits', 'traitsui', 'pyface', 'pandas', 'scipy', 'matplotlib'):
    pytest.importorskip(module)

from hcft.cycles import turning_points  # noqa: E402
from hcft.hcft import HCFT  # noqa: E402
from hcft.helper_classes.csv_tools import get_headers  # noqa: E402

FS = 20.0
N_MAIN = 200


def signal():
    rng = np.random.default_rng(11)
    f = FS / 25.5
    ramp = np.linspace(0.0, -300.0, int(20 * FS))
    t_head = np.arange(int(60 * FS)) / FS
    head = -290.0 - 10.0 * np.cos(2 * np.pi * f * t_head)          # 20 kN loops: below the cycle range
    t_main = np.arange(int(N_MAIN * 25.5)) / FS
    main = -265.0 - 35.0 * np.cos(2 * np.pi * f * t_main)          # 70 kN loops
    F = np.concatenate((ramp, head, main))
    F = np.round((F + rng.normal(0, 0.3, F.size)) / 0.05) * 0.05
    F[:ramp.size] = ramp
    t = np.arange(F.size) / FS
    u = F / 2000.0 + rng.normal(0, 1e-4, F.size)
    return t, F, u


@pytest.fixture
def hcft(tmp_path):
    t, F, u = signal()
    path = tmp_path / 'test.csv'
    with open(path, 'w') as f:
        f.write('time;force;disp\n')
        for row in zip(t, F, u, strict=True):
            f.write('{:.3f};{:.2f};{:.6f}\n'.format(*row))
    h = HCFT()
    h.decimal, h.delimiter, h.skip_first_rows = '.', ';', 1
    h.file_path = str(path)
    h.columns_headers = get_headers(str(path), delimiter=';', decimal='.')
    h.npy_folder_path = str(tmp_path / 'NPY')
    os.makedirs(h.npy_folder_path)
    h.file_name = 'test'
    h.file_loaded = True
    h.time_column, h.force_column = 'time', 'force'
    h.parse_csv_to_npy_fired()
    h.peak_force_before_cycles = 250.0
    return h


def creep(h):
    h.generate_filtered_and_creep_npy_fired()
    assert 'SOMETHING WENT WRONG' not in h.log, h.log
    return {c: (np.load(h.get_max_npy_file_path(c)), np.load(h.get_min_npy_file_path(c))) for c in ('force', 'disp')}


def test_min_cycle_range_is_the_core_band(hcft):
    hcft.cutting_method = 'Define min cycle range(force difference)'
    hcft.min_cycle_force_range = 50.0
    out = creep(hcft)
    fmax, fmin = out['force']
    assert abs(fmin.size - N_MAIN) <= 1 and abs(fmax.size - fmin.size) <= 1
    assert np.all(np.abs(fmax + 300.0) < 2.0)                   # the peaks of compression, one per cycle
    assert np.all(np.abs(fmin + 230.0) < 2.0)
    assert out['disp'][0].size == fmax.size and out['disp'][1].size == fmin.size
    F = np.load(hcft.get_npy_file_path('force'))
    rest = F[np.flatnonzero(np.abs(F) > 250.0)[0]:]
    idx, kind = turning_points(-rest, 50.0)
    up, lo = idx[kind == 1], idx[kind == -1]
    assert np.array_equal(fmax, rest[up])
    assert np.array_equal(fmin, rest[lo[lo > up[0]]])          # not the foot of the loading ramp


def test_max_min_levels_filter_the_turning_points(hcft):
    hcft.cutting_method = 'Define Max, Min'
    hcft.force_max, hcft.force_min = 290.0, 240.0
    fmax, fmin = creep(hcft)['force']
    assert abs(fmin.size - N_MAIN) <= 1 and abs(fmax.size - fmin.size) <= 1
    assert np.all(np.abs(fmax) > 290.0) and np.all(np.abs(fmin) < 240.0)


def test_an_impossible_range_is_refused(hcft, monkeypatch):
    monkeypatch.setattr('hcft.hcft.GUI.invoke_later', lambda *args, **kw: None)   # no dialog without a GUI loop
    hcft.cutting_method = 'Define min cycle range(force difference)'
    hcft.min_cycle_force_range = 0.0
    hcft.generate_filtered_and_creep_npy_fired()
    assert 'must be positive' in hcft.log
    assert not os.path.exists(hcft.get_max_npy_file_path('force'))
