# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Windows 64-bit build of HCFT (one-folder mode).

Run through build.ps1, or directly from the repository root:
    uv run --group build pyinstaller packaging/windows/hcft.spec --noconfirm
Output: dist/hcft/hcft.exe
"""
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo)

HERE = Path(SPECPATH)
ROOT = HERE.parents[1]

VERSION = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
                    (ROOT / 'hcft' / 'version.py').read_text()).group(1)
VERSION_TUPLE = tuple((list(map(int, re.findall(r'\d+', VERSION))) + [0, 0, 0, 0])[:4])


def not_wx_or_tests(name):
    return not {'wx', 'tests'} & set(name.split('.'))


# pyface and traitsui find their GUI toolkit through package entry points
# ("pyface.toolkits", "traitsui.toolkits"), so their metadata must be bundled.
# They also import most of their modules lazily (toolkit objects, module-level
# __getattr__ in the api modules), which PyInstaller can't see, so all their
# submodules are collected explicitly.
datas = [(str(ROOT / 'hcft' / 'resources'), 'hcft/resources')]
hiddenimports = []
for package in ('pyface', 'traitsui'):
    datas += copy_metadata(package)
    datas += collect_data_files(package, excludes=['**/tests/**'])  # editor/dialog images
    hiddenimports += collect_submodules(package, filter=not_wx_or_tests)
datas += copy_metadata('traits')

excludes = [
    # Other GUI toolkits / Qt bindings: only PyQt5 is used
    'wx', 'pyface.ui.wx', 'traitsui.wx', 'tkinter', '_tkinter',
    'PySide2', 'PySide6', 'PyQt6',
    # Development tools that may be present in the environment
    'IPython', 'jupyter', 'notebook', 'pytest',
]

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE),
    kids=[
        StringFileInfo([StringTable('040904B0', [
            StringStruct('CompanyName', 'RWTH Aachen University - Institute of Structural Concrete'),
            StringStruct('FileDescription', 'High-Cycle Fatigue Tool'),
            StringStruct('FileVersion', VERSION),
            StringStruct('InternalName', 'hcft'),
            StringStruct('LegalCopyright', 'GPL-3.0, H. Spartali & BMCS-Group'),
            StringStruct('OriginalFilename', 'hcft.exe'),
            StringStruct('ProductName', 'High-Cycle Fatigue Tool'),
            StringStruct('ProductVersion', VERSION),
        ])]),
        VarFileInfo([VarStruct('Translation', [0x0409, 1200])]),
    ],
)

a = Analysis(
    [str(HERE / 'hcft_launcher.py')],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[str(HERE / 'rthook_ets_toolkit.py')],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hcft',
    icon=str(ROOT / 'hcft' / 'resources' / 'hcft_icon.ico'),
    version=version_info,
    console=False,  # GUI app, no console window
    upx=False,      # UPX-packed files often trigger antivirus false positives
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name='hcft',
    upx=False,
)
