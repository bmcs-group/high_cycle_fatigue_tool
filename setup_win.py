import os
import sys
from cx_Freeze import setup, Executable
import cx_Freeze.hooks
import matplotlib
import scipy

scipy_path = os.path.dirname(scipy.__file__)

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))

packages_to_fetch = [
    'mayavi',
    'numpy',
    'six',
    'pyface',
    'traits',
    'traitsui'
]
base_path = os.path.join(PYTHON_INSTALL_DIR, 'Lib', 'site-packages')

folders = list(next(os.walk(base_path))[1])
folders = [fol for fol in folders if fol.endswith('-info')]
folders = [fol for fol in folders if fol.split(
    '-')[0].lower() in packages_to_fetch]
out = [f for f in folders]
folders = [os.path.join(base_path, fol) for fol in folders]
folders = [(fol, '.\lib' + '\\' + out[k]) for k, fol in enumerate(folders)]

build_exe_options = {
    "packages": ['sys',
                 'os',
                 #                  'numpy',
                 #                  'mayavi',
                 'pygments',
                 #                  'traitsui.qt4.toolkit',
                 'pathlib',
                 'scipy',
                 'string',
                 'pyface',
                 'matplotlib',
                 'pandas',
                 'util.traits'
                 #                  'mayavi.core',
                 #                  'mayavi.core.ui',
                 #                  'mayavi.core.ui.api',
                 #                  "pyface.ui.qt4",
                 #                  "matplotlib.backends.backend_qt4",
                 #                  'pyface.qt',
                 #                  "tvtk.vtk_module",
                 #                  "tvtk.pyface.ui.wx",
                 #                  'tvtk.pyface.ui.qt4',
                 #                  'pyface.qt.QtGui',
                 #                  'pyface.qt.QtCore'
                 ],
    # to fix a bug in importing this module
    "excludes": ['scipy.spatial.cKDTree'],
    "includes": [],

    "build_exe": r'D:\cx_freeze_build_exe',

    "include_files": folders  # .append((str(scipy_path), 'scipy'))
}

executables = [
    Executable(r'D:\Homam\Synced\Programming\Python\high-cycle-fatigue-tool\hcft.py',
               targetName="hcft.exe", base='Win32GUI')
]

setup(name='hcft',
      version='1.0',
      description='',
      options={"build_exe": build_exe_options},
      executables=executables
      )
