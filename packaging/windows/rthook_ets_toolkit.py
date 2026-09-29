# PyInstaller runtime hook: select the Qt toolkit explicitly, so pyface/traitsui
# and matplotlib never try to probe for other (not bundled) toolkits.
import os

os.environ.setdefault('ETS_TOOLKIT', 'qt')
os.environ.setdefault('QT_API', 'pyqt5')
