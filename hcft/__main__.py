import sys

from hcft.api import HCFT
from hcft.app_icon import ICON_FILE
from hcft.utils.plot_style import set_latex_mpl_format


def set_app_icon():
    """Show the HCFT icon on all windows and in the Windows taskbar."""
    if sys.platform == 'win32':
        import ctypes
        # Own taskbar identity, otherwise Windows groups the tool under python.exe
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('bmcs-group.hcft')

    from pyface.qt import QtGui
    # pyface reuses this QApplication instead of creating its own
    app = QtGui.QApplication.instance() or QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(str(ICON_FILE)))


def main():
    # To use Times New Roman font and LaTeX style math in matplotlib
    set_latex_mpl_format(font_size=18)
    set_app_icon()

    hcft = HCFT()
    hcft.configure_traits()


if __name__ == '__main__':
    main()
