from pyface.qt import QtGui
from traitsui.api import TextEditor
from traitsui.qt.text_editor import CustomEditor


class _LogEditor(CustomEditor):
    """Multi-line text editor which scrolls down to the last line whenever the text changes"""

    def update_editor(self):
        super().update_editor()
        self.control.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.control.ensureCursorVisible()


class LogEditor(TextEditor):
    """Text editor factory for logs, use it with style='custom'"""

    def _get_custom_editor_class(self):
        return _LogEditor
