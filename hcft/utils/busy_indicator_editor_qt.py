from pyface.qt import QtGui
from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.qt.editor import Editor


class _BusyIndicatorEditor(Editor):
    """Shows the edited Str trait (the running task) next to an animated progress bar"""

    def init(self, parent):
        self.control = QtGui.QWidget()
        layout = QtGui.QHBoxLayout(self.control)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QtGui.QLabel()
        font = self._label.font()
        font.setBold(True)
        self._label.setFont(font)
        self._progress_bar = QtGui.QProgressBar()
        # Equal min and max makes the progress bar show a busy animation instead of a percentage
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._label)
        layout.addWidget(self._progress_bar, stretch=1)
        self.set_tooltip()

    def update_editor(self):
        self._label.setText(self.str_value)


class BusyIndicatorEditor(BasicEditorFactory):
    klass = _BusyIndicatorEditor
