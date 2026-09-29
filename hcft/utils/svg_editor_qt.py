from pyface.qt import QtCore, QtGui
from pyface.qt.QtSvg import QSvgRenderer
from traitsui.basic_editor_factory import BasicEditorFactory

try:
    from traitsui.qt.editor import Editor
except ImportError:
    # traitsui < 7.0
    from traitsui.qt4.editor import Editor


class _SVGWidget(QtGui.QWidget):
    """Draws an SVG file as vector graphics, so it stays sharp at any size, keeping its aspect ratio."""

    def __init__(self, svg_file, parent=None):
        super().__init__(parent)
        self.renderer = QSvgRenderer(svg_file, self)
        default_size = self.renderer.defaultSize()
        self.setMinimumSize(default_size * 0.6)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

    def sizeHint(self):
        return self.renderer.defaultSize()

    def paintEvent(self, event):
        # Largest rectangle with the SVG aspect ratio that fits in the widget, centered
        size = QtCore.QSizeF(self.renderer.defaultSize())
        size.scale(QtCore.QSizeF(self.size()), QtCore.Qt.KeepAspectRatio)
        rect = QtCore.QRectF(QtCore.QPointF(0, 0), size)
        rect.moveCenter(QtCore.QRectF(self.rect()).center())

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.renderer.render(painter, rect)
        painter.end()


class _SVGEditor(Editor):
    """Read-only editor showing the SVG file whose path is the trait value."""

    def init(self, parent):
        self.control = _SVGWidget(self.value)
        self.set_tooltip()

    def update_editor(self):
        self.control.renderer.load(self.value)
        self.control.update()


class SVGEditor(BasicEditorFactory):

    klass = _SVGEditor
