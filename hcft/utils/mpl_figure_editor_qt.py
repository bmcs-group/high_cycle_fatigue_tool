#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — March 2014
"""
Qt adaptation of Gael Varoquaux's tutorial to integrate Matplotlib
http://docs.enthought.com/traitsui/tutorials/traits_ui_scientific_app.html#extending-traitsui-adding-a-matplotlib-figure-to-our-application

based on Qt-based code shared by Didrik Pinte, May 2012
http://markmail.org/message/z3hnoqruk56g2bje

adapted and tested to work with PySide from Anaconda in March 2014
"""

from matplotlib.backends.backend_qt5 import \
    NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyface.qt import QtGui
from traits.api import Instance
from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from traitsui.qt4.editor import Editor


class _MPLFigureEditor(Editor):

    scrollable = True

    toolbar = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.object.on_trait_change(self.update_editor, 'data_changed')
        self.set_tooltip()

    def update_editor(self):
        figure = self.value
        figure.canvas.mpl_connect('key_press_event', self.key_press_callback)
        figure.canvas.draw()

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        frame = QtGui.QWidget()
        mpl_canvas = FigureCanvas(self.value)
        mpl_canvas.setParent(frame)

        vbox = QtGui.QVBoxLayout()

        if self.toolbar:
            mpl_toolbar = NavigationToolbar2QT(mpl_canvas, frame)
            vbox.addWidget(mpl_toolbar)

        vbox.addWidget(mpl_canvas)
        frame.setLayout(vbox)

        return frame

    def key_press_callback(self, event):
        'whenever a key is pressed'
        figure = self.value
        if not event.inaxes:
            return
        if event.key == 'k':
            if figure.axes[0].get_xscale() == 'log':
                figure.axes[0].set_xscale('linear')
                figure.canvas.draw()
            else:
                figure.axes[0].set_xscale('log')
                figure.canvas.draw()

        if event.key == 'l':
            if figure.axes[0].get_yscale() == 'log':
                figure.axes[0].set_yscale('linear')
                figure.canvas.draw()
            else:
                figure.axes[0].set_yscale('log')
                figure.canvas.draw()


class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


if __name__ == "__main__":
    # Create a window to demo the editor
    from traits.api import HasTraits, Int, Float, on_trait_change
    from traitsui.api import View, Item
    from numpy import sin, cos, linspace, pi
    from traits.etsconfig.api import ETSConfig

    ETSConfig.toolkit = 'qt4'

    class Test(HasTraits):

        figure = Instance(Figure, ())
        n = Int(11)
        a = Float(0.5)

        view = View(Item('figure', editor=MPLFigureEditor(),
                         show_label=False),
                    Item('n'),
                    Item('a'),
                    width=400,
                    height=300,
                    resizable=True)

        def __init__(self):
            super(Test, self).__init__()
            axes = self.figure.add_subplot(111)
            self._t = linspace(0, 2 * pi, 200)
            self.plot()

        @on_trait_change('n,a')
        def plot(self):
            t = self._t
            a = self.a
            n = self.n
            axes = self.figure.axes[0]
            if not axes.lines:
                axes.plot(
                    sin(t) * (1 + a * cos(n * t)), cos(t) * (1 + a * cos(n * t)))
            else:
                l = axes.lines[0]
                l.set_xdata(sin(t) * (1 + a * cos(n * t)))
                l.set_ydata(cos(t) * (1 + a * cos(n * t)))
            canvas = self.figure.canvas
            if canvas is not None:
                canvas.draw()

    t = Test()
    t.configure_traits()
