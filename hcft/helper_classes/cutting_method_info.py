import traits.api as tr
import traitsui.api as ui

from hcft.app_icon import app_icon, RESOURCES_DIR
from hcft.utils.svg_editor_qt import SVGEditor


class CuttingMethodInfo(tr.HasStrictTraits):
    # Drawing explaining the difference between the methods for cutting fake cycles
    explanation_svg = tr.Constant(str(RESOURCES_DIR / 'cutting_method_explanation.svg'))

    # =========================================================================
    # Configuration of the view
    # =========================================================================
    traits_view = ui.View(
        ui.UItem('explanation_svg', editor=SVGEditor(), resizable=True, springy=True),
        buttons=[ui.OKButton],
        title='Methods for cutting fake cycles',
        icon=app_icon,
        resizable=True
    )


if __name__ == '__main__':
    CuttingMethodInfo().configure_traits()
