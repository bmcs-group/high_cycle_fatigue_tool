import traits.api as tr
import traitsui.api as ui

from hcft.app_icon import app_icon

SMOOTHING_EXPLANATION = """\
Savitzky-Golay filter parameters, used by two options:

- "Activate" (smooth ascending branch): smooths the displacements before "Peak force before cycles" when \
generating the filtered files, generate them again after changing the parameters.

- "Smooth" (creep-fatigue plot): smooths the max and min curves when adding a creep plot.

Window length: points used for each fit (odd, bigger than the polynomial order), bigger is smoother.
Polynomial order: smaller is smoother, bigger follows the data more closely.
"""


class SmoothingInfo(tr.HasStrictTraits):
    explanation = tr.Constant(SMOOTHING_EXPLANATION)

    # =========================================================================
    # Configuration of the view
    # =========================================================================
    traits_view = ui.View(
        # A read only custom text editor wraps the long lines (the readonly style doesn't)
        ui.UItem('explanation', style='custom', editor=ui.TextEditor(read_only=True)),
        buttons=[ui.OKButton],
        title='Smoothing parameters',
        icon=app_icon,
        resizable=True,
        width=520,
        height=210
    )


if __name__ == '__main__':
    SmoothingInfo().configure_traits()
