import traits.api as tr
import traitsui.api as ui

from hcft.app_icon import app_icon, RESOURCES_DIR
from hcft.utils.svg_editor_qt import SVGEditor


class PlotSettings(tr.HasStrictTraits):
    num_of_first_rows_to_take = tr.Range(low=0, high=10 ** 9, value=6000, mode='spinner')
    num_of_rows_to_skip_after_each_section = tr.Range(low=0, high=10 ** 9, value=20000, mode='spinner')
    num_of_rows_in_each_section = tr.Range(
        low=0, high=10 ** 9, value=200, mode='spinner')

    # Drawing explaining the settings above
    explanation_svg = tr.Constant(str(RESOURCES_DIR / 'plot_settings_explanation.svg'))

    # =========================================================================
    # Configuration of the view
    # =========================================================================
    # Items are ordered like the rows in the data: first rows, then sections separated by skipped rows
    traits_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('num_of_first_rows_to_take',
                        label='First rows to take',
                        tooltip='Number of rows at the beginning of the data which are all plotted,\n'
                                'e.g. the initial loading up to the start of the cycles'),
                ui.Item('num_of_rows_in_each_section',
                        label='Rows in each section',
                        tooltip='Number of consecutive rows plotted in each section after the first rows,\n'
                                'e.g. enough rows for a few load cycles'),
                ui.Item('num_of_rows_to_skip_after_each_section',
                        label='Rows to skip after each section',
                        tooltip='Number of rows which are not plotted between two sections'),
                show_border=True,
                label='Rows to plot'
            ),
            ui.VGroup(
                ui.UItem('explanation_svg', editor=SVGEditor(), resizable=True, springy=True),
                show_border=True,
                label='How the rows to plot are chosen'
            )
        ),
        buttons=[ui.OKButton, ui.CancelButton],
        title='Plot settings',
        icon=app_icon,
        resizable=True
    )
