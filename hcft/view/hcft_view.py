import traitsui.api as ui
import traitsui.editors
from hcft.utils.mpl_figure_editor_qt import MPLFigureEditor

from hcft.view.hcft_view_handler import ViewHandler, menu_exit, menu_utilities_csv_joiner, menu_about_tool

# =========================================================================
# Configuration of the views
# =========================================================================

import_csv_view_group = ui.VGroup(
    ui.VGroup(
        ui.Item('decimal'),
        ui.Item('delimiter'),
        ui.HGroup(
            ui.UItem('open_file_button', has_focus=True),
            ui.UItem('file_path', style='readonly', width=0.1)),
        label='Importing csv file',
        show_border=True))

processing_csv_view_group = ui.VGroup(
                                ui.VGroup(
                                    ui.VGroup(
                                        ui.Item('take_time_from_time_column'),
                                        ui.Item('time_column',
                                                enabled_when='take_time_from_time_column == True'),
                                        ui.Item('records_per_second',
                                                enabled_when='take_time_from_time_column == False'),
                                        label='Time processing',
                                        show_border=True),
                                    ui.Item('add_columns_average', label='Add cols avg / multiplier'),
                                    ui.Item('skip_first_rows'),
                                    ui.UItem('parse_csv_to_npy', resizable=True),
                                    label='Processing csv file',
                                    show_border=True))

plotting_view_group = ui.VGroup(
                        ui.VGroup(
                            ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
                            ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
                            ui.VGroup(
                                ui.HGroup(ui.UItem('add_plot'),
                                          ui.Item('apply_filters'),
                                          ui.Item('plot_settings_btn',
                                                  label='Settings',
                                                  show_label=False,
                                                  enabled_when='plot_settings_active == True'),
                                          ui.Item('plot_settings_active',
                                                  show_label=False)
                                          ),
                                ui.Item('plot_data_range', enabled_when='plot_data_range_active == True'),
                                ui.UItem('clear_last_plot'),
                                show_border=True,
                                label='Plotting X axis with Y axis'
                            ),
                            ui.VGroup(
                                ui.HGroup(ui.UItem('add_creep_plot'),
                                          ui.VGroup(
                                              ui.Item('normalize_cycles'),
                                              ui.Item('smooth'),
                                              ui.Item('plot_every_nth_point'))
                                          ),
                                show_border=True,
                                label='Plotting Creep-fatigue of X axis variable'
                            ),
                            ui.UItem('clear_plot', resizable=True),
                            ui.UItem('export_plot', label='Export plot as CSV', resizable=True),
                            show_border=True,
                            label='Plotting'))

filters_view_group = ui.VGroup(
                        ui.Item('force_column'),
                        ui.VGroup(ui.VGroup(
                            ui.Item('window_length'),
                            ui.Item('polynomial_order'),
                            enabled_when='activate_ascending_branch_smoothing == True or smooth == True'),
                            show_border=True,
                            label='Smoothing parameters (Savitzky-Golay filter):'
                        ),
                        ui.VGroup(ui.VGroup(
                            ui.Item('activate_ascending_branch_smoothing'),
                            ui.Item('peak_force_before_cycles',
                                    enabled_when='activate_ascending_branch_smoothing == True')),
                            show_border=True,
                            label='Smooth ascending branch for all displacements:'
                        ),
                        ui.VGroup(ui.Item('cutting_method'),
                                  ui.VGroup(ui.Item('force_max'),
                                            ui.Item('force_min'),
                                            label='Max, Min:',
                                            show_border=True,
                                            enabled_when='cutting_method == "Define Max, Min"'),
                                  ui.VGroup(ui.Item('min_cycle_force_range'),
                                            label='Min cycle force range:',
                                            show_border=True,
                                            enabled_when='cutting_method == \
                                                                "Define min cycle range(force difference)"'),
                                  show_border=True,
                                  label='Cut fake cycles for creep:'),
                        ui.VSplit(
                            ui.UItem('generate_filtered_and_creep_npy'),
                            ui.VGroup(
                                ui.Item('log',
                                        width=0.1, style='custom'),
                                ui.UItem('clear_log'),
                                ui.UItem('clear_cache', label='Clear cache (path, npy and json)'))),
                        show_border=True,
                        label='Filters'
                    )

plot_figure_view = ui.UItem('figure', editor=MPLFigureEditor(),
                            resizable=True,
                            springy=True,
                            width=0.8,
                            label='2d plots')

# =========================================================================
# Configuration of the window
# =========================================================================

hcft_window = ui.View(
                ui.HSplit(
                    ui.VSplit(
                        import_csv_view_group,
                        processing_csv_view_group,
                        plotting_view_group
                    ),
                    filters_view_group,
                    plot_figure_view
                ),
                title='High-Cycle Fatigue Tool',
                resizable=True,
                width=0.9,
                height=0.9,
                scrollable=False,
                handler=ViewHandler(),
                menubar=ui.MenuBar(
                    ui.Menu(menu_exit, name='File'),
                    ui.Menu(menu_utilities_csv_joiner, name='Utilities'),
                    ui.Menu(menu_about_tool, name='Help'),
                )
            )
