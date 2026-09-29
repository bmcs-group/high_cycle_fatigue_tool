import traitsui.api as ui
import traitsui.editors
from hcft.app_icon import app_icon
from hcft.utils.busy_indicator_editor_qt import BusyIndicatorEditor
from hcft.utils.log_editor_qt import LogEditor
from hcft.utils.mpl_figure_editor_qt import MPLFigureEditor

from hcft.view.hcft_view_handler import ViewHandler, menu_exit, menu_utilities_csv_joiner, menu_about_tool

# =========================================================================
# Configuration of the views
# =========================================================================

# Conditions to deactivate the options which are not available yet or while a task is running
NOT_BUSY = 'busy == False'
FILE_LOADED = 'file_loaded == True and ' + NOT_BUSY
NPY_AVAILABLE = 'npy_available == True and ' + NOT_BUSY
FILTERED_NPY_AVAILABLE = 'filtered_npy_available == True and ' + NOT_BUSY
SMOOTHING_ACTIVE = 'activate_ascending_branch_smoothing == True or smooth == True'

import_csv_view_group = ui.VGroup(
    ui.VGroup(
        ui.Item('decimal'),
        ui.Item('delimiter'),
        ui.HGroup(
            ui.UItem('open_file_button', has_focus=True),
            ui.UItem('file_path', style='readonly', width=0.1)),
        label='Importing csv file',
        show_border=True,
        enabled_when=NOT_BUSY))

processing_csv_view_group = ui.VGroup(
                                ui.VGroup(
                                    ui.VGroup(
                                        ui.Item('take_time_from_time_column'),
                                        ui.Item('time_column',
                                                enabled_when='take_time_from_time_column == True'),
                                        ui.Item('records_per_second',
                                                visible_when='take_time_from_time_column == False'),
                                        label='Time processing',
                                        show_border=True),
                                    ui.Item('add_columns_average', label='Add cols avg / multiplier'),
                                    ui.Item('skip_first_rows'),
                                    ui.UItem('parse_csv_to_npy', resizable=True),
                                    label='Processing csv file',
                                    show_border=True,
                                    enabled_when=FILE_LOADED))

plotting_view_group = ui.VGroup(
                        ui.VGroup(
                            ui.VGroup(
                                ui.HGroup(ui.Item('x_axis'), ui.Item('x_axis_multiplier')),
                                ui.HGroup(ui.Item('y_axis'), ui.Item('y_axis_multiplier')),
                                enabled_when=FILE_LOADED),
                            ui.VGroup(
                                ui.HGroup(ui.UItem('add_plot'),
                                          ui.Item('apply_filters',
                                                  enabled_when='filtered_npy_available == True'),
                                          ui.Item('plot_settings_btn',
                                                  label='Settings',
                                                  show_label=False,
                                                  enabled_when='plot_settings_active == True'),
                                          ui.Item('plot_settings_active',
                                                  show_label=False),
                                          enabled_when=NPY_AVAILABLE
                                          ),
                                # The range is ignored for the filtered data and when the plot settings choose the rows
                                ui.Item('plot_data_range',
                                        tooltip='Number of first rows to plot,\n'
                                                'not used when "Apply filters" or the plot settings are active',
                                        enabled_when='plot_data_range_active == True and apply_filters == False'
                                                     ' and plot_settings_active == False and ' + NPY_AVAILABLE),
                                # Clearing works also for curves plotted from a previously opened file
                                ui.UItem('clear_last_plot', enabled_when=NOT_BUSY),
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
                                label='Plotting Creep-fatigue of X axis variable',
                                enabled_when=FILTERED_NPY_AVAILABLE
                            ),
                            ui.UItem('clear_plot', resizable=True, enabled_when=NOT_BUSY),
                            ui.UItem('export_plot', label='Export plot as CSV', resizable=True,
                                     enabled_when=NOT_BUSY),
                            show_border=True,
                            label='Plotting'))

filters_view_group = ui.VGroup(
                        ui.VGroup(
                            ui.Item('force_column'),
                            # Used by all filters (not only the smoothing), therefore it's always active
                            ui.Item('peak_force_before_cycles',
                                    tooltip='The cycles start at the first force exceeding this value (absolute),\n'
                                            'the rows before are the ascending branch and are all kept,\n'
                                            'from the rows after only the max and min points are kept'),
                            ui.VGroup(ui.VGroup(
                                ui.Item('activate_ascending_branch_smoothing')),
                                show_border=True,
                                label='Smooth ascending branch for all displacements:'
                            ),
                            # Active only when a smoothing is active (ascending branch or creep plot smoothing), the
                            # info button stays active
                            ui.VGroup(ui.VGroup(
                                ui.HGroup(ui.Item('window_length', enabled_when=SMOOTHING_ACTIVE),
                                          ui.UItem('smoothing_info',
                                                   tooltip='Show which options use these parameters')),
                                ui.Item('polynomial_order', enabled_when=SMOOTHING_ACTIVE)),
                                show_border=True,
                                label='Smoothing parameters (ascending branch and creep plot smoothing):'
                            ),
                            # Only the parameters of the selected cutting method are shown
                            ui.VGroup(ui.HGroup(ui.Item('cutting_method'),
                                                ui.UItem('cutting_method_info',
                                                         tooltip='Show the difference between the methods')),
                                      ui.VGroup(ui.Item('force_max'),
                                                ui.Item('force_min'),
                                                label='Max, Min:',
                                                show_border=True,
                                                visible_when='cutting_method == "Define Max, Min"'),
                                      ui.VGroup(ui.Item('min_cycle_force_range'),
                                                label='Min cycle force range:',
                                                show_border=True,
                                                visible_when='cutting_method == \
                                                                    "Define min cycle range(force difference)"'),
                                      show_border=True,
                                      label='Cut fake cycles for creep:'),
                            enabled_when=NPY_AVAILABLE),
                        ui.VSplit(
                            ui.UItem('generate_filtered_and_creep_npy', enabled_when=NPY_AVAILABLE),
                            ui.VGroup(
                                ui.UItem('busy_message', editor=BusyIndicatorEditor(), visible_when='busy == True'),
                                ui.Item('log', editor=LogEditor(),
                                        width=0.1, style='custom'),
                                ui.UItem('clear_log'),
                                ui.UItem('clear_cache', label='Clear cache (path, npy and json)',
                                         enabled_when=FILE_LOADED))),
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
                icon=app_icon,
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
