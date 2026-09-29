"""
Created on Apr 24, 2019

@author: Homam Spartali, Rostislav Chudoba

Note: To use this tool, the input file must have the columns headers in
the first row.

"""
import json
import os
import tempfile
import traceback
from threading import Thread

import matplotlib as mpl
import numpy as np
import pandas as pd
import traits.api as tr
from pyface.api import FileDialog, GUI, MessageDialog, OK, YES, warning
from scipy.signal import savgol_filter
from pyface.confirmation_dialog import confirm

from hcft.helper_classes.columns_average import Column, ColumnsAverage
from hcft.helper_classes.csv_tools import get_headers
from hcft.helper_classes.cutting_method_info import CuttingMethodInfo
from hcft.helper_classes.files_tools import ask_to_open_saved_file
from hcft.helper_classes.plot_filtering_settings import PlotSettings
from hcft.helper_classes.smoothing_info import SmoothingInfo
from hcft.utils.plot_style import get_color
from hcft.view.hcft_view import hcft_window


# noinspection PyTypeChecker,DuplicatedCode,PyMethodMayBeStatic
class HCFT(tr.HasStrictTraits):
    """High-Cycle Fatigue Tool"""
    # =========================================================================
    # Traits definitions
    # =========================================================================
    # Assigning the view
    traits_view = hcft_window

    # CSV import
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    file_path = tr.File()
    open_file_button = tr.Button('Open file')
    columns_headers = tr.List
    npy_folder_path = tr.Str
    file_name = tr.Str

    # CSV processing
    take_time_from_time_column = tr.Bool(True)
    records_per_second = tr.Float(100)
    time_column = tr.Enum(values='columns_headers')
    skip_first_rows = tr.Range(low=1, high=10 ** 9, value=3, mode='spinner')
    add_columns_average = tr.Button
    columns_to_be_averaged = tr.List
    parse_csv_to_npy = tr.Button
    cache_folder_name = tr.Str('NPY')
    cached_path_file_name = tr.Str('HCFT_last_file_path.txt')
    # Number of csv rows parsed at once (limits the memory usage for huge files)
    csv_chunk_size = tr.Int(10 ** 6)

    # Plotting
    x_axis = tr.Enum(values='columns_headers')
    y_axis = tr.Enum(values='columns_headers')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    add_plot = tr.Button
    clear_last_plot = tr.Button
    apply_filters = tr.Bool
    plot_settings_btn = tr.Button
    plot_settings = PlotSettings()
    plot_settings_active = tr.Bool
    normalize_cycles = tr.Bool
    smooth = tr.Bool
    plot_every_nth_point = tr.Range(low=1, high=1000000, mode='spinner')
    peak_force_before_cycles = tr.Float
    add_creep_plot = tr.Button(desc='Creep plot of X axis array')
    clear_plot = tr.Button
    export_plot = tr.Button

    max_plot_data_range = tr.Int(0)
    plot_data_range = tr.Range(low=0, high='max_plot_data_range', mode='slider')
    plot_data_range_active = tr.Bool
    plot_when_plot_data_range_changes =  tr.Bool(True)

    plot_x_array = tr.Array
    plot_y_array = tr.Array

    force_column = tr.Enum(values='columns_headers')
    window_length = tr.Range(low=1, high=10 ** 9 - 1, value=31, mode='spinner')
    polynomial_order = tr.Range(low=1, high=10 ** 9, value=2, mode='spinner')
    activate_ascending_branch_smoothing = tr.Bool(False, label='Activate')

    generate_filtered_and_creep_npy = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)
    min_cycle_force_range = tr.Float(50)
    cutting_method = tr.Enum('Define min cycle range(force difference)', 'Define Max, Min')
    cutting_method_info = tr.Button('Info')
    # Reference to the open info window, otherwise it gets garbage collected and closes
    _cutting_method_info_ui = tr.Any
    smoothing_info = tr.Button('Info')
    _smoothing_info_ui = tr.Any

    log = tr.Str('')
    clear_log = tr.Button
    clear_cache = tr.Button

    # State of the current file, used to deactivate the options which are not available yet
    file_loaded = tr.Bool(False)
    npy_available = tr.Bool(False)
    filtered_npy_available = tr.Bool(False)
    # Description of the task running on a different thread, empty when no task is running
    busy_message = tr.Str('')
    busy = tr.Property(tr.Bool, depends_on='busy_message')

    def _get_busy(self):
        return self.busy_message != ''

    # =========================================================================
    # Assigning default values
    # =========================================================================
    ax = tr.Any
    figure = tr.Instance(mpl.figure.Figure)

    def _figure_default(self):
        figure = mpl.figure.Figure(facecolor='white', layout='tight')
        self.create_axes(figure)
        return figure

    def create_axes(self, figure):
        self.ax = figure.add_subplot(1, 1, 1)

    # =========================================================================
    # File management
    # =========================================================================
    def _open_file_button_fired(self):
        try:
            self.reset()

            if self.file_path == '':
                cached_path = self._get_cached_file_path_from_system_tmp()
                default_path = os.path.expanduser("~") if cached_path == '' else cached_path
            else:
                default_path = self.file_path

            dialog = FileDialog(title='Select text file', action='open', default_path=default_path)
            result = dialog.open()

            # Test if the user opened a file to avoid throwing an exception if he doesn't

            if result == OK:
                self.file_path = dialog.path
                self._cache_file_path_in_system_tmp(self.file_path)
            else:
                return
            self.file_loaded = False
            # Populate headers list which fills the x-axis and y-axis with values automatically
            self.columns_headers = get_headers(self.file_path, decimal=self.decimal, delimiter=self.delimiter)

            # Saving file name and path and creating NPY folder
            dir_path = os.path.dirname(self.file_path)
            self.npy_folder_path = os.path.join(dir_path, self.cache_folder_name)
            if not os.path.exists(self.npy_folder_path):
                os.makedirs(self.npy_folder_path)

            self.file_name = os.path.splitext(os.path.basename(self.file_path))[0]

            self.import_data_json()
            self.file_loaded = True

        except:
            self.log_exception()
        finally:
            self.update_npy_availability()

    @tr.observe('columns_headers.items, force_column')
    def _update_npy_availability_on_columns_change(self, event):
        # e.g. a newly added columns average doesn't have an npy file until the csv is parsed again
        self.update_npy_availability()

    def update_npy_availability(self):
        self.npy_available = self.file_loaded and all(
            os.path.exists(self.get_npy_file_path(column_name)) for column_name in self.columns_headers)
        self.filtered_npy_available = (self.npy_available
                                       and os.path.exists(self.get_filtered_npy_file_path(self.force_column))
                                       and os.path.exists(self.get_max_npy_file_path(self.force_column)))
        if not self.filtered_npy_available:
            self.apply_filters = False

    def run_in_thread(self, target, busy_message):
        # Run method on different thread so GUI doesn't freeze, the options are deactivated while it's busy
        if self.busy:
            return
        self.busy_message = busy_message

        def run():
            try:
                target()
            finally:
                self.busy_message = ''

        Thread(target=run).start()

    def _cache_file_path_in_system_tmp(self, path):
        try:
            cache_file_path = os.path.join(tempfile.gettempdir(), self.cached_path_file_name)
            with open(cache_file_path, "w") as f:
                f.write(path)
        except:
            self.print_custom('Caching file path failed. This has no effect on the results.')

    def _get_cached_file_path_from_system_tmp(self):
        cache_file_path = os.path.join(tempfile.gettempdir(), self.cached_path_file_name)
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as f:
                cached_path = f.read().strip()
                if os.path.exists(cached_path):
                    return cached_path
        return ''

    def _add_columns_average_fired(self):
        try:
            columns_average = ColumnsAverage()
            for name in self.columns_headers:
                columns_average.columns.append(Column(column_name=name))

            # kind='modal' pauses the implementation until the window is closed
            columns_average.configure_traits(kind='modal')

            columns_to_be_averaged_temp = []
            for col in columns_average.columns:
                if col.selected:
                    columns_to_be_averaged_temp.append(col)

            if columns_to_be_averaged_temp:  # If it's not empty
                self.columns_to_be_averaged.append(columns_to_be_averaged_temp)

                avg_file_suffix = self.get_suffix_for_columns_to_be_averaged(columns_to_be_averaged_temp)
                self.columns_headers.append(avg_file_suffix)
        except:
            self.log_exception()

    def _parse_csv_to_npy_fired(self):
        self.run_in_thread(self.parse_csv_to_npy_fired, 'Parsing csv...')

    def parse_csv_to_npy_fired(self):
        try:
            self.print_custom('Parsing csv into npy files...')

            """ Exporting npy arrays of original columns """
            # The csv file is read only once in chunks and the values of each column are appended to a temporary
            # binary file. In this way, only one chunk is kept in memory, which enables parsing huge files.
            num_of_original_columns = len(self.columns_headers) - len(self.columns_to_be_averaged)
            original_columns = self.columns_headers[:num_of_original_columns]
            tmp_paths = [self.get_npy_file_path(column_name) + '.tmp' for column_name in original_columns]
            tmp_files = [open(tmp_path, 'wb') for tmp_path in tmp_paths]
            try:
                # One could provide the path directly to pd.read_csv but in this way we insure that this works also if
                # the path to the file include chars like ü,ä
                # (with) makes sure the file stream is closed after using it
                with open(self.file_path, encoding='latin-1') as file_stream:
                    # header=None, because the headers row is already counted in skip_first_rows
                    reader = pd.read_csv(file_stream, delimiter=self.delimiter, decimal=self.decimal,
                                         skiprows=self.skip_first_rows, header=None,
                                         usecols=range(num_of_original_columns), chunksize=self.csv_chunk_size)
                    non_numeric_columns = set()
                    for chunk in reader:
                        for i, tmp_file in enumerate(tmp_files):
                            column = chunk[i]
                            if not pd.api.types.is_numeric_dtype(column):
                                column = self.to_numeric(column)
                                non_numeric_columns.add(original_columns[i])
                            column.to_numpy(dtype=np.float64).tofile(tmp_file)
                    for column_name in non_numeric_columns:
                        self.print_custom('Warning: non-numeric values in column "', column_name,
                                          '" are replaced by nan.')
            finally:
                for tmp_file in tmp_files:
                    tmp_file.close()

            for column_name, tmp_path in zip(original_columns, tmp_paths):
                column_array = np.fromfile(tmp_path, dtype=np.float64)
                os.remove(tmp_path)
                if column_name == self.time_column and self.take_time_from_time_column is False:
                    column_array = np.arange(column_array.size) / self.records_per_second
                np.save(self.get_npy_file_path(column_name), column_array)

            self.max_plot_data_range = column_array.size
            self.plot_when_plot_data_range_changes = False
            self.plot_data_range = self.max_plot_data_range
            self.plot_data_range_active = True

            """ Exporting npy arrays of averaged columns """
            for cols in self.columns_to_be_averaged:
                avg = sum(col.multi_factor * np.load(self.get_npy_file_path(col.column_name)).flatten()
                          for col in cols) / len(cols)
                np.save(self.get_average_npy_file_path(cols), avg)

            self.export_data_json()
            self.print_custom('Finished parsing csv into npy files.')
        except:
            self.log_exception()
        finally:
            self.update_npy_availability()

    def to_numeric(self, column):
        # A column with some non-numeric rows (e.g., repeated headers in joined files) is not converted by
        # pd.read_csv, therefore it's converted here and the non-numeric values are replaced by nan
        return pd.to_numeric(column.astype(str).str.replace(self.decimal, '.', regex=False), errors='coerce')

    def get_npy_file_path(self, column_name):
        return os.path.join(self.npy_folder_path, self.file_name + '_' + column_name + '.npy')

    def get_filtered_npy_file_path(self, column_name):
        return os.path.join(self.npy_folder_path, self.file_name + '_' + column_name + '_filtered.npy')

    def get_max_npy_file_path(self, column_name):
        return os.path.join(self.npy_folder_path, self.file_name + '_' + column_name + '_max.npy')

    def get_min_npy_file_path(self, column_name):
        return os.path.join(self.npy_folder_path, self.file_name + '_' + column_name + '_min.npy')

    def get_average_npy_file_path(self, columns_names):
        avg_file_suffix = self.get_suffix_for_columns_to_be_averaged(columns_names)
        return os.path.join(self.npy_folder_path, self.file_name + '_' + avg_file_suffix + '.npy')

    def get_suffix_for_columns_to_be_averaged(self, columns):
        names_list = []
        for col in columns:
            names_list.append(col.column_name)
        suffix_for_saved_file_name = 'avg_' + '_'.join(names_list)
        return suffix_for_saved_file_name

    def _get_columns_to_be_averaged_json(self):
        list_of_cols_list_json = []
        for column_list in self.columns_to_be_averaged:
            cols_list_json = []
            for col in column_list:
                col_json = {'column_name': col.column_name,
                               'selected': col.selected,
                               'multi_factor': col.multi_factor}
                cols_list_json.append(col_json)
            list_of_cols_list_json.append(cols_list_json)
        return list_of_cols_list_json

    def _assign_columns_to_be_averaged_from_json(self, list_of_cols_list_json):
        self.columns_to_be_averaged = []
        for cols_list_json in list_of_cols_list_json:
            cols_list = []
            for col_json in cols_list_json:
                col = Column(column_name=col_json['column_name'],
                                selected=col_json['selected'],
                                multi_factor=col_json['multi_factor'])
                cols_list.append(col)
            self.columns_to_be_averaged.append(cols_list)

    def export_data_json(self):
        # Output data MUST have exactly similar keys and variable names
        output_data = {'take_time_from_time_column': self.take_time_from_time_column,
                       'time_column': self.time_column,
                       'records_per_second': self.records_per_second,
                       'skip_first_rows': self.skip_first_rows,
                       'columns_headers': self.columns_headers,
                       'columns_to_be_averaged_json': self._get_columns_to_be_averaged_json(),
                       'x_axis': self.x_axis,
                       'y_axis': self.y_axis,
                       'x_axis_multiplier': self.x_axis_multiplier,
                       'y_axis_multiplier': self.y_axis_multiplier,
                       'force_column': self.force_column,
                       'window_length': self.window_length,
                       'polynomial_order': self.polynomial_order,
                       'peak_force_before_cycles': self.peak_force_before_cycles,
                       'cutting_method': self.cutting_method,
                       'force_max': self.force_max,
                       'force_min': self.force_min,
                       'min_cycle_force_range': self.min_cycle_force_range,
                       'max_plot_data_range': self.max_plot_data_range,
                       'plot_data_range' : self.plot_data_range,
                       'plot_data_range_active' : self.plot_data_range_active
                       }
        with open(self.get_json_file_path(), 'w') as outfile:
            json.dump(output_data, outfile, sort_keys=True, indent=4)
        self.print_custom('.json data file exported successfully.')

    def import_data_json(self):
        self.plot_when_plot_data_range_changes = False
        json_path = self.get_json_file_path()
        if not os.path.isfile(json_path):
            return
        # class_vars is a list with class variables names
        # vars(self) & self.__dict__.items() didn't include some Trait variables like force_column = tr.Enum(values=..
        class_vars = [attr for attr in dir(self) if not attr.startswith("_")]
        with open(json_path) as infile:
            data_in = json.load(infile)
        for key_data, value_data in data_in.items():
            # Backward compatibility for columns_to_be_averaged
            if key_data == 'columns_to_be_averaged':
                self.columns_to_be_averaged = []
                for cols_names_list in value_data:
                    cols_avg = []
                    for col_name in cols_names_list:
                        cols_avg.append(Column(column_name=col_name))
                    self.columns_to_be_averaged.append(cols_avg)
            elif key_data == 'columns_to_be_averaged_json':
                self._assign_columns_to_be_averaged_from_json(value_data)
            elif key_data in class_vars:
                # Equivalent to: self.key_data = value_data
                setattr(self, key_data, value_data)
        self.print_custom('.json data file imported successfully.')

    def get_json_file_path(self):
        return os.path.join(self.npy_folder_path, self.file_name + '.json')

    def _generate_filtered_and_creep_npy_fired(self):
        self.run_in_thread(self.generate_filtered_and_creep_npy_fired, 'Generating filtered and creep files...')

    def generate_filtered_and_creep_npy_fired(self):
        try:
            self.export_data_json()
            if not self.npy_files_exist(self.get_npy_file_path(self.force_column)):
                return
            self.print_custom('Generating filtered and creep files...')

            # 1- Export filtered force
            force = np.load(self.get_npy_file_path(self.force_column)).flatten()
            # The cycles start at the first force exceeding the peak force before cycles, the rows before are the
            # ascending branch
            exceeding_indices = np.flatnonzero(np.abs(force) > abs(self.peak_force_before_cycles))
            if exceeding_indices.size == 0:
                self.warn_user('The force never exceeds "Peak force before cycles" (', self.peak_force_before_cycles,
                               '), please choose a smaller value!')
                return
            peak_force_before_cycles_index = exceeding_indices[0]
            # The Savitzky-Golay filter needs at least window_length points
            if self.activate_ascending_branch_smoothing and peak_force_before_cycles_index < self.window_length:
                self.warn_user('The ascending branch has only ', peak_force_before_cycles_index,
                               ' rows, which is less than the smoothing window length (', self.window_length,
                               '). Please increase "Peak force before cycles" (', self.peak_force_before_cycles,
                               '), reduce the window length or deactivate the ascending branch smoothing.')
                return
            force_ascending = force[0:peak_force_before_cycles_index]
            force_rest = force[peak_force_before_cycles_index:]

            force_max_indices, force_min_indices = self.get_array_max_and_min_indices(force_rest)

            force_max_min_indices = np.concatenate((force_min_indices, force_max_indices))
            force_max_min_indices.sort()

            force_rest_filtered = force_rest[force_max_min_indices]
            force_filtered = np.concatenate((force_ascending, force_rest_filtered))
            np.save(self.get_filtered_npy_file_path(self.force_column), force_filtered)

            # 2- Export filtered displacements
            # Export displacements combining processed ascending branch and unprocessed min/max values
            self.export_filtered_displacements(force_max_min_indices, peak_force_before_cycles_index)

            # 3- Export creep for displacements
            # Cut unwanted max min values to get correct full cycles and remove false min/max values caused by noise
            self.export_displacements_creep(force_rest, force_max_indices, force_min_indices,
                                            peak_force_before_cycles_index)

            self.print_custom('Filtered and creep npy files are generated.')
        except:
            self.log_exception()
        finally:
            self.update_npy_availability()

    def export_filtered_displacements(self, force_max_min_indices, peak_force_before_cycles_index):
        for i in range(len(self.columns_headers)):
            if self.columns_headers[i] != self.force_column and self.columns_headers[i] != self.time_column:

                disp = np.load(self.get_npy_file_path(self.columns_headers[i])).flatten()
                disp_ascending = disp[0:peak_force_before_cycles_index]
                disp_rest = disp[peak_force_before_cycles_index:]

                if self.activate_ascending_branch_smoothing:
                    disp_ascending = savgol_filter(disp_ascending, window_length=self.window_length,
                                                   polyorder=self.polynomial_order)

                disp_rest_filtered = disp_rest[force_max_min_indices]
                filtered_disp = np.concatenate((disp_ascending, disp_rest_filtered))
                np.save(self.get_filtered_npy_file_path(self.columns_headers[i]), filtered_disp)

    def export_displacements_creep(self, force_rest, force_max_indices, force_min_indices,
                                   peak_force_before_cycles_index):
        if self.cutting_method == "Define Max, Min":
            force_max_indices_cut, force_min_indices_cut = self.cut_indices_of_min_max_range(force_rest,
                                                                                             force_max_indices,
                                                                                             force_min_indices,
                                                                                             self.force_max,
                                                                                             self.force_min)
        elif self.cutting_method == "Define min cycle range(force difference)":
            force_max_indices_cut, force_min_indices_cut = self.cut_indices_of_defined_range(force_rest,
                                                                                             force_max_indices,
                                                                                             force_min_indices,
                                                                                             self.min_cycle_force_range)
        self.print_custom("Cycles number= ", len(force_min_indices))
        self.print_custom("Cycles number after cutting fake cycles = ", len(force_min_indices_cut))

        for i in range(len(self.columns_headers)):
            if self.columns_headers[i] != self.time_column:
                array = np.load(self.get_npy_file_path(self.columns_headers[i])).flatten()
                array_rest = array[peak_force_before_cycles_index:]
                array_rest_maxima = array_rest[force_max_indices_cut]
                array_rest_minima = array_rest[force_min_indices_cut]
                np.save(self.get_max_npy_file_path(self.columns_headers[i]), array_rest_maxima)
                np.save(self.get_min_npy_file_path(self.columns_headers[i]), array_rest_minima)

    def get_array_max_and_min_indices(self, input_array):
        # Checking dominant sign
        positive_values_count = np.sum(np.array(input_array) >= 0)
        negative_values_count = input_array.size - positive_values_count

        # Getting max and min indices
        if positive_values_count > negative_values_count:
            force_max_indices = self.get_max_indices(input_array)
            force_min_indices = self.get_min_indices(input_array)
        else:
            force_max_indices = self.get_min_indices(input_array)
            force_min_indices = self.get_max_indices(input_array)

        return force_max_indices, force_min_indices

    def get_plateaus_last_indices(self, a):
        # Consecutive repeated values (plateaus) are treated as one point represented by its last index
        if a.size == 0:
            return np.array([], dtype=np.intp)
        return np.append(np.flatnonzero(a[1:] != a[:-1]), a.size - 1)

    def get_max_indices(self, a):
        # Vectorized detection of local maxima, a plateau is a max if both of its neighbors are smaller.
        # This method doesn't qualify first and last elements as max
        indices = self.get_plateaus_last_indices(a)
        values = a[indices]
        is_max = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
        return indices[1:-1][is_max]

    def get_min_indices(self, a):
        # Vectorized detection of local minima, a plateau is a min if both of its neighbors are bigger.
        # This method doesn't qualify first and last elements as min
        indices = self.get_plateaus_last_indices(a)
        values = a[indices]
        is_min = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
        return indices[1:-1][is_min]

    def cut_indices_of_min_max_range(self, array, max_indices, min_indices,
                                     range_upper_value, range_lower_value):
        cut_max_indices = max_indices[np.abs(array[max_indices]) > abs(range_upper_value)]
        cut_min_indices = min_indices[np.abs(array[min_indices]) < abs(range_lower_value)]
        return cut_max_indices, cut_min_indices

    def cut_indices_of_defined_range(self, array, max_indices, min_indices, range_):
        # Each max is paired with the min of the same order, only pairs with bigger force difference than range_ are
        # kept
        n = min(max_indices.size, min_indices.size)
        is_full_cycle = np.abs(array[max_indices[:n]] - array[min_indices[:n]]) > range_
        cut_max_indices = max_indices[:n][is_full_cycle]
        cut_min_indices = min_indices[:n][is_full_cycle]

        if max_indices.size > min_indices.size:
            cut_max_indices = np.append(cut_max_indices, max_indices[-1])
        elif min_indices.size > max_indices.size:
            cut_min_indices = np.append(cut_min_indices, min_indices[-1])

        return cut_max_indices, cut_min_indices

    def _window_length_changed(self, new):
        if new <= self.polynomial_order:
            dialog = MessageDialog(
                title='Attention!',
                message='Window length must be bigger than polynomial order.')
            dialog.open()

        if new % 2 == 0 or new <= 0:
            dialog = MessageDialog(
                title='Attention!',
                message='Window length must be odd positive integer.')
            dialog.open()

    def _polynomial_order_changed(self, new):
        if new >= self.window_length:
            dialog = MessageDialog(
                title='Attention!',
                message='Polynomial order must be smaller than window length.')
            dialog.open()

    def _plot_data_range_changed(self):
        if self.plot_when_plot_data_range_changes:
            # The last curve is plotted again with the new range, keeping its color
            color = self.ax.lines[-1].get_color() if len(self.ax.lines) != 0 else None
            self._clear_last_plotted_curve()
            self.add_plot_fired(color=color)
        else:
            self.plot_when_plot_data_range_changes = True

    # =========================================================================
    # Plotting
    # =========================================================================
    data_changed = tr.Event

    def _plot_settings_btn_fired(self):
        try:
            self.plot_settings.configure_traits(kind='modal')
        except:
            self.log_exception()

    def _cutting_method_info_fired(self):
        try:
            self._cutting_method_info_ui = self.show_info_window(self._cutting_method_info_ui, CuttingMethodInfo)
        except:
            self.log_exception()

    def _smoothing_info_fired(self):
        try:
            self._smoothing_info_ui = self.show_info_window(self._smoothing_info_ui, SmoothingInfo)
        except:
            self.log_exception()

    def show_info_window(self, ui, info_class):
        if ui is not None and ui.control is not None:
            # Already open, bring it to the front instead of opening another one
            ui.control.raise_()
            ui.control.activateWindow()
            return ui
        # Non-modal, so it can stay open while choosing the options
        return info_class().edit_traits(kind='live')

    def npy_files_exist(self, path):
        if os.path.exists(path):
            return True
        else:
            self.warn_user('Please parse csv file to generate npy files first!')
            return False

    def filtered_and_creep_npy_files_exist(self, path):
        if os.path.exists(path):
            return True
        else:
            self.warn_user('Please generate filtered and creep npy files first!')
            return False

    def _clear_last_plotted_curve(self):
        if len(self.ax.lines) != 0:
            self.ax.lines[-1].remove()
            # Refresh the legend
            self.ax.legend()
            # Reset the axes limits
            self.ax.relim()
            # Update figure
            self.data_changed = True

    def _clear_last_plot_fired(self):
        self._clear_last_plotted_curve()

    def _add_plot_fired(self):
        self.run_in_thread(self.add_plot_fired, 'Adding plot...')

    def add_plot_fired(self, color=None):
        try:
            if self.apply_filters:
                if not self.filtered_and_creep_npy_files_exist(self.get_filtered_npy_file_path(self.x_axis)):
                    return
                # TODO link this _filtered to the path creation function
                x_axis_name = self.x_axis + '_filtered'
                y_axis_name = self.y_axis + '_filtered'
                self.print_custom('Loading npy files...')
                # when mmap_mode!=None, the array will be loaded as 'numpy.memmap'
                # object which doesn't load the array to memory until it's
                # indexed
                x_axis_array = np.load(self.get_filtered_npy_file_path(self.x_axis), mmap_mode='r')
                y_axis_array = np.load(self.get_filtered_npy_file_path(self.y_axis), mmap_mode='r')
            else:
                if not self.npy_files_exist(self.get_npy_file_path(self.x_axis)):
                    return

                x_axis_name = self.x_axis
                y_axis_name = self.y_axis
                self.print_custom('Loading npy files...')
                # when mmap_mode!=None, the array will be loaded as 'numpy.memmap'
                # object which doesn't load the array to memory until it's
                # indexed
                x_axis_array = np.load(self.get_npy_file_path(self.x_axis), mmap_mode='r')
                y_axis_array = np.load(self.get_npy_file_path(self.y_axis), mmap_mode='r')

            # Only the rows needed for the plot are indexed from the memmap arrays and therefore read from the disk
            if self.plot_settings_active:
                indices = self.get_indices_array(len(x_axis_array),
                                                 self.plot_settings.num_of_first_rows_to_take,
                                                 self.plot_settings.num_of_rows_to_skip_after_each_section,
                                                 self.plot_settings.num_of_rows_in_each_section)
                x_axis_array = x_axis_array[indices]
                y_axis_array = y_axis_array[indices]
            elif not self.apply_filters and self.plot_data_range_active:
                x_axis_array = x_axis_array[:self.plot_data_range]
                y_axis_array = y_axis_array[:self.plot_data_range]

            x_axis_array = self.x_axis_multiplier * x_axis_array
            y_axis_array = self.y_axis_multiplier * y_axis_array

            self.print_custom('Adding Plot...')
            mpl.rcParams['agg.path.chunksize'] = 10000
            ax = self.ax

            ax.set_xlabel(x_axis_name[:60])
            ax.set_ylabel(y_axis_name[:60])

            curve_label = self.file_name + ', ' + x_axis_name
            ax.plot(x_axis_array, y_axis_array, linewidth=1.2, color=get_color() if color is None else color,
                    label=curve_label)
            ax.legend(prop={'size': 14 if len(curve_label) < 50 else 10.5})

            self.data_changed = True
            self.print_custom('Finished adding plot.')

        except:
            self.log_exception()

    def _add_creep_plot_fired(self):
        self.run_in_thread(self.add_creep_plot_fired, 'Adding creep-fatigue plot...')

    def add_creep_plot_fired(self):
        try:
            if not self.filtered_and_creep_npy_files_exist(self.get_max_npy_file_path(self.x_axis)):
                return

            self.print_custom('Loading npy files...')
            disp_max = self.x_axis_multiplier * np.load(self.get_max_npy_file_path(self.x_axis))
            disp_min = self.x_axis_multiplier * np.load(self.get_min_npy_file_path(self.x_axis))
            complete_cycles_number = disp_max.size

            self.print_custom('Adding creep-fatigue plot...')
            mpl.rcParams['agg.path.chunksize'] = 10000

            if self.plot_every_nth_point > 1:
                disp_max = disp_max[0::self.plot_every_nth_point]
                disp_min = disp_min[0::self.plot_every_nth_point]

            if self.smooth:
                # The Savitzky-Golay filter needs at least window_length points
                if min(disp_max.size, disp_min.size) - 1 < self.window_length:
                    self.warn_user('Only ', min(disp_max.size, disp_min.size), ' cycles to plot, which is less '
                                   'than the smoothing window length (', self.window_length, '). Please reduce '
                                   'the window length or "Plot every nth point", or deactivate "Smooth".')
                    return
                # Keeping the first item of the array and filtering the rest
                disp_max = np.concatenate((
                    np.array([disp_max[0]]),
                    savgol_filter(disp_max[1:], window_length=self.window_length, polyorder=self.polynomial_order)
                ))
                disp_min = np.concatenate((
                    np.array([disp_min[0]]),
                    savgol_filter(disp_min[1:], window_length=self.window_length, polyorder=self.polynomial_order)
                ))

            ax = self.ax
            ax.set_xlabel('Cycles number')
            ax.set_ylabel(self.x_axis)

            cycles_end = 1. if self.normalize_cycles else complete_cycles_number
            ax.plot(np.linspace(0, cycles_end, disp_max.size), disp_max, linewidth=1.2, color=get_color(),
                    label='Max, ' + self.file_name + ', ' + self.x_axis)
            ax.plot(np.linspace(0, cycles_end, disp_min.size), disp_min, linewidth=1.2, color=get_color(),
                    label='Min, ' + self.file_name + ', ' + self.x_axis)

            ax.legend()
            self.data_changed = True
            self.print_custom('Finished adding creep-fatigue plot.')

        except:
            self.log_exception()

    def get_indices_array(self,
                          array_size,
                          first_rows,
                          distance,
                          num_of_rows_after_each_distance):
        # Indices of the first rows followed by sections with (num_of_rows_after_each_distance) rows, which are
        # separated by (distance) skipped rows
        result_1 = np.arange(min(first_rows, array_size))
        sections_starts = np.arange(start=first_rows, stop=array_size,
                                    step=max(distance + num_of_rows_after_each_distance, 1))
        result_2 = (sections_starts[:, np.newaxis] + np.arange(num_of_rows_after_each_distance)).ravel()
        # The last section might exceed the array size
        result_2 = result_2[result_2 < array_size]
        return np.concatenate((result_1, result_2))

    def _clear_plot_fired(self):
        self.figure.clear()
        self.create_axes(self.figure)
        self.data_changed = True

    def _export_plot_fired(self):
        if len(self.ax.lines) == 0:
            self.warn_user('The plot has no curves to export!')
            return

        x_label = self.ax.get_xlabel()
        y_label = self.ax.get_ylabel()
        df = pd.DataFrame()

        max_data_length = 0
        for i, line in enumerate(self.ax.lines):
            max_data_length = max(len(line.get_xdata()), max_data_length)

        for i, line in enumerate(self.ax.lines):
            x_vals = np.asarray(line.get_xdata(), dtype=np.float64).flatten()
            y_vals = np.asarray(line.get_ydata(), dtype=np.float64).flatten()

            line_data_len_diff = max_data_length - len(x_vals)
            if line_data_len_diff != 0:
                x_vals = np.pad(x_vals, (0, line_data_len_diff), constant_values=np.nan)
                y_vals = np.pad(y_vals, (0, line_data_len_diff), constant_values=np.nan)

            curve_label = line.get_label()
            x_header = str(i * 2) + '_' + curve_label + '_' + x_label
            y_header = str(i * 2 + 1) + '_' + curve_label + '_' + y_label
            df[x_header] = x_vals
            df[y_header] = y_vals

        # A new file name is suggested, so the input file isn't overwritten by mistake
        if self.file_loaded:
            default_path = os.path.join(os.path.dirname(self.file_path), self.file_name + '_plot.csv')
        else:
            default_path = os.path.join(os.path.expanduser('~'), 'plot.csv')
        dialog = FileDialog(title='Save plot as CSV file', action='save as', default_path=default_path,
                            wildcard=FileDialog.create_wildcard('CSV files', '*.csv'))
        if dialog.open() == OK:
            file_path = dialog.path
            df.to_csv(file_path, decimal=self.decimal, sep=self.delimiter, index=False)
            self.print_custom('Plot exported to "', file_path, '"')
            ask_to_open_saved_file(file_path, title='Plot exported')


    # =========================================================================
    # Logging
    # =========================================================================
    def print_custom(self, *input_args):
        print(*input_args)
        if self.log == '':
            self.log = ''.join(str(e) for e in list(input_args))
        else:
            self.log = self.log + '\n' + \
                       ''.join(str(e) for e in list(input_args))

    def warn_user(self, *input_args):
        # For actions which can't be done, the message is logged and shown in a dialog. The dialog is opened in the
        # GUI thread, because this can be called from the threads running the tasks
        self.print_custom(*input_args)
        message = ''.join(str(e) for e in input_args)
        GUI.invoke_later(warning, None, message, 'Attention!')

    def log_exception(self):
        self.print_custom('SOMETHING WENT WRONG!')
        self.print_custom('--------- Error message: ---------')
        self.print_custom(traceback.format_exc())
        self.print_custom('----------------------------------')

    def _clear_log_fired(self):
        self.log = ''

    def _clear_cache_fired(self):
        confirmed = confirm(
            parent=None,
            message="Saved settings and the processed *.npy files will be deleted. Are you sure?",
            title="Confirm Deletion",
        )

        if confirmed == YES:
        # dialog = MessageDialog(
        #     title="Confirm Deletion",
        #     message="Saved settings and the processed *.npy files will be deleted. Are you sure?",
        #     buttons=["Yes", "No"],
        # )
        # result = dialog.open()
        # if result == 'Yes':
            deleted_files = []
            cached_file_path = os.path.join(tempfile.gettempdir(), self.cached_path_file_name)
            if os.path.exists(cached_file_path):
                os.remove(cached_file_path)
                deleted_files.append(cached_file_path)

            if os.path.exists(self.npy_folder_path):
                files = os.listdir(self.npy_folder_path)
                for file in files:
                    if file.startswith(self.file_name):
                        file_path = os.path.join(self.npy_folder_path, file)
                        os.remove(file_path)
                        deleted_files.append(file)
                self.print_custom('---------------------')
                self.print_custom('Cache cleared successfully.')
                self.print_custom('Following files are deleted:')
                for deleted_file in deleted_files:
                    self.print_custom('-  ' + deleted_file)
            else:
                self.print_custom(f"Directory '{self.npy_folder_path}' does not exist.")
            self.update_npy_availability()

    # =========================================================================
    # Other functions
    # =========================================================================
    def reset(self):
        self.plot_data_range_active = False
        self.columns_to_be_averaged = []
        self.log = ''

if __name__ == '__main__':
    hcft = HCFT(file_path=os.path.expanduser("~"))
    hcft.configure_traits()
