"""
Created on Apr 24, 2019

@author: Homam Spartali, Rostislav Chudoba

Note: To use this tool, the input file must have the columns headers in
the first row.

"""
import json
import os
import traceback
from threading import Thread

import matplotlib as mpl
import numpy as np
import pandas as pd
import traits.api as tr
from pyface.api import FileDialog, MessageDialog, OK
from scipy.signal import savgol_filter

from helper_classes.columns_average import Column, ColumnsAverage
from helper_classes.csv_tools import get_headers
from helper_classes.plot_filtering_settings import PlotSettings
from view.hcft_view import hcft_window


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
    file_path = tr.File
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

    # Plotting
    x_axis = tr.Enum(values='columns_headers')
    y_axis = tr.Enum(values='columns_headers')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    add_plot = tr.Button
    apply_filters = tr.Bool
    plot_settings_btn = tr.Button
    plot_settings = PlotSettings()
    plot_settings_active = tr.Bool
    normalize_cycles = tr.Bool
    smooth = tr.Bool
    plot_every_nth_point = tr.Range(low=1, high=1000000, mode='spinner')
    old_peak_force_before_cycles = tr.Float
    peak_force_before_cycles = tr.Float
    add_creep_plot = tr.Button(desc='Creep plot of X axis array')
    clear_plot = tr.Button

    force_column = tr.Enum(values='columns_headers')
    window_length = tr.Range(low=1, high=10 ** 9 - 1, value=31, mode='spinner')
    polynomial_order = tr.Range(low=1, high=10 ** 9, value=2, mode='spinner')
    activate_ascending_branch_smoothing = tr.Bool(False, label='Activate')

    generate_filtered_and_creep_npy = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)
    min_cycle_force_range = tr.Float(50)
    cutting_method = tr.Enum('Define min cycle range(force difference)', 'Define Max, Min')

    log = tr.Str('')
    clear_log = tr.Button

    # =========================================================================
    # Assigning default values
    # =========================================================================
    figure = tr.Instance(mpl.figure.Figure)

    def _figure_default(self):
        figure = mpl.figure.Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    # =========================================================================
    # File management
    # =========================================================================
    def _open_file_button_fired(self):
        try:
            self.reset()

            extensions = ['*.csv', '*.txt']  # handle only one extension...
            wildcard = ';'.join(extensions)
            dialog = FileDialog(title='Select text file',
                                action='open', wildcard=wildcard,
                                default_path=self.file_path)
            result = dialog.open()

            # Test if the user opened a file to avoid throwing an exception if he doesn't
            if result == OK:
                self.file_path = dialog.path
            else:
                return

            # Populate headers list which fills the x-axis and y-axis with values automatically
            self.columns_headers = get_headers(self.file_path, decimal=self.decimal, delimiter=self.delimiter)

            # Saving file name and path and creating NPY folder
            dir_path = os.path.dirname(self.file_path)
            self.npy_folder_path = os.path.join(dir_path, 'NPY')
            if not os.path.exists(self.npy_folder_path):
                os.makedirs(self.npy_folder_path)

            self.file_name = os.path.splitext(os.path.basename(self.file_path))[0]

            self.import_data_json()

        except:
            self.log_exception()

    def _add_columns_average_fired(self):
        try:
            columns_average = ColumnsAverage()
            for name in self.columns_headers:
                columns_average.columns.append(Column(column_name=name))

            # kind='modal' pauses the implementation until the window is closed
            columns_average.configure_traits(kind='modal')

            columns_to_be_averaged_temp = []
            for i in columns_average.columns:
                if i.selected:
                    columns_to_be_averaged_temp.append(i.column_name)

            if columns_to_be_averaged_temp:  # If it's not empty
                self.columns_to_be_averaged.append(columns_to_be_averaged_temp)

                avg_file_suffix = self.get_suffix_for_columns_to_be_averaged(columns_to_be_averaged_temp)
                self.columns_headers.append(avg_file_suffix)
        except:
            self.log_exception()

    def _parse_csv_to_npy_fired(self):
        # Run method on different thread so GUI doesn't freeze
        # thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.parse_csv_to_npy_fired)
        thread.start()

    def parse_csv_to_npy_fired(self):
        try:
            self.print_custom('Parsing csv into npy files...')
            self.export_data_json()

            """ Exporting npy arrays of original columns """
            for i in range(len(self.columns_headers) - len(self.columns_to_be_averaged)):
                column_name = self.columns_headers[i]
                column_array = np.array(pd.read_csv(self.file_path, delimiter=self.delimiter, decimal=self.decimal,
                                                    skiprows=self.skip_first_rows, usecols=[i]))

                # TODO detect column name before loading completely to skip loading if the following condition applies
                if column_name == self.time_column and self.take_time_from_time_column is False:
                    column_array = np.arange(start=0.0, stop=len(column_array) / self.records_per_second,
                                             step=1.0 / self.records_per_second)

                np.save(self.get_npy_file_path(column_name), column_array)

            """ Exporting npy arrays of averaged columns """
            for columns_names in self.columns_to_be_averaged:
                temp_array = np.zeros((1))
                for column_name in columns_names:
                    temp_array = temp_array + np.load(self.get_npy_file_path(column_name)).flatten()
                avg = temp_array / len(columns_names)

                np.save(self.get_average_npy_file_path(columns_names), avg)

            self.print_custom('Finished parsing csv into npy files.')
        except:
            self.log_exception()

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

    def get_suffix_for_columns_to_be_averaged(self, columns_names):
        suffix_for_saved_file_name = 'avg_' + '_'.join(columns_names)
        return suffix_for_saved_file_name

    def export_data_json(self):
        # Output data MUST have exactly similar keys and variable names
        output_data = {'take_time_from_time_column': self.take_time_from_time_column,
                       'time_column': self.time_column,
                       'records_per_second': self.records_per_second,
                       'skip_first_rows': self.skip_first_rows,
                       'columns_headers': self.columns_headers,
                       'columns_to_be_averaged': self.columns_to_be_averaged,
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
                       'min_cycle_force_range': self.min_cycle_force_range}
        with open(self.get_json_file_path(), 'w') as outfile:
            json.dump(output_data, outfile, sort_keys=True, indent=4)
        self.print_custom('.json data file exported successfully.')

    def import_data_json(self):
        json_path = self.get_json_file_path()
        if not os.path.isfile(json_path):
            return
        # class_vars is a list with class variables names
        # vars(self) & self.__dict__.items() didn't include some Trait variables like force_column = tr.Enum(values=..
        class_vars = [attr for attr in dir(self) if not attr.startswith("_") and not attr.startswith("__")]
        with open(json_path) as infile:
            data_in = json.load(infile)
        for key_data, value_data in data_in.items():
            for key_class in class_vars:
                if key_data == key_class:
                    # Equivalent to: self.key_class = value_data
                    setattr(self, key_class, value_data)
                    break
        self.print_custom('.json data file imported successfully.')

    def get_json_file_path(self):
        return os.path.join(self.npy_folder_path, self.file_name + '.json')

    def _generate_filtered_and_creep_npy_fired(self):
        # Run method on different thread so GUI doesn't freeze
        # thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.generate_filtered_and_creep_npy_fired)
        thread.start()

    def generate_filtered_and_creep_npy_fired(self):
        try:
            self.export_data_json()
            if not self.npy_files_exist(self.get_npy_file_path(self.force_column)):
                return
            self.print_custom('Generating filtered and creep files...')

            # 1- Export filtered force
            force = np.load(self.get_npy_file_path(self.force_column)).flatten()
            peak_force_before_cycles_index = np.where(abs((force)) > abs(self.peak_force_before_cycles))[0][0]
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

    def get_max_indices(self, a):
        # TODO try to vectorize this
        # This method doesn't qualify first and last elements as max
        max_indices = []
        i = 1
        while i < a.size - 1:
            previous_element = a[i - 1]

            # Skip repeated elements and record previous element value
            first_repeated_element = True
            while a[i] == a[i + 1] and i < a.size - 1:
                if first_repeated_element:
                    previous_element = a[i - 1]
                    first_repeated_element = False
                if i < a.size - 2:
                    i += 1
                else:
                    break

            # Append value if it's a local max
            if a[i] > a[i + 1] and a[i] > previous_element:
                max_indices.append(i)
            i += 1
        return np.array(max_indices)

    def get_min_indices(self, a):
        # TODO try to vectorize this
        # This method doesn't qualify first and last elements as min
        min_indices = []
        i = 1
        while i < a.size - 1:
            previous_element = a[i - 1]

            # Skip repeated elements and record previous element value
            first_repeated_element = True
            while a[i] == a[i + 1]:
                if first_repeated_element:
                    previous_element = a[i - 1]
                    first_repeated_element = False
                if i < a.size - 2:
                    i += 1
                else:
                    break

            # Append value if it's a local min
            if a[i] < a[i + 1] and a[i] < previous_element:
                min_indices.append(i)
            i += 1
        return np.array(min_indices)

    def cut_indices_of_min_max_range(self, array, max_indices, min_indices,
                                     range_upper_value, range_lower_value):
        # TODO try to vectorize this
        cut_max_indices = []
        cut_min_indices = []

        for max_index in max_indices:
            if abs(array[max_index]) > abs(range_upper_value):
                cut_max_indices.append(max_index)
        for min_index in min_indices:
            if abs(array[min_index]) < abs(range_lower_value):
                cut_min_indices.append(min_index)
        return cut_max_indices, cut_min_indices

    def cut_indices_of_defined_range(self, array, max_indices, min_indices, range_):
        # TODO try to vectorize this
        cut_max_indices = []
        cut_min_indices = []

        for max_index, min_index in zip(max_indices, min_indices):
            if abs(array[max_index] - array[min_index]) > range_:
                cut_max_indices.append(max_index)
                cut_min_indices.append(min_index)

        if max_indices.size > min_indices.size:
            cut_max_indices.append(max_indices[-1])
        elif min_indices.size > max_indices.size:
            cut_min_indices.append(min_indices[-1])

        return cut_max_indices, cut_min_indices

    def _activate_changed(self):
        if not self.activate_ascending_branch_smoothing:
            self.old_peak_force_before_cycles = self.peak_force_before_cycles
            self.peak_force_before_cycles = 0
        else:
            self.peak_force_before_cycles = self.old_peak_force_before_cycles

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

    # =========================================================================
    # Plotting
    # =========================================================================
    data_changed = tr.Event

    def _plot_settings_btn_fired(self):
        try:
            self.plot_settings.configure_traits(kind='modal')
        except:
            self.log_exception()

    def npy_files_exist(self, path):
        if os.path.exists(path):
            return True
        else:
            self.print_custom('Please parse csv file to generate npy files first!')
            return False

    def filtered_and_creep_npy_files_exist(self, path):
        if os.path.exists(path):
            return True
        else:
            self.print_custom('Please generate filtered and creep npy files first!')
            return False

    def _add_plot_fired(self):
        # Run method on different thread so GUI doesn't freeze
        # thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.add_plot_fired)
        thread.start()

    def add_plot_fired(self):
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

            if self.plot_settings_active:
                print(self.plot_settings.num_of_first_rows_to_take)
                print(self.plot_settings.num_of_rows_to_skip_after_each_section)
                print(self.plot_settings.num_of_rows_in_each_section)
                print(np.size(x_axis_array))
                indices = self.get_indices_array(np.size(x_axis_array),
                                                 self.plot_settings.num_of_first_rows_to_take,
                                                 self.plot_settings.num_of_rows_to_skip_after_each_section,
                                                 self.plot_settings.num_of_rows_in_each_section)
                x_axis_array = self.x_axis_multiplier * x_axis_array[indices]
                y_axis_array = self.y_axis_multiplier * y_axis_array[indices]
            else:
                x_axis_array = self.x_axis_multiplier * x_axis_array
                y_axis_array = self.y_axis_multiplier * y_axis_array

            self.print_custom('Adding Plot...')
            mpl.rcParams['agg.path.chunksize'] = 10000

            ax = self.figure.add_subplot(1, 1, 1)

            ax.set_xlabel(x_axis_name)
            ax.set_ylabel(y_axis_name)
            ax.plot(x_axis_array, y_axis_array, 'k', linewidth=1.2, color=np.random.rand(3),
                    label=self.file_name + ', ' + x_axis_name)

            ax.legend()
            self.data_changed = True
            self.print_custom('Finished adding plot.')

        except:
            self.log_exception()

    def _add_creep_plot_fired(self):
        # Run method on different thread so GUI doesn't freeze
        # thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.add_creep_plot_fired)
        thread.start()

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

            ax = self.figure.add_subplot(1, 1, 1)

            ax.set_xlabel('Cycles number')
            ax.set_ylabel(self.x_axis)

            if self.plot_every_nth_point > 1:
                disp_max = disp_max[0::self.plot_every_nth_point]
                disp_min = disp_min[0::self.plot_every_nth_point]

            if self.smooth:
                # Keeping the first item of the array and filtering the rest
                disp_max = np.concatenate((
                    np.array([disp_max[0]]),
                    savgol_filter(disp_max[1:], window_length=self.window_length, polyorder=self.polynomial_order)
                ))
                disp_min = np.concatenate((
                    np.array([disp_min[0]]),
                    savgol_filter(disp_min[1:], window_length=self.window_length, polyorder=self.polynomial_order)
                ))

            if self.normalize_cycles:
                ax.plot(np.linspace(0, 1., disp_max.size), disp_max,
                        'k', linewidth=1.2, color=np.random.rand(3), label='Max'
                                                                           + ', ' + self.file_name + ', ' + self.x_axis)
                ax.plot(np.linspace(0, 1., disp_min.size), disp_min,
                        'k', linewidth=1.2, color=np.random.rand(3), label='Min'
                                                                           + ', ' + self.file_name + ', ' + self.x_axis)
            else:
                ax.plot(np.linspace(0, complete_cycles_number,
                                    disp_max.size), disp_max,
                        'k', linewidth=1.2, color=np.random.rand(3), label='Max'
                                                                           + ', ' + self.file_name + ', ' + self.x_axis)
                ax.plot(np.linspace(0, complete_cycles_number,
                                    disp_min.size), disp_min,
                        'k', linewidth=1.2, color=np.random.rand(3), label='Min'
                                                                           + ', ' + self.file_name + ', ' + self.x_axis)

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
        result_1 = np.arange(first_rows)
        result_2 = np.arange(start=first_rows, stop=array_size,
                             step=distance + num_of_rows_after_each_distance)
        result_2_updated = np.array([], dtype=np.int_)

        for result_2_value in result_2:
            data_slice = np.arange(result_2_value, result_2_value +
                                   num_of_rows_after_each_distance)
            result_2_updated = np.concatenate((result_2_updated, data_slice))

        result = np.concatenate((result_1, result_2_updated))
        return result

    def _clear_plot_fired(self):
        self.figure.clear()
        self.data_changed = True

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

    def log_exception(self):
        self.print_custom('SOMETHING WENT WRONG!')
        self.print_custom('--------- Error message: ---------')
        self.print_custom(traceback.format_exc())
        self.print_custom('----------------------------------')

    def _clear_log_fired(self):
        self.log = ''

    # =========================================================================
    # Other functions
    # =========================================================================
    def reset(self):
        self.columns_to_be_averaged = []
        self.log = ''

if __name__ == '__main__':
    hcft = HCFT(file_path=os.path.expanduser("~"))
    hcft.configure_traits()
