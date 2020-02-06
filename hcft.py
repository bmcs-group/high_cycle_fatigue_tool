'''
Created on Apr 24, 2019

@author: Homam Spartali, Rostislav Chudoba

Note: To use this tool, the csv file must have the columns headers in
the first row.

'''
import os
from pathlib import Path
import string
import sys
from threading import Thread
import traceback

import pyface
from pyface.api import FileDialog, MessageDialog, OK
from scipy.signal import savgol_filter
from util.traits.editors import MPLFigureEditor

import matplotlib as mpl
import numpy as np
import pandas as pd
import traits.api as tr
import traitsui.api as ui
from traitsui.extras.checkbox_column \
    import CheckboxColumn


average_columns_editor = ui.TableEditor(
    sortable=False,
    configurable=False,
    auto_size=False,
    columns=[CheckboxColumn(name='selected', label='Select',
                            width=0.12),
             ui.ObjectColumn(name='column_name', editable=False, width=0.24,
                             horizontal_alignment='left')])


class Column(tr.HasStrictTraits):
    column_name = tr.Str
    selected = tr.Bool(False)


class ColumnsAverage(tr.HasStrictTraits):
    columns = tr.List(Column)

    # Trait view definitions:
    traits_view = ui.View(
        ui.Item('columns',
                show_label=False,
                editor=average_columns_editor
                ),
        buttons=[ui.OKButton, ui.CancelButton],
        title='Select arrays to be averaged',
        width=0.15,
        height=0.3,
        resizable=True
    )


class HCFT(tr.HasStrictTraits):
    '''High-Cycle Fatigue Tool
    '''
    #=========================================================================
    # Traits definitions
    #=========================================================================
    decimal = tr.Enum(',', '.')
    delimiter = tr.Str(';')
    records_per_second = tr.Float(100)
    take_time_from_time_column = tr.Bool(True)
    file_csv = tr.File
    open_file_csv = tr.Button('Input file')
    skip_first_rows = tr.Range(low=1, high=10**9, mode='spinner')
    columns_headers_list = tr.List([])
    x_axis = tr.Enum(values='columns_headers_list')
    y_axis = tr.Enum(values='columns_headers_list')
    force_column = tr.Enum(values='columns_headers_list')
    time_column = tr.Enum(values='columns_headers_list')
    x_axis_multiplier = tr.Enum(1, -1)
    y_axis_multiplier = tr.Enum(-1, 1)
    npy_folder_path = tr.Str
    file_name = tr.Str
    apply_filters = tr.Bool
    normalize_cycles = tr.Bool
    smooth = tr.Bool
    plot_every_nth_point = tr.Range(low=1, high=1000000, mode='spinner')
    old_peak_force_before_cycles = tr.Float
    peak_force_before_cycles = tr.Float
    window_length = tr.Range(low=1, high=10**9 - 1, value=31, mode='spinner')
    polynomial_order = tr.Range(low=1, high=10**9, value=2, mode='spinner')
    activate = tr.Bool(False)
    add_plot = tr.Button
    add_creep_plot = tr.Button(desc='Creep plot of X axis array')
    clear_plot = tr.Button
    parse_csv_to_npy = tr.Button
    generate_filtered_and_creep_npy = tr.Button
    add_columns_average = tr.Button
    force_max = tr.Float(100)
    force_min = tr.Float(40)
    min_cycle_force_range = tr.Float(50)
    cutting_method = tr.Enum(
        'Define min cycle range(force difference)', 'Define Max, Min')
    columns_to_be_averaged = tr.List
    figure = tr.Instance(mpl.figure.Figure)
    log = tr.Str('')
    clear_log = tr.Button

    def _figure_default(self):
        figure = mpl.figure.Figure(facecolor='white')
        figure.set_tight_layout(True)
        return figure

    #=========================================================================
    # File management
    #=========================================================================

    def _open_file_csv_fired(self):
        try:

            self.reset()

            """ Handles the user clicking the 'Open...' button.
            """
            extns = ['*.csv', ]  # seems to handle only one extension...
            wildcard = '|'.join(extns)

            dialog = FileDialog(title='Select text file',
                                action='open', wildcard=wildcard,
                                default_path=self.file_csv)

            result = dialog.open()

            """ Test if the user opened a file to avoid throwing an exception if he 
            doesn't """
            if result == OK:
                self.file_csv = dialog.path
            else:
                return

            """ Filling x_axis and y_axis with values """
            headers_array = np.array(
                pd.read_csv(
                    self.file_csv, delimiter=self.delimiter, decimal=self.decimal,
                    nrows=1, header=None
                )
            )[0]
            for i in range(len(headers_array)):
                headers_array[i] = self.get_valid_file_name(headers_array[i])
            self.columns_headers_list = list(headers_array)

            """ Saving file name and path and creating NPY folder """
            dir_path = os.path.dirname(self.file_csv)
            self.npy_folder_path = os.path.join(dir_path, 'NPY')
            if os.path.exists(self.npy_folder_path) == False:
                os.makedirs(self.npy_folder_path)

            self.file_name = os.path.splitext(
                os.path.basename(self.file_csv))[0]

        except Exception as e:
            self.deal_with_exception(e)

    def _parse_csv_to_npy_fired(self):
        # Run method on different thread so GUI doesn't freeze
        #thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.parse_csv_to_npy_fired)
        thread.start()

    def parse_csv_to_npy_fired(self):
        try:
            self.print_custom('Parsing csv into npy files...')

            for i in range(len(self.columns_headers_list) -
                           len(self.columns_to_be_averaged)):
                current_column_name = self.columns_headers_list[i]
                column_array = np.array(pd.read_csv(
                    self.file_csv, delimiter=self.delimiter, decimal=self.decimal,
                    skiprows=self.skip_first_rows, usecols=[i]))

                if current_column_name == self.time_column and \
                        self.take_time_from_time_column == False:
                    column_array = np.arange(start=0.0,
                                             stop=len(column_array) /
                                             self.records_per_second,
                                             step=1.0 / self.records_per_second)

                np.save(os.path.join(self.npy_folder_path, self.file_name +
                                     '_' + current_column_name + '.npy'),
                        column_array)

            """ Exporting npy arrays of averaged columns """
            for columns_names in self.columns_to_be_averaged:
                temp = np.zeros((1))
                for column_name in columns_names:
                    temp = temp + np.load(os.path.join(self.npy_folder_path,
                                                       self.file_name +
                                                       '_' + column_name +
                                                       '.npy')).flatten()
                avg = temp / len(columns_names)

                avg_file_suffex = self.get_suffex_for_columns_to_be_averaged(
                    columns_names)
                np.save(os.path.join(self.npy_folder_path, self.file_name +
                                     '_' + avg_file_suffex + '.npy'), avg)

            self.print_custom('Finsihed parsing csv into npy files.')
        except Exception as e:
            self.deal_with_exception(e)

    def get_suffex_for_columns_to_be_averaged(self, columns_names):
        suffex_for_saved_file_name = 'avg_' + '_'.join(columns_names)
        return suffex_for_saved_file_name

    def get_valid_file_name(self, original_file_name):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        new_valid_file_name = ''.join(
            c for c in original_file_name if c in valid_chars)
        return new_valid_file_name

    def _clear_plot_fired(self):
        self.figure.clear()
        self.data_changed = True

    def _add_columns_average_fired(self):
        try:
            columns_average = ColumnsAverage()
            for name in self.columns_headers_list:
                columns_average.columns.append(Column(column_name=name))

            # kind='modal' pauses the implementation until the window is closed
            columns_average.configure_traits(kind='modal')

            columns_to_be_averaged_temp = []
            for i in columns_average.columns:
                if i.selected:
                    columns_to_be_averaged_temp.append(i.column_name)

            if columns_to_be_averaged_temp:  # If it's not empty
                self.columns_to_be_averaged.append(columns_to_be_averaged_temp)

                avg_file_suffex = self.get_suffex_for_columns_to_be_averaged(
                    columns_to_be_averaged_temp)
                self.columns_headers_list.append(avg_file_suffex)
        except Exception as e:
            self.deal_with_exception(e)

    def _generate_filtered_and_creep_npy_fired(self):
        # Run method on different thread so GUI doesn't freeze
        #thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.generate_filtered_and_creep_npy_fired)
        thread.start()

    def generate_filtered_and_creep_npy_fired(self):
        try:
            if self.npy_files_exist(os.path.join(
                    self.npy_folder_path, self.file_name + '_' + self.force_column
                    + '.npy')) == False:
                return

            self.print_custom('Generating filtered and creep files...')

            # 1- Export filtered force
            force = np.load(os.path.join(self.npy_folder_path,
                                         self.file_name + '_' + self.force_column
                                         + '.npy')).flatten()
            peak_force_before_cycles_index = np.where(
                abs((force)) > abs(self.peak_force_before_cycles))[0][0]
            force_ascending = force[0:peak_force_before_cycles_index]
            force_rest = force[peak_force_before_cycles_index:]

            force_max_indices, force_min_indices = self.get_array_max_and_min_indices(
                force_rest)

            force_max_min_indices = np.concatenate(
                (force_min_indices, force_max_indices))
            force_max_min_indices.sort()

            force_rest_filtered = force_rest[force_max_min_indices]
            force_filtered = np.concatenate(
                (force_ascending, force_rest_filtered))
            np.save(os.path.join(self.npy_folder_path, self.file_name +
                                 '_' + self.force_column + '_filtered.npy'),
                    force_filtered)

            # 2- Export filtered displacements
            for i in range(0, len(self.columns_headers_list)):
                if self.columns_headers_list[i] != self.force_column and \
                        self.columns_headers_list[i] != self.time_column:

                    disp = np.load(os.path.join(self.npy_folder_path, self.file_name
                                                + '_' +
                                                self.columns_headers_list[i]
                                                + '.npy')).flatten()
                    disp_ascending = disp[0:peak_force_before_cycles_index]
                    disp_rest = disp[peak_force_before_cycles_index:]

                    if self.activate == True:
                        disp_ascending = savgol_filter(
                            disp_ascending, window_length=self.window_length,
                            polyorder=self.polynomial_order)

                    disp_rest_filtered = disp_rest[force_max_min_indices]
                    filtered_disp = np.concatenate(
                        (disp_ascending, disp_rest_filtered))
                    np.save(os.path.join(self.npy_folder_path, self.file_name + '_'
                                         + self.columns_headers_list[i] +
                                         '_filtered.npy'), filtered_disp)

            # 3- Export creep for displacements
            # Cutting unwanted max min values to get correct full cycles and remove
            # false min/max values caused by noise
            if self.cutting_method == "Define Max, Min":
                force_max_indices_cutted, force_min_indices_cutted = \
                    self.cut_indices_of_min_max_range(force_rest,
                                                      force_max_indices,
                                                      force_min_indices,
                                                      self.force_max,
                                                      self.force_min)
            elif self.cutting_method == "Define min cycle range(force difference)":
                force_max_indices_cutted, force_min_indices_cutted = \
                    self.cut_indices_of_defined_range(force_rest,
                                                      force_max_indices,
                                                      force_min_indices,
                                                      self.min_cycle_force_range)

            self.print_custom("Cycles number= ", len(force_min_indices))
            self.print_custom("Cycles number after cutting fake cycles = ",
                              len(force_min_indices_cutted))

            for i in range(0, len(self.columns_headers_list)):
                if self.columns_headers_list[i] != self.time_column:
                    array = np.load(os.path.join(self.npy_folder_path, self.file_name +
                                                 '_' +
                                                 self.columns_headers_list[i]
                                                 + '.npy')).flatten()
                    array_rest = array[peak_force_before_cycles_index:]
                    array_rest_maxima = array_rest[force_max_indices_cutted]
                    array_rest_minima = array_rest[force_min_indices_cutted]
                    np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                         self.columns_headers_list[i] + '_max.npy'), array_rest_maxima)
                    np.save(os.path.join(self.npy_folder_path, self.file_name + '_' +
                                         self.columns_headers_list[i] + '_min.npy'), array_rest_minima)

            self.print_custom('Filtered and creep npy files are generated.')
        except Exception as e:
            self.deal_with_exception(e)

    def cut_indices_of_min_max_range(self, array, max_indices, min_indices,
                                     range_upper_value, range_lower_value):
        cutted_max_indices = []
        cutted_min_indices = []

        for max_index in max_indices:
            if abs(array[max_index]) > abs(range_upper_value):
                cutted_max_indices.append(max_index)
        for min_index in min_indices:
            if abs(array[min_index]) < abs(range_lower_value):
                cutted_min_indices.append(min_index)
        return cutted_max_indices, cutted_min_indices

    def cut_indices_of_defined_range(self, array, max_indices, min_indices, range_):
        cutted_max_indices = []
        cutted_min_indices = []

        for max_index, min_index in zip(max_indices, min_indices):
            if abs(array[max_index] - array[min_index]) > range_:
                cutted_max_indices.append(max_index)
                cutted_min_indices.append(min_index)

        if max_indices.size > min_indices.size:
            cutted_max_indices.append(max_indices[-1])
        elif min_indices.size > max_indices.size:
            cutted_min_indices.append(min_indices[-1])

        return cutted_max_indices, cutted_min_indices

    def get_array_max_and_min_indices(self, input_array):

        # Checking dominant sign
        positive_values_count = np.sum(np.array(input_array) >= 0)
        negative_values_count = input_array.size - positive_values_count

        # Getting max and min indices
        if (positive_values_count > negative_values_count):
            force_max_indices = self.get_max_indices(input_array)
            force_min_indices = self.get_min_indices(input_array)
        else:
            force_max_indices = self.get_min_indices(input_array)
            force_min_indices = self.get_max_indices(input_array)

        return force_max_indices, force_min_indices

    def get_max_indices(self, a):
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

            if a[i] > a[i + 1] and a[i] > previous_element:
                max_indices.append(i)
            i += 1
        return np.array(max_indices)

    def get_min_indices(self, a):
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

            if a[i] < a[i + 1] and a[i] < previous_element:
                min_indices.append(i)
            i += 1
        return np.array(min_indices)

    def _activate_changed(self):
        if self.activate == False:
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

    #=========================================================================
    # Plotting
    #=========================================================================

    def npy_files_exist(self, path):
        if os.path.exists(path) == True:
            return True
        else:
            dialog = MessageDialog(
                title='Attention!',
                message='Please parse csv file to generate npy files first.')
            dialog.open()
            return False

    def filtered_and_creep_npy_files_exist(self, path):
        if os.path.exists(path) == True:
            return True
        else:
            dialog = MessageDialog(
                title='Attention!',
                message='Please generate filtered and creep npy files first.')
            dialog.open()
            return False

    data_changed = tr.Event

    def _add_plot_fired(self):
        # Run method on different thread so GUI doesn't freeze
        #thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.add_plot_fired)
        thread.start()

    def add_plot_fired(self):
        try:
            self.print_custom('Loading npy files...')
            if self.apply_filters:
                if self.filtered_and_creep_npy_files_exist(os.path.join(
                        self.npy_folder_path, self.file_name + '_' + self.x_axis
                        + '_filtered.npy')) == False:
                    return

                x_axis_name = self.x_axis + '_filtered'
                y_axis_name = self.y_axis + '_filtered'
                x_axis_array = self.x_axis_multiplier * \
                    np.load(os.path.join(self.npy_folder_path,
                                         self.file_name + '_' + self.x_axis
                                         + '_filtered.npy'))
                y_axis_array = self.y_axis_multiplier * \
                    np.load(os.path.join(self.npy_folder_path,
                                         self.file_name + '_' + self.y_axis
                                         + '_filtered.npy'))
            else:
                if self.npy_files_exist(os.path.join(
                        self.npy_folder_path, self.file_name + '_' + self.x_axis
                        + '.npy')) == False:
                    return

                x_axis_name = self.x_axis
                y_axis_name = self.y_axis
                x_axis_array = self.x_axis_multiplier * \
                    np.load(os.path.join(self.npy_folder_path,
                                         self.file_name + '_' + self.x_axis
                                         + '.npy'))
                y_axis_array = self.y_axis_multiplier * \
                    np.load(os.path.join(self.npy_folder_path,
                                         self.file_name + '_' + self.y_axis
                                         + '.npy'))

            self.print_custom('Adding Plot...')
            mpl.rcParams['agg.path.chunksize'] = 10000

            ax = self.figure.add_subplot(1, 1, 1)

            ax.set_xlabel(x_axis_name)
            ax.set_ylabel(y_axis_name)
            ax.plot(x_axis_array, y_axis_array, 'k',
                    linewidth=1.2, color=np.random.rand(3), label=self.file_name +
                    ', ' + x_axis_name)

            ax.legend()
            self.data_changed = True
            self.print_custom('Finished adding plot.')

        except Exception as e:
            self.deal_with_exception(e)

    def _add_creep_plot_fired(self):
        # Run method on different thread so GUI doesn't freeze
        #thread = Thread(target = threaded_function, function_args = (10,))
        thread = Thread(target=self.add_creep_plot_fired)
        thread.start()

    def add_creep_plot_fired(self):

        try:

            if self.filtered_and_creep_npy_files_exist(os.path.join(
                    self.npy_folder_path, self.file_name + '_' + self.x_axis
                    + '_max.npy')) == False:
                return

            self.print_custom('Loading npy files...')
            disp_max = self.x_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '_max.npy'))
            disp_min = self.x_axis_multiplier * \
                np.load(os.path.join(self.npy_folder_path,
                                     self.file_name + '_' + self.x_axis + '_min.npy'))
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
                    savgol_filter(disp_max[1:],
                                  window_length=self.window_length,
                                  polyorder=self.polynomial_order)
                ))
                disp_min = np.concatenate((
                    np.array([disp_min[0]]),
                    savgol_filter(disp_min[1:],
                                  window_length=self.window_length,
                                  polyorder=self.polynomial_order)
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

        except Exception as e:
            self.deal_with_exception(e)

    def reset(self):
        self.columns_to_be_averaged = []
        self.log = ''

    def print_custom(self, *input_args):
        print(*input_args)
        if self.log == '':
            self.log = ''.join(str(e) for e in list(input_args))
        else:
            self.log = self.log + '\n' + \
                ''.join(str(e) for e in list(input_args))

    def deal_with_exception(self, e):
        self.print_custom('SOMETHING WENT WRONG!')
        self.print_custom('--------- Error message: ---------')
        self.print_custom(traceback.format_exc())
        self.print_custom('----------------------------------')

    def _clear_log_fired(self):
        self.log = ''

    #=========================================================================
    # Configuration of the view
    #=========================================================================

    traits_view = ui.View(
        ui.HSplit(
            ui.VSplit(
                ui.VGroup(
                    ui.VGroup(
                        ui.HGroup(
                            ui.UItem('open_file_csv', has_focus=True),
                            ui.UItem('file_csv', style='readonly', width=0.1)),
                        label='Importing csv file',
                        show_border=True)),
                ui.VGroup(
                    ui.VGroup(
                        ui.VGroup(
                            ui.Item('take_time_from_time_column'),
                            ui.Item('time_column',
                                    enabled_when='take_time_from_time_column == True'),
                            ui.Item('records_per_second',
                                    enabled_when='take_time_from_time_column == False'),
                            label='Time calculation',
                            show_border=True),
                        ui.UItem('add_columns_average'),
                        ui.Item('decimal'),
                        ui.Item('delimiter'),
                        ui.Item('skip_first_rows'),
                        ui.UItem('parse_csv_to_npy', resizable=True),
                        label='Processing csv file',
                        show_border=True)),
                ui.VGroup(
                    ui.VGroup(
                        ui.HGroup(ui.Item('x_axis'), ui.Item(
                            'x_axis_multiplier')),
                        ui.HGroup(ui.Item('y_axis'), ui.Item(
                            'y_axis_multiplier')),
                        ui.VGroup(
                            ui.HGroup(ui.UItem('add_plot'),
                                      ui.Item('apply_filters')
                                      ),
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
                        show_border=True,
                        label='Plotting'))
            ),
            ui.VGroup(
                ui.Item('force_column'),
                ui.VGroup(ui.VGroup(
                    ui.Item('window_length'),
                    ui.Item('polynomial_order'),
                    enabled_when='activate == True or smooth == True'),
                    show_border=True,
                    label='Smoothing parameters (Savitzky-Golay filter):'
                ),
                ui.VGroup(ui.VGroup(
                    ui.Item('activate'),
                    ui.Item('peak_force_before_cycles',
                            enabled_when='activate == True')
                ),
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
                                    enabled_when='cutting_method == "Define min cycle range(force difference)"'),
                          show_border=True,
                          label='Cut fake cycles for creep:'),

                ui.VSplit(
                    ui.UItem('generate_filtered_and_creep_npy'),
                    ui.VGroup(
                        ui.Item('log',
                                width=0.1, style='custom'),
                        ui.UItem('clear_log'))),
                show_border=True,
                label='Filters'
            ),
            ui.UItem('figure', editor=MPLFigureEditor(),
                     resizable=True,
                     springy=True,
                     width=0.8,
                     label='2d plots')
        ),
        title='High-cycle fatigue tool',
        resizable=True,
        width=0.85,
        height=0.7

    )


if __name__ == '__main__':
    hcft = HCFT(file_csv=os.path.expanduser("~"))
    hcft.configure_traits()
