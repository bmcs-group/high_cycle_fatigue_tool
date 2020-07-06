'''
Created on 15 Apr 2020

@author: Homam
'''
import traits.api as tr
import traitsui.api as ui
import pyface.api as pf
import numpy as np
import pandas as pd
import os
from itertools import takewhile, repeat

from pyface.message_dialog import MessageDialog
from traitsui.editors import ProgressEditor


class CSVFile(tr.HasStrictTraits):
    path = tr.Str
    count_lines = tr.Button
    _lines_number = tr.Int  # _ is the private field convention
    show_lines_number = tr.Bool
    num_of_first_lines_to_show = tr.Int(10)
    num_of_last_lines_to_show = tr.Int(10)

    first_lines = tr.Property(depends_on='path, num_of_first_lines_to_show, first_lines_to_skip')
    last_lines = tr.Property(depends_on='path, num_of_last_lines_to_show, last_lines_to_skip')
    first_lines_to_skip = tr.Range(low=0, high=10 ** 9, mode='spinner')
    last_lines_to_skip = tr.Range(low=0, high=10 ** 9, mode='spinner')

    def _count_lines_fired(self):
        #         return sum(1 for line in open(self.path))
        self.show_lines_number = True
        self._lines_number = self._count_lines_in_file(self.path)

    def get_lines_number(self):
        # If it was not yet calculated, calc it first
        if self._lines_number == 0:
            self._lines_number = self._count_lines_in_file(self.path)
        return self._lines_number

    def _count_lines_in_file(self, file_name):
        ''' This method will count the number of lines in a huge file pretty
        quickly using custom buffering'''
        f = open(file_name, 'rb')
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for i in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen) + 1

    @tr.cached_property
    def _get_first_lines(self):
        first_lines_list = []
        with open(self.path) as myfile:
            for i in range(self.num_of_first_lines_to_show):
                try:
                    # Get next line if it exists!
                    line = next(myfile)
                    # The following will not executed if an exception was thrown
                    first_lines_list.append(line)
                except StopIteration:
                    # If last line ends with \n then a new empty line is 
                    # actually there
                    if len(first_lines_list) != 0:
                        if first_lines_list[-1].endswith('\n'):
                            first_lines_list.append('')
                    break
        first_lines_list = self.add_line_numbers(first_lines_list)
        first_lines_list = first_lines_list[self.first_lines_to_skip:]
        first_lines_str = ''.join(first_lines_list)
        return first_lines_str

    def add_line_numbers(self, lines_list):
        new_list = []
        for line_num, line in zip(range(1, len(lines_list) + 1), lines_list):
            new_list.append('(' + str(line_num) + ')--> ' + str(line))
        return new_list

    def add_reverse_line_numbers(self, lines_list):
        new_list = []
        for line_num, line in zip(range(len(lines_list), 0, -1), lines_list):
            new_list.append('(' + str(line_num) + ')--> ' + str(line))
        return new_list

    @tr.cached_property
    def _get_last_lines(self):
        last_lines_list = self.get_last_n_lines(self.path, self.num_of_last_lines_to_show, False)
        last_lines_list = self.add_reverse_line_numbers(last_lines_list)
        last_lines_list = last_lines_list[0:len(last_lines_list) - self.last_lines_to_skip]
        last_lines_str = ''.join(last_lines_list)
        return last_lines_str

    def get_last_n_lines(self, file_name, N, skip_empty_lines=False):
        # Create an empty list to keep the track of last N lines
        list_of_lines = []
        # Open file for reading in binary mode
        with open(file_name, 'rb') as read_obj:
            # Move the cursor to the end of the file
            read_obj.seek(0, os.SEEK_END)
            # Create a buffer to keep the last read line
            buffer = bytearray()
            # Get the current position of pointer i.e eof
            pointer_location = read_obj.tell()
            # Loop till pointer reaches the top of the file
            while pointer_location >= 0:
                # Move the file pointer to the location pointed by pointer_location
                read_obj.seek(pointer_location)
                # Shift pointer location by -1
                pointer_location = pointer_location - 1
                # read that byte / character
                new_byte = read_obj.read(1)
                # If the read byte is new line character then it means one line is read
                if new_byte == b'\n':
                    # Save the line in list of lines
                    line = buffer.decode()[::-1]
                    if (skip_empty_lines):
                        line_is_empty = line.isspace()
                        if (line_is_empty == False):
                            list_of_lines.append(line)
                    else:
                        list_of_lines.append(line)
                    # If the size of list reaches N, then return the reversed list
                    if len(list_of_lines) == N:
                        return list(reversed(list_of_lines))
                    # Reinitialize the byte array to save next line
                    buffer = bytearray()
                else:
                    # If last read character is not eol then add it in buffer
                    buffer.extend(new_byte)

            # As file is read completely, if there is still data in buffer, then its first line.
            if len(buffer) > 0:
                list_of_lines.append(buffer.decode()[::-1])
        # return the reversed list
        return list(reversed(list_of_lines))

    traits_view = ui.View(
        ui.Item('path', style='readonly', label='File'),
        ui.HGroup(ui.UItem('count_lines'),
                  ui.Item('_lines_number', style='readonly',
                          visible_when='show_lines_number == True')),
        ui.VSplit(
            ui.HGroup(ui.Item('first_lines', style='custom'),
                      'first_lines_to_skip',
                      label='First lines in the file'),
            ui.HGroup(ui.Item('last_lines', style='custom'),
                      'last_lines_to_skip',
                      label='Last lines in the file')
        ))


class CSVJoiner(tr.HasStrictTraits):
    open_csv_files = tr.Button
    csv_files = tr.List(CSVFile)
    num_of_first_lines_to_show = tr.Range(low=0, high=10 ** 9, value=10, mode='spinner')
    num_of_last_lines_to_show = tr.Range(low=0, high=10 ** 9, value=10, mode='spinner')
    selected = tr.Instance(CSVFile)
    join_csv_files = tr.Button
    accumulate_time = tr.Bool
    files_end_with_empty_line = tr.Bool(True)
    columns_headers = tr.List
    time_column = tr.Enum(values='columns_headers')
    progress = tr.Int

    def _join_csv_files_fired(self):
        output_file_path = self.get_output_file_path()
        with open(output_file_path, 'w') as outfile:
            for csv_file, i in zip(self.csv_files, range(len(self.csv_files))):
                current_line = 1
                num_of_first_lines_to_skip = csv_file.first_lines_to_skip
                num_of_last_lines_to_skip = csv_file.last_lines_to_skip
                last_line_to_write = csv_file.get_lines_number() - num_of_last_lines_to_skip
                progress_of_a_file = 1.0 / len(self.csv_files)
                initial_progress = i/len(self.csv_files)
                with open(csv_file.path) as opened_csv_file:
                    for line in opened_csv_file:
                        if current_line > num_of_first_lines_to_skip and current_line <= last_line_to_write:
                            outfile.write(line)
                            self.progress = int((initial_progress + progress_of_a_file * (
                                    current_line / last_line_to_write)) * 100)
                        current_line += 1
                if not self.files_end_with_empty_line:
                    outfile.write('\n')
        self.progress = 100
        dialog = MessageDialog(title='Finished!', message='Files joined successfully, see "' + output_file_path + '"')
        dialog.open()

    def get_output_file_path(self):
        file_path = self.csv_files[0].path
        file_path_without_ext = os.path.splitext(file_path)[0]
        file_ext = os.path.splitext(file_path)[1]
        return file_path_without_ext + '_joined' + file_ext

    def _accumulate_time_changed(self):
        pass

    #         if self.csv_files == []:
    #             return
    #         np.array(pd.read_csv(
    #                     self.file_csv, delimiter=self.delimiter, decimal=self.decimal,
    #                     nrows=1, header=None
    #                 )
    #             )[0]
    #         if self.accumulate_time:
    #             class TimeColumnChooser(tr.HasTraits):
    #                 time_column = tr.Enum(values = 'columns_headers')
    #             chooser = TimeColumnChooser()
    #             chooser.configure_traits(kind='modal')

    def _num_of_first_lines_to_show_changed(self):
        for file in self.csv_files:
            file.num_of_first_lines_to_show = self.num_of_first_lines_to_show

    def _num_of_last_lines_to_show_changed(self):
        for file in self.csv_files:
            file.num_of_last_lines_to_show = self.num_of_last_lines_to_show

    def _open_csv_files_fired(self):
        extensions = ['*.csv', '*.txt']  # handle only one extension...
        wildcard = ';'.join(extensions)
        dialog = pf.FileDialog(title='Select csv files',
                               action='open files',
                               wildcard=wildcard,
                               default_path=os.path.expanduser("~"))
        result = dialog.open()

        csv_files_paths = []
        # Test if the user opened a file to avoid throwing an exception 
        # if he doesn't
        if result == pf.OK:
            csv_files_paths = dialog.paths
        else:
            return

        self.csv_files = []
        for file_path in csv_files_paths:
            csv_file = CSVFile(path=file_path,
                               num_of_first_lines_to_show=
                               self.num_of_first_lines_to_show,
                               num_of_last_lines_to_show=
                               self.num_of_last_lines_to_show, )
            self.csv_files.append(csv_file)

    # =========================================================================
    # Configuration of the view
    # =========================================================================
    traits_view = ui.View(
        ui.VGroup(
            ui.UItem('open_csv_files', width=150),
            ui.HGroup(ui.Item('num_of_first_lines_to_show'), ui.spring),
            ui.HGroup(ui.Item('num_of_last_lines_to_show'), ui.spring),
            ui.HGroup(ui.Item('files_end_with_empty_line'), ui.Item('accumulate_time', enabled_when='False'),
                      ui.spring),
            ui.VGroup(
                ui.Item('csv_files',
                        show_label=False,
                        style='custom',
                        editor=ui.ListEditor(use_notebook=True,
                                             deletable=False,
                                             selected='selected',
                                             export='DockWindowShell',
                                             page_name='.name')
                        )
            ),
            ui.HGroup(
                ui.UItem('join_csv_files', width=150),
                ui.UItem('progress', editor=ProgressEditor(min=0, max=100))
            ),
            show_border=True
        ),
        title='CSV files joiner',
        resizable=True,
        width=0.6,
        height=0.7
    )


if __name__ == '__main__':
    csv_joiner = CSVJoiner()
    csv_joiner.configure_traits()
