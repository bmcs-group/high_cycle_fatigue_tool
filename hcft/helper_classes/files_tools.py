import os
import string
import subprocess
import sys

from pyface.api import YES, NO
from pyface.confirmation_dialog import confirm


def get_valid_file_name(original_file_name):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    new_valid_file_name = ''.join(
        c for c in original_file_name if c in valid_chars)
    return new_valid_file_name


def open_file(path):
    """Opens the file with the default application of the system"""
    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


def open_file_directory(path):
    """Opens the directory of the file in the file manager, with the file selected where supported"""
    if sys.platform == 'win32':
        subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', '-R', path])
    else:
        subprocess.Popen(['xdg-open', os.path.dirname(path)])


def ask_to_open_saved_file(path, title='File saved'):
    """Asks the user whether to open the saved file, its directory or nothing"""
    result = confirm(parent=None,
                     message='File saved successfully:\n' + path,
                     title=title,
                     cancel=True,
                     default=YES,
                     yes_label='Open file',
                     no_label='Open folder')
    if result == YES:
        open_file(path)
    elif result == NO:
        open_file_directory(path)
