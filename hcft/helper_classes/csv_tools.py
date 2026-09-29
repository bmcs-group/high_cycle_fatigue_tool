import pandas as pd
from .files_tools import get_valid_file_name


def get_headers(file_path, delimiter, decimal):
    # One could provide the path directly to pd.read_csv but in this way we insure that this works also if the
    # path to the file include chars like ü,ä
    # (with) makes sure the file stream is closed after using it
    with open(file_path, encoding='latin-1') as file_stream:
        headers = pd.read_csv(file_stream, delimiter=delimiter, decimal=decimal, nrows=1, header=None).iloc[0]
    return [get_valid_file_name(str(header)) for header in headers]
