import numpy as np
import pandas as pd
from .files_tools import get_valid_file_name


def get_headers(file_path, delimiter, decimal):
    # One could provide the path directly to pd.read_csv but in this way we insure that this works also if the
    # path to the file include chars like ü,ä
    # (with) makes sure the file stream is closed after using it
    with open(file_path, encoding='unicode_escape') as file_stream:
        headers_array = np.array(pd.read_csv(file_stream, delimiter=delimiter, decimal=decimal, nrows=1, header=None))[0]
    for i in range(len(headers_array)):
        headers_array[i] = get_valid_file_name(headers_array[i])
    return list(headers_array)
