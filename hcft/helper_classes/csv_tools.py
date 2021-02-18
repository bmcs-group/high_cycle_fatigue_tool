import numpy as np
import pandas as pd
from .files_tools import get_valid_file_name


def get_headers(file, delimiter, decimal):
    headers_array = np.array(pd.read_csv(file, delimiter=delimiter, decimal=decimal, nrows=1, header=None))[0]
    for i in range(len(headers_array)):
        headers_array[i] = get_valid_file_name(headers_array[i])
    return list(headers_array)
