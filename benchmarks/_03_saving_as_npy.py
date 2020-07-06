import pandas as pd
import numpy as np

from benchmarks.utils import get_avg_impl_time
from benchmarks.constants import csv_file
from benchmarks.constants import npy_file

data = np.array(pd.read_csv(csv_file, delimiter=';'))


def save_npy():
    np.save(npy_file, data)


impl_time = get_avg_impl_time(save_npy, calc_count=3, print_out=True)

print('Saving numpy array to disk using np.save()= ' + str(impl_time))
