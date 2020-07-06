import h5py
import pandas as pd
import numpy as np

from benchmarks.utils import get_avg_impl_time
from benchmarks.constants import hdf_file_pd
from benchmarks.constants import hdf_file_h5py
from benchmarks.constants import hdf_data_key


def load_hdf():
    data = np.array(pd.read_hdf(hdf_file_pd))
    print(data[0, 0])
    print(data[0, 1])
    c = data[:, 0] * data[:, 1]
    print(c[0])

get_avg_impl_time(load_hdf, calc_count=3, print_out=True)

def load_hdf_h5py():
    # hdf5_store = h5py.File(hdf_file_h5py, "a")
    # results = hdf5_store.create_dataset("results", (1000, 1000, 1000, 5), compression="gzip")
    #
    # results[100, 25, 1, 4] = 42

    hdf5_store = h5py.File(hdf_file_h5py, "r")
    print(hdf5_store[hdf_data_key][0,0])


get_avg_impl_time(load_hdf_h5py, calc_count=1, print_out=True)
