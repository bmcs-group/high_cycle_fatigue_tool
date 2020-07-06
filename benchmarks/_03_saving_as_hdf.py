import h5py
import pandas as pd
from benchmarks.constants import csv_file
from benchmarks.constants import hdf_file_pd
from benchmarks.constants import hdf_file_h5py
from benchmarks.constants import hdf_data_key
from benchmarks.utils import get_avg_impl_time

import numpy as np


df = pd.read_csv(csv_file, delimiter=';')
np_array = np.array(df)

# def save_hdf_as_chunks_using_pd_HDFStore_append():
#     df_cols_to_index = ['first', 'second']  # list of columns (labels) that should be indexed
#     store = pd.HDFStore(hdf_file_pd)
#
#     for chunk in pd.read_csv(csv_file, delimiter=';', chunksize=500000):
#         # don't index data columns in each iteration - we'll do it later ...
#         store.append(hdf_data_key, chunk, data_columns=df_cols_to_index, index=False)
#
#     store.create_table_index(hdf_data_key, columns=df_cols_to_index, optlevel=9, kind='full')
#     store.close()
#
#
# get_avg_impl_time(save_hdf_as_chunks_using_pd_HDFStore_append, calc_count=1, print_out=True)


def save_hdf_using_pd_to_hdf():
    df.to_hdf(hdf_file_pd, hdf_data_key, mode='w', format='table')


get_avg_impl_time(save_hdf_using_pd_to_hdf, calc_count=6, print_out=True)


def save_hdf_using_h5py():
    # hdf5_store.create_dataset("results", a, compression="gzip")
    with h5py.File(hdf_file_h5py, 'w') as hf:
        hf.create_dataset(hdf_data_key, data=np_array)


get_avg_impl_time(save_hdf_using_h5py, calc_count=6, print_out=True)
