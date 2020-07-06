""" This file compares the methods:
- pd.read_csv (with and without performing np.array(pd.read_csv(..)))
- np.genfromtxt
- np.loadtxt
for importing the csv file into a numpy array
"""

import pandas as pd
import numpy as np

from benchmarks.utils import get_avg_impl_time
from benchmarks.constants import csv_file

csv_file = csv_file

# -------------------------------------------------------------------
# Test the time of importing a csv file using read_csv of pandas (this has many additional features like chunks and decimal)
# -------------------------------------------------------------------
# 1 - Checking if np.array(pd.read_csv(..)) causes an overhead


def load_csv_using_pd_read_csv():
    a = pd.read_csv(csv_file, delimiter=';', usecols=[0])


def load_csv_using_pd_read_csv_with_np_array():
    a = np.array(pd.read_csv(csv_file, delimiter=';', usecols=[0]))


impl_time_1 = get_avg_impl_time(load_csv_using_pd_read_csv, print_out=True)
impl_time_2 = get_avg_impl_time(load_csv_using_pd_read_csv_with_np_array, print_out=True)
print('Reading csv file using read_csv= ' + str(impl_time_1))
print('Reading csv file using read_csv with initializing numpy array= ' + str(impl_time_2))

# Result: np.array(pd.read_csv(..)) doesn't cause any overhead

# ------------------------------------------------------------------------------
# Test the time of importing a csv file using genfromtxt of numpy
# ------------------------------------------------------------------------------


def load_csv_using_np_genfromtxt():
    a = np.genfromtxt(csv_file, delimiter=';', usecols=[0], dtype='bytes_')


impl_time = get_avg_impl_time(load_csv_using_np_genfromtxt, print_out=True)
print('Reading csv file using genfromtxt= ' + str(impl_time))

# ------------------------------------------------------------------------------
# Test the time of importing a csv file using loadtxt of numpy
# ------------------------------------------------------------------------------


def load_csv_using_np_loadtxt():
    a = np.loadtxt(csv_file, delimiter=';', usecols=[0], dtype='bytes_')


impl_time = get_avg_impl_time(load_csv_using_np_loadtxt, print_out=True)
print('Reading csv file using loadtxt= ' + str(impl_time))

