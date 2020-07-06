import numpy as np

from benchmarks.utils import get_avg_impl_time
from benchmarks.constants import npy_file


def load_npy():
    data = np.load(npy_file)
    print(data[0, 0])
    print(data[0, 1])
    c = data[:, 0] * data[:, 1]
    print(c[0])


get_avg_impl_time(load_npy, calc_count=1, print_out=True)


# Average (load_npy) impl time = 0.41626343727111814 sec



def load_npy_and_multiply():
    data = np.load(npy_file)
    print(data[0, 0])
    print(data[0, 1])
    c = data[:, 0] * data[:, 1]
    print(c[0])


get_avg_impl_time(load_npy_and_multiply, calc_count=1, print_out=True)
