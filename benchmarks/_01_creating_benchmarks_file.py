import numpy as np
from benchmarks.constants import csv_file


def create_benchmarks_file():
    size = 3 * 10 ** 7
    csv_array = np.random.random((size, 2))

    np.savetxt(csv_file, csv_array, delimiter=";")


if __name__ == '__main__':
    create_benchmarks_file()
