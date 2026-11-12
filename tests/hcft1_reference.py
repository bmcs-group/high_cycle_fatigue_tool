"""hcft 1's method, the reference for the core (concept 01, gate G3).

Lifted verbatim from ``hcft/hcft.py`` of v1.1.0 (``0071cf3``), the methods of the class ``HCFT`` with ``self``
removed. Do not edit: the core is checked against these.
"""
import numpy as np


def get_array_max_and_min_indices(input_array):
    # Checking dominant sign
    positive_values_count = np.sum(np.array(input_array) >= 0)
    negative_values_count = input_array.size - positive_values_count

    # Getting max and min indices
    if positive_values_count > negative_values_count:
        force_max_indices = get_max_indices(input_array)
        force_min_indices = get_min_indices(input_array)
    else:
        force_max_indices = get_min_indices(input_array)
        force_min_indices = get_max_indices(input_array)

    return force_max_indices, force_min_indices


def get_plateaus_last_indices(a):
    # Consecutive repeated values (plateaus) are treated as one point represented by its last index
    if a.size == 0:
        return np.array([], dtype=np.intp)
    return np.append(np.flatnonzero(a[1:] != a[:-1]), a.size - 1)


def get_max_indices(a):
    # Vectorized detection of local maxima, a plateau is a max if both of its neighbors are smaller.
    # This method doesn't qualify first and last elements as max
    indices = get_plateaus_last_indices(a)
    values = a[indices]
    is_max = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    return indices[1:-1][is_max]


def get_min_indices(a):
    # Vectorized detection of local minima, a plateau is a min if both of its neighbors are bigger.
    # This method doesn't qualify first and last elements as min
    indices = get_plateaus_last_indices(a)
    values = a[indices]
    is_min = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    return indices[1:-1][is_min]


def cut_indices_of_min_max_range(array, max_indices, min_indices,
                                 range_upper_value, range_lower_value):
    cut_max_indices = max_indices[np.abs(array[max_indices]) > abs(range_upper_value)]
    cut_min_indices = min_indices[np.abs(array[min_indices]) < abs(range_lower_value)]
    return cut_max_indices, cut_min_indices


def cut_indices_of_defined_range(array, max_indices, min_indices, range_):
    # Each max is paired with the min of the same order, only pairs with bigger force difference than range_ are
    # kept
    n = min(max_indices.size, min_indices.size)
    is_full_cycle = np.abs(array[max_indices[:n]] - array[min_indices[:n]]) > range_
    cut_max_indices = max_indices[:n][is_full_cycle]
    cut_min_indices = min_indices[:n][is_full_cycle]

    if max_indices.size > min_indices.size:
        cut_max_indices = np.append(cut_max_indices, max_indices[-1])
    elif min_indices.size > max_indices.size:
        cut_min_indices = np.append(cut_min_indices, min_indices[-1])

    return cut_max_indices, cut_min_indices
