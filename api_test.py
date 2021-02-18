from api import HCFT
import numpy as np

if __name__ == '__main__':
    hcft = HCFT()

    # Load force array and generate max and min of it
    force_array = np.load(replace_this_with_force_npy_file_path).flatten()
    force_max_indices, force_min_indices = hcft.get_array_max_and_min_indices(force_array)

    # Optional (if you want max and min indicies in one array)
    force_max_min_indices = np.concatenate((force_min_indices, force_max_indices))
    force_max_min_indices.sort()

    # Cut the fake cycles because of noise (important!)
    force_max_indices_cut, force_min_indices_cut = hcft.cut_indices_of_defined_range(force_array,
                                                                                     force_max_indices,
                                                                                     force_min_indices,
                                                                                     min_cycle_force_range)

    # Load displacement array and get the values corresponding to max and min of the force
    disp_array = np.load(replace_this_with_displacement_npy_file_path).flatten()
    array_rest_maxima = disp_array[force_max_indices_cut]
    array_rest_minima = disp_array[force_min_indices_cut]