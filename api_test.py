from hcft.api import HCFT
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import os
from scipy.signal import savgol_filter




def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x.reshape(1,1), y.reshape(1,1)



if __name__ == '__main__':

    files = ['CT80-40_13713_Zykl']
    # files = ['CT80-39_6322_Zykl']

    for l in files:

        hcft = HCFT()

        home_dir = os.path.expanduser('~')

        path = os.path.join(home_dir, 'Escritorio\\NPY\\')

       # path = os.path.join(
       #      'F:/' , 'C80 Charge 1/NPY/')

        # Load force array and generate max and min of it
        force_array = np.array(np.load(os.path.join(path, l + '_Kraft.npy')))#*(1000/(0.25*np.pi*150**2))

        force_max_indices, force_min_indices = hcft.get_array_max_and_min_indices(force_array.flatten())

        # Optional (if you want max and min indicies in one array)
        force_max_min_indices = np.concatenate((force_min_indices, force_max_indices))
        force_max_min_indices.sort()

        # Cut the fake cycles because of noise (important!)
        force_max_indices_cut, force_min_indices_cut = np.array(hcft.cut_indices_of_defined_range(force_array.flatten(),
                                                                                          force_max_indices,
                                                                                         force_min_indices,
                                                                                          15))
        force_max_indices_cut = force_max_indices_cut
        force_min_indices_cut = force_min_indices_cut
        # force_max_indices_cut = np.array(
        #    [331, 581, 829, 1082, 1333, 1583, 1832, 2080, 2333, 2585, 2836, 3084, 3334, 3584, 3835, 4083, 4333, 4583, 4833,
        #     5083, 5335, 5586, 5837, 6088, 6338, 6585, 6836, 7086, 7335, 7586, 7838, 8090, 8338, 8589, 8839, 9088, 9340,
        #     9588, 9838, 10088, 10340, 10589, 10841, 11090, 11341, 11591, 11840, 12080])
        #
        # force_min_indices_cut = np.floor(
        #    (force_max_indices_cut[1:] - force_max_indices_cut[:-1]) / 2 + force_max_indices_cut[:-1]).astype(int)

        disp_meass = '_WA_1'

        # Load displacement array and get the values corresponding to max and min of the force
        disp_array = np.array(np.load(os.path.join(path, l + disp_meass + '.npy')))#/300
        # array_rest_maxima = disp_array.flatten()[force_max_indices_cut]
        # array_rest_minima = disp_array.flatten()[force_min_indices_cut]

        # Initialization
        x_descending = []
        x_ascending = []
        y_descending = []
        y_ascending = []
        inter_descending = []
        inter_ascending = []
        dissipated_energy = [0]

        disp_array = np.abs(disp_array.reshape(len(disp_array),1))
        force_array = np.abs(force_array.reshape(len(force_array),1))

        # Dividing the experimental data in ascending and descending branches
        if force_max_indices_cut[0] > force_min_indices_cut[0]:
            force_min_indices_cut = np.delete(force_min_indices_cut,0)
        for i in range(len(force_max_indices_cut) - 1):
            x_descending.append(disp_array[force_max_indices_cut[i]-1:force_min_indices_cut[i]+1].tolist())
            x_ascending.append(disp_array[force_min_indices_cut[i]:force_max_indices_cut[i + 1]+2].tolist())
            y_descending.append(force_array[force_max_indices_cut[i]-1:force_min_indices_cut[i]+1].tolist())
            y_ascending.append(force_array[force_min_indices_cut[i]:force_max_indices_cut[i + 1]+2].tolist())
        warnings=0
        # Looping over each loading-unloading cycle
        for k in range(len(force_max_indices_cut) - 2):
            intersect = 0
            # plt.figure()
            # plt.plot(x_descending[k], y_descending[k])
            # plt.plot(x_ascending[k], y_ascending[k])
            # plt.show()
            for i in range(len(x_descending[k]) - 1):  # index i for i-th segment of descending branch
                vector1 = np.array([
                    np.array(x_descending[k][i + 1]) - np.array(x_descending[k][i]), np.array(y_descending[k][i + 1]) - np.array(y_descending[k][i])]).reshape(-1)
                # 1st of the 6 needed vectors for mathematical condition, p-p+r from example
                for j in range(1, len(x_ascending[k])):
                    vector2 = np.array([np.array(x_ascending[k][len(x_ascending[k]) - j]) - np.array(np.array(x_descending[k][i + 1])),
                                        np.array(y_ascending[k][len(y_ascending[k]) - j]) - np.array(y_descending[k][i + 1])]).reshape(-1)
                    vector3 = np.array([np.array(x_ascending[k][len(x_ascending[k]) - j - 1]) - np.array(x_descending[k][i + 1]),
                                        np.array(y_ascending[k][len(y_ascending[k]) - j - 1]) - np.array(y_descending[k][i + 1])]).reshape(-1)
                    vector4 = np.array(
                        [np.array(x_ascending[k][len(x_ascending[k]) - j - 1]) - np.array(x_ascending[k][len(x_ascending[k]) - j]),
                         np.array(y_ascending[k][len(y_ascending[k]) - j - 1]) - np.array(y_ascending[k][len(y_ascending[k]) - j])]).reshape(-1)
                    # q-q+s vector from example
                    vector5 = np.array([np.array(x_descending[k][i]) - np.array(x_ascending[k][len(x_ascending[k]) - j - 1]),
                                        np.array(y_descending[k][i]) - np.array(y_ascending[k][len(y_ascending[k]) - j - 1])]).reshape(-1)
                    vector6 = np.array([np.array(x_descending[k][i + 1]) - np.array(x_ascending[k][len(x_ascending[k]) - j - 1]),
                                        np.array(y_descending[k][i + 1]) - np.array(y_ascending[k][len(y_ascending[k]) - j - 1])]).reshape(-1)
                    d1 = np.cross(vector1, vector2)
                    d2 = np.cross(vector1, vector3)
                    d3 = np.cross(vector4, vector5)
                    d4 = np.cross(vector4, vector6)
                    # Mathematical condition
                    if (np.sign(d1) != np.sign(d2)) & (np.sign(d3) != np.sign(d4)):
                        intersect = 1
                        break
                if intersect == 1:
                    break

            if intersect == 1:
                # Intersection between A,B - C,D
                A = np.array([x_descending[k][i], y_descending[k][i]])
                B = np.array([x_descending[k][i + 1], y_descending[k][i + 1]])
                C = np.array([x_ascending[k][len(x_ascending[k]) - j], y_ascending[k][len(x_ascending[k]) - j]])
                D = np.array([x_ascending[k][len(x_ascending[k]) - j - 1], y_ascending[k][len(x_ascending[k]) - j - 1]])
                inter_descending.append([A, B])
                inter_ascending.append([C, D])
                # x and y coordinates of the intersection
                inter_x, inter_y = line_intersection((A, B), (C, D))

               # # Plotting loop
               # plt.figure()
               # plt.plot(np.concatenate((x_descending[k], x_ascending[k])), np.concatenate((y_descending[k], y_ascending[k])))
               # # Plotting intersection segment from descending branch
               # plt.plot(x_descending[k][i], y_descending[k][i], 'ro')
               # plt.plot(x_descending[k][i + 1], y_descending[k][i + 1], 'ro')
               # # Plotting intersection and intersection segment from ascending branch
               # plt.plot(inter_x, inter_y, 'yo')
               # plt.plot(x_ascending[k][len(x_ascending[k]) - j], y_ascending[k][len(x_ascending[k]) - j], 'go')
               # plt.plot(x_ascending[k][len(x_ascending[k]) - j - 1], y_ascending[k][len(x_ascending[k]) - j - 1], 'go')
               # plt.show()

                # Creating array containing just the loop

                x = np.concatenate(
                    ([inter_x, np.array(x_descending[k][i + 1:]), np.array(x_ascending[k][:(len(x_ascending[k]) - j)]),
                     inter_x]),axis=0)
                y = np.concatenate(
                    ([inter_y, np.array(y_descending[k][i + 1:]), np.array(y_ascending[k][:(len(x_ascending[k]) - j )]),
                     inter_y]),axis=0)

                int = np.abs(scipy.integrate.cumtrapz(y.reshape(-1), x.reshape(-1)))

                if int[-1]<1e-5:
                    dissipated_energy.append(np.array(dissipated_energy[-1]))
                    warnings += 1
                else:
                    dissipated_energy.append(np.array(int[-1]))
            else:
                dissipated_energy.append(np.array(dissipated_energy[-1]))

        print(warnings)


        def savitzky_golay(y, window_size, order, deriv=0, rate=1):
            r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
            The Savitzky-Golay filter removes high frequency noise from data.
            It has the advantage of preserving the original shape and
            features of the signal better than other types of filtering
            approaches, such as moving averages techniques.
            Parameters
            ----------
            y : array_like, shape (N,)
                the values of the time history of the signal.
            window_size : int
                the length of the window. Must be an odd integer number.
            order : int
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.
            deriv: int
                the order of the derivative to compute (default = 0 means only smoothing)
            Returns
            -------
            ys : ndarray, shape (N)
                the smoothed signal (or it's n-th derivative).
            Notes
            -----
            The Savitzky-Golay is a type of low-pass filter, particularly
            suited for smoothing noisy data. The main idea behind this
            approach is to make for each point a least-square fit with a
            polynomial of high order over a odd-sized window centered at
            the point.
            Examples
            --------
            t = np.linspace(-4, 4, 500)
            y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
            ysg = savitzky_golay(y, window_size=31, order=4)
            import matplotlib.pyplot as plt
            plt.plot(t, y, label='Noisy signal')
            plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
            plt.plot(t, ysg, 'r', label='Filtered signal')
            plt.legend()
            plt.show()
            References
            ----------
            .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
               Data by Simplified Least Squares Procedures. Analytical
               Chemistry, 1964, 36 (8), pp 1627-1639.
            .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
               W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
               Cambridge University Press ISBN-13: 9780521880688
            """

            from math import factorial

            # try:
            #     window_size = np.abs(np.int(window_size))
            #     order = np.abs(np.int(order))
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order + 1)
            half_window = (window_size - 1) // 2
            # precompute coefficients
            b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
            m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
            lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve(m[::-1], y, mode='valid')

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot((np.arange(len(dissipated_energy[1:])))/len(np.arange(len(dissipated_energy[1:])))
                ,dissipated_energy[1:], linewidth=2.5)

        ax.set_xlabel(r'N', fontsize=25)
        ax.set_ylabel(r'J/$m^3$, per loop area', fontsize=25)
        plt.savefig(path + l + disp_meass + 'energy_loop.pdf')
        plt.show()

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot((np.arange(len(dissipated_energy[1:]))),
                np.cumsum(dissipated_energy[1:]), linewidth=2.5)

        ax.set_xlabel(r'N', fontsize=25)
        ax.set_ylabel(r'J/$cm^3$, accumulated area', fontsize=25)
        plt.savefig(path + l + disp_meass + 'energy_accumulated.pdf')
        plt.show()

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot(disp_array ,force_array, linewidth=2.5)

        ax.set_xlabel(r'mm', fontsize=25)
        ax.set_ylabel(r'kN', fontsize=25)
        plt.savefig(path + l + disp_meass + 'stress-strain.pdf')
        plt.show()

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot((np.arange(len(disp_array[force_max_indices_cut])))/len(disp_array[force_max_indices_cut]),
                disp_array[force_max_indices_cut], linewidth=2.5)
        ax.plot((np.arange(len(disp_array[force_min_indices_cut])))/len(disp_array[force_min_indices_cut]),
                disp_array[force_min_indices_cut], linewidth=2.5)

        ax.set_xlabel(r'N/$N_f$', fontsize=25)
        ax.set_ylabel(r'kN', fontsize=25)
        plt.savefig(path + l + disp_meass +'fatigue-creep.pdf')
        plt.show()

        print(np.cumsum(dissipated_energy[1:])[-1])