import numpy as np
import time


def get_avg_impl_time(func, calc_count=3, print_out=False):
    impl_times = np.zeros((calc_count))
    for i in range(calc_count):
        start = time.time()
        func()
        end = time.time()
        impl_times[i] = str(end - start)
        if print_out:
            print(str(i) + ' (' + func.__name__ + ') impl time = ' + str(impl_times[i]))

    avg_impl_time = np.average(impl_times)
    if print_out:
        print('Average (' + func.__name__ + ') impl time = ' + str(avg_impl_time) + ' sec')
    return avg_impl_time
