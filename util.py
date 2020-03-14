import time
import numpy as np


def timing(f):
    def g(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        delta = end - start
        print('{} costs {:.3}s'.format(f.__doc__, delta))
        return res

    return g


def mse(p1, p2):
    return np.mean((p1 - p2) ** 2)


def psnr(p1, p2):
    res = 2 * np.log10(255) - np.log10(mse(p1, p2))
    return res * 10
