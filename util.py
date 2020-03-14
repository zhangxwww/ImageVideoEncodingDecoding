import time
import numpy as np

def timing(name):
    def g(f):
        def h(*args, **kwargs):
            start = time.time()
            res = f(*args, **kwargs)
            end = time.time()
            delta = end - start
            print('{} costs {:.3}s'.format(name, delta))
            return res
        return h
    return g

def mse(p1, p2):
    return np.mean((p1 - p2) ** 2)

def psnr(p1, p2):
    res = 2 * np.log2(255) - np.log2(mse(p1, p2))
    return res * 10
