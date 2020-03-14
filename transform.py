import numpy as np


def dct1d(f):
    f = f.reshape(-1, 1)
    n = f.shape[0]
    c = np.ones((n,))
    c[0] /= np.sqrt(2)
    u = np.linspace(0, n - 1, n).reshape(1, -1)
    x = np.linspace(0, n - 1, n).reshape(-1, 1)
    cos = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
    F = np.sum(f * cos, axis=1)
    F = F * c * np.sqrt(2 / n)
    return F


def dct2d(f):
    pass


def idct1d(F):
    pass


def idct2d(F):
    pass


def retain(n):
    pass


def transform_experiment(img):

    dct1d(img[0])
    pass
