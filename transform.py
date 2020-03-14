import numpy as np


def dct1d(f):
    f = f.reshape(-1, 1)
    n = f.shape[0]
    c = np.ones((n,))
    c[0] /= np.sqrt(2)
    u = np.linspace(0, n - 1, n).reshape(1, -1)
    x = np.linspace(0, n - 1, n).reshape(-1, 1)
    cos = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
    F = np.sum(f * cos, axis=0)
    F = F * c * np.sqrt(2 / n)
    return F


def dct2d(f):
    pass


def idct1d(F):
    F = F.reshape(-1, 1)
    n = F.shape[0]
    c = np.ones((n, 1))
    c[0, 0] /= np.sqrt(2)
    u = np.linspace(0, n - 1, n).reshape(-1, 1)
    x = np.linspace(0, n - 1, n).reshape(1, -1)
    cos = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
    f = np.sum(c * F * cos, axis=0)
    f = f * np.sqrt(2 / n)
    return f


def idct2d(F):
    pass


def retain(n):
    pass


def transform_experiment(img):
    f = img[0]
    dct = dct1d(f)
    idct = idct1d(dct)
    print(f - idct)
    pass
