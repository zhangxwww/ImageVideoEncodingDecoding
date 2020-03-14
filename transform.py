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


def dct2d(f):
    n = f.shape[0]
    A = _A(n)
    return A.dot(f).dot(A.T)


def idct2d(F):
    n = F.shape[0]
    A = _A(n)
    return A.T.dot(F).dot(A)


def _A(n):
    c = np.ones((1, n)) * np.sqrt(2 / n)
    c[0, 0] = np.sqrt(1 / n)
    i = np.linspace(0, n - 1, n).reshape(1, -1)
    j = np.linspace(0, n - 1, n).reshape(-1, 1)
    A = c * np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    return A


def retain(F, n):
    mask = np.zeros_like(F)
    if F.ndim == 2:
        mask = _zigzag(F.shape[0]) < n + 0
    else:
        mask[:n] = 1
    return F * mask


def _zigzag(n):
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = i + j
            if s < n:
                a[i, j] = s * (s + 1) // 2
                a[i, j] += j if s % 2 == 0 else i
            else:
                s = 2 * n - 2 - i - j
                a[i, j] = n * n - s * (s + 1) // 2 - n
                a[i, j] += j if s % 2 == 0 else i
    return a


def transform_experiment(img):
    f = img[0]
    dct = dct1d(f)
    dct = retain(dct, 4)
    idct = idct1d(dct)
    print((f - idct).sum())

    dct = dct2d(img)
    dct = retain(dct, 64 * 64)
    idct = idct2d(dct)

    print((img - idct).sum())
    pass
