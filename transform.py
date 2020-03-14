import numpy as np

from util import timing, psnr


def transform_experiment(img):
    img_f = img.astype(float)
    from imageProcesser import convert_to_image

    t1 = trial_1(img_f)
    r1 = _inverse_first_row_then_column(t1)
    convert_to_image(r1)
    print(psnr(img, r1.astype(np.uint8)))

    t2 = trial_2(img_f)
    r2 = idct2d(t2)
    convert_to_image(r2)
    print(psnr(img, r2.astype(np.uint8)))

    t3 = trial_3(img_f)
    r3 = _idct_block(t3)
    convert_to_image(r3)
    print(psnr(img, r3.astype(np.uint8)))


@timing('1D-DCT on the whole image')
def trial_1(img):
    return _first_row_then_column(img)


@timing('2D-DCT on the whole image')
def trial_2(img):
    return dct2d(img)


@timing('2D-DCT on 8*8 blocks')
def trial_3(img):
    return _dct_block(img)


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


def retain(F, n):
    mask = _zigzag(F.shape[0]) < n + 0
    return F * mask


def _A(n):
    c = np.ones((1, n)) * np.sqrt(2 / n)
    c[0, 0] = np.sqrt(1 / n)
    i = np.linspace(0, n - 1, n).reshape(1, -1)
    j = np.linspace(0, n - 1, n).reshape(-1, 1)
    A = c * np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    return A


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


def _first_row_then_column(img):
    n, m = img.shape
    rows = np.zeros_like(img)
    for i in range(n):
        rows[i] = dct1d(img[i])
    res = np.zeros_like(img)
    for i in range(m):
        res[:, i] = dct1d(rows[:, i])
    return res


def _inverse_first_row_then_column(F):
    n, m = F.shape
    rows = np.zeros_like(F)
    for i in range(m):
        rows[:, i] = idct1d(F[:, i])
    res = np.zeros_like(F)
    for i in range(n):
        res[i] = idct1d(rows[i])
    return res


def _dct_block(img):
    n_block_rows = img.shape[0] // 8
    n_block_cols = img.shape[1] // 8
    res = np.zeros_like(img)
    for i in range(n_block_rows):
        for j in range(n_block_cols):
            res[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] \
                = dct2d(img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
    return res


def _idct_block(F):
    n_block_rows = F.shape[0] // 8
    n_block_cols = F.shape[1] // 8
    res = np.zeros_like(F)
    for i in range(n_block_rows):
        for j in range(n_block_cols):
            res[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] \
                = idct2d(F[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
    return res
