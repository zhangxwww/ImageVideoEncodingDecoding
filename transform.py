import numpy as np

from imageVideoProcessor import convert_to_image, save_image
from util import timing, psnr, generate_image_name


def transform_experiment(img):
    img_f = img.astype(float)

    n, m = img.shape
    N = n * m

    tests = [
        {'fn_t': trial_1, 'fn_r': reconstruct_1},
        {'fn_t': trial_2, 'fn_r': reconstruct_2},
        {'fn_t': trial_3, 'fn_r': reconstruct_3},
    ]

    for ret in [1, 4, 16, 64]:
        print('================')
        print('1/{} coefficients'.format(ret))
        for test in tests:
            t, r = test['fn_t'], test['fn_r']
            print(t.__doc__)
            F = t(img_f, N // ret)
            f = r(F)
            im = convert_to_image(f)
            im_name = generate_image_name(t.__doc__, ret)
            save_image(im, im_name, exp='exp1')
            print('PSNR: {:.3}'.format(psnr(img, f.astype(np.uint8))))
        print()
    print()


@timing
def trial_1(img, ret):
    """1D-DCT on the whole image"""
    return retain(_first_row_then_column(img), ret)


@timing
def trial_2(img, ret):
    """2D-DCT on the whole image"""
    return retain(dct2d(img), ret)


@timing
def trial_3(img, ret):
    """2D-DCT on 8*8 blocks"""
    return _dct_block(img, ret)


def reconstruct_1(F):
    return _inverse_first_row_then_column(F).astype('uint8')


def reconstruct_2(F):
    return idct2d(F).astype('uint8')


def reconstruct_3(F):
    return _idct_block(F).astype('uint8')


_dct_c = {}
_dct_cos = {}


def dct1d(f):
    f = f.reshape(-1, 1)
    n = f.shape[0]
    global _dct_c, _dct_cos
    if n in _dct_c.keys():
        c = _dct_c[n]
    else:
        c = np.ones((n,))
        c[0] /= np.sqrt(2)
        _dct_c[n] = c
    if n in _dct_cos.keys():
        cos = _dct_cos[n]
    else:
        u = np.linspace(0, n - 1, n).reshape(1, -1)
        x = np.linspace(0, n - 1, n).reshape(-1, 1)
        cos = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
        _dct_cos[n] = cos
    F = np.sum(f * cos, axis=0)
    F = F * c * np.sqrt(2 / n)
    return F


_idct_c = {}
_idct_cos = {}


def idct1d(F):
    F = F.reshape(-1, 1)
    n = F.shape[0]
    if n in _idct_c.keys():
        c = _idct_c[n]
    else:
        c = np.ones((n, 1))
        c[0, 0] /= np.sqrt(2)
        _idct_c[n] = c
    if n in _idct_cos.keys():
        cos = _idct_cos[n]
    else:
        u = np.linspace(0, n - 1, n).reshape(-1, 1)
        x = np.linspace(0, n - 1, n).reshape(1, -1)
        cos = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
        _idct_cos[n] = cos
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


_u = {}

def uniform_retain(p, ret):
    p = p.reshape((-1,))
    n = p.shape[0]
    global _u
    if (n, ret) in _u.keys():
        index = _u[(n, ret)]
    else:
        index = np.random.choice(n, ret, replace=False)
        _u[(n, ret)] = index
    return p[index]


_a = {}


def _A(n):
    global _a
    if n in _a.keys():
        return _a[n]
    c = np.ones((1, n)) * np.sqrt(2 / n)
    c[0, 0] = np.sqrt(1 / n)
    i = np.linspace(0, n - 1, n).reshape(1, -1)
    j = np.linspace(0, n - 1, n).reshape(-1, 1)
    A = c * np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    _a[n] = A
    return A


_zz = {}


def _zigzag(n):
    global _zz
    if n in _zz.keys():
        return _zz[n]
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
    _zz[n] = a
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


def _dct_block(img, ret):
    n_block_rows = img.shape[0] // 8
    n_block_cols = img.shape[1] // 8
    res = np.zeros_like(img)
    r = ret // (64 * 64)
    for i in range(n_block_rows):
        for j in range(n_block_cols):
            tmp = dct2d(img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            res[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = retain(tmp, r)
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
