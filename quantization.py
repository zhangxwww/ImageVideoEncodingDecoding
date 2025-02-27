import numpy as np

from transform import dct2d, idct2d
from util import psnr, plot_curve


def quantization_experiment(img):
    img = img.astype(float)
    n, m = img.shape
    n_block_rows, n_block_cols = n // 8, m // 8
    n_blocks = n_block_cols * n_block_rows

    q = {
        'Q': Q_,
        'Cannon': Cannon_,
        'Nikon': Nikon_,
        'Mine': Mine_
    }

    all_psnr = np.zeros((len(q), n_blocks, 20))
    for idx, block in enumerate(block_generator(img, n_block_rows, n_block_cols)):
        for iidx, alpha in enumerate(np.linspace(0.1, 2, 20)):
            for iiidx, qq in enumerate(q.values()):
                psnr_, f_ = process_block(block, alpha * qq)
                all_psnr[iiidx, idx, iidx] = psnr_
    psnr_curve = all_psnr.mean(axis=1)
    print('================')
    print('Average PSNR over all blocks with a=1:')
    for idx, item in enumerate(q.keys()):
        print('{}: {:.3}'.format(item, psnr_curve[idx, 9]))
        print()

    plot_curve(
        x=list(map('{:.1f}'.format, np.linspace(0.1, 2, 20))),
        y=psnr_curve,
        x_label='a', y_label='PSNR',
        title='a-PSNR curve',
        legend=q.keys(),
        filename='a-PSNR'
    )


def block_generator(img, n, m):
    for i in range(n):
        for j in range(m):
            yield img[i * 8: (i + 1) * 8, j * 8:(j + 1) * 8]


def process_block(block, Q):
    F = dct2d(block)
    quant = np.round(F / Q)
    F = quant * Q
    f = idct2d(F.astype(float))
    return psnr(block, f), f


Cannon_ = np.array([
    [1, 1, 1, 2, 3, 6, 8, 10],
    [1, 1, 2, 3, 4, 8, 9, 8],
    [2, 2, 2, 3, 6, 8, 10, 8],
    [2, 2, 3, 4, 7, 12, 11, 9],
    [3, 3, 8, 11, 10, 16, 15, 11],
    [3, 5, 8, 10, 12, 15, 16, 13],
    [7, 10, 11, 12, 15, 17, 17, 14],
    [14, 13, 13, 15, 15, 14, 14, 14],
])

Nikon_ = np.array([
    [2, 1, 1, 2, 3, 5, 6, 7],
    [1, 1, 2, 2, 3, 7, 7, 7],
    [2, 2, 2, 3, 5, 7, 8, 7],
    [2, 2, 3, 3, 6, 10, 10, 7],
    [2, 3, 4, 7, 8, 13, 12, 9],
    [3, 4, 7, 8, 10, 12, 14, 11],
    [6, 8, 9, 10, 12, 15, 14, 12],
    [9, 11, 11, 12, 13, 12, 12, 12],
])

Q_ = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

Mine_ = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 9],
    [3, 4, 5, 6, 7, 8, 9, 10],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [5, 6, 7, 8, 9, 10, 11, 12],
    [6, 7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 12, 13, 14, 15]
])
