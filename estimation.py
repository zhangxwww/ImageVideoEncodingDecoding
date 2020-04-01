import numpy as np

from imageVideoProcessor import draw_rectangle, save_video
from transform import dct2d, retain, uniform_retain
from util import mad, mse, plot_curve, timing

BLOCK_SIZE = 16
REFERENCE_BLOCK_LEFT = 310
REFERENCE_BLOCK_TOP = 220

RETAIN = BLOCK_SIZE * BLOCK_SIZE // 16


def motion_estimation_experiment(video):
    n_frames = video.shape[0]
    print('================')
    print('Pixel domain block matching')
    mse_p, ms = block_matching(video.copy(), pixel=True)
    save_result(video.copy(), 'pixel', ms)
    print('Compression domain block matching')
    mse_c, ms = block_matching(video.copy(), pixel=False)
    save_result(video.copy(), 'compression', ms)
    print('Pixel domain block matching with part pixels')
    mse_p_r, ms = block_matching(video.copy(), RETAIN, pixel=True)
    save_result(video.copy(), 'pixel_part', ms)
    print('Compression domain block matching with part coefficients')
    mse_c_r, ms = block_matching(video.copy(), RETAIN, pixel=False)
    save_result(video.copy(), 'compression_part', ms)
    plot_curve(
        x=np.linspace(0, n_frames - 2, n_frames - 1).astype(int),
        y=[mse_p, mse_c, mse_p_r, mse_c_r],
        x_label='frame', y_label='mse',
        title='MSE curve',
        legend=(
            'Pixel domain',
            'Compression domain',
            'Pixel domain with 1/16 pixels',
            'Compression domain with 1/16 coefficients'),
        major=5,
        filename='MSE-frame'
    )


@timing
def block_matching(video, ret=-1, pixel=True):
    n_frame = video.shape[0]
    reference_frame = video[0]
    ref_block = reference_frame[REFERENCE_BLOCK_TOP:REFERENCE_BLOCK_TOP + BLOCK_SIZE,
                REFERENCE_BLOCK_LEFT:REFERENCE_BLOCK_LEFT + BLOCK_SIZE]
    last_match = (REFERENCE_BLOCK_TOP, REFERENCE_BLOCK_LEFT)
    mse_curve = []
    last_matches = []
    for k in range(1, n_frame):
        if pixel:
            trans = (lambda x: uniform_retain(x, ret)) if ret > 0 else None
        else:
            trans = (lambda x: retain(dct2d(x), ret)) if ret > 0 else dct2d
        last, mse_ = match(ref_block, video[k], last_match, trans=trans)
        i, j = last
        last_matches.append((i, j, k))
        last_match = (i, j)
        mse_curve.append(mse_)
    return np.array(mse_curve), last_matches


def match(ref_block, frame, last_match, R=8, trans=None):
    ref = (trans(ref_block)) if trans else ref_block
    last_match_top, last_match_left = last_match
    min_mad = float('inf')
    arg_min_mad = None
    mse_ = 0
    for i in range(-R, R + 1):
        if last_match_top + i < 0 \
                or last_match_top + i + BLOCK_SIZE >= frame.shape[0]:
            continue
        for j in range(-R, R + 1):
            if last_match_left + j < 0 \
                    or last_match_left + j + BLOCK_SIZE >= frame.shape[1]:
                continue
            block = frame[last_match_top + i: last_match_top + i + BLOCK_SIZE,
                    last_match_left + j:last_match_left + j + BLOCK_SIZE]
            if trans:
                m = mad(ref, trans(block))
            else:
                m = mad(ref_block, block)
            if m < min_mad:
                min_mad = m
                arg_min_mad = (last_match_top + i, last_match_left + j)
                mse_ = mse(ref_block, block)
    return arg_min_mad, mse_


def save_result(video, name, matches):
    n_frame = video.shape[0]
    reference_frame = video[0]
    draw_rectangle(
        reference_frame,
        REFERENCE_BLOCK_LEFT,
        REFERENCE_BLOCK_TOP,
        BLOCK_SIZE, BLOCK_SIZE, '{}_{}'.format(name, 0)
    )
    for i, j, k in matches:
        draw_rectangle(video[k], j, i, BLOCK_SIZE, BLOCK_SIZE, '{}_{}'.format(name, k))
    save_video(n_frame, name, exp='exp3')
