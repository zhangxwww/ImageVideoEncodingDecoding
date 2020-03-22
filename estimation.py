from imageVideoProcessor import draw_rectangle, save_video
from util import mad

BLOCK_SIZE = 16
REFERENCE_BLOCK_LEFT = 310
REFERENCE_BLOCK_TOP = 220


def motion_estimation_experiment(video):
    n_frame = video.shape[0]
    reference_frame = video[0]
    ref_block = reference_frame[REFERENCE_BLOCK_TOP:REFERENCE_BLOCK_TOP + BLOCK_SIZE,
                REFERENCE_BLOCK_LEFT:REFERENCE_BLOCK_LEFT + BLOCK_SIZE]
    draw_rectangle(
        reference_frame,
        REFERENCE_BLOCK_LEFT,
        REFERENCE_BLOCK_TOP,
        BLOCK_SIZE, BLOCK_SIZE, 'pixel_{}'.format(0)
    )
    last_match = (REFERENCE_BLOCK_TOP, REFERENCE_BLOCK_LEFT)
    for k in range(1, n_frame):
        i, j = pixel_domain_block_matching(ref_block, video[k], last_match)
        draw_rectangle(
            video[k], j, i, BLOCK_SIZE, BLOCK_SIZE, 'pixel_{}'.format(k)
        )
        last_match = (i, j)
    save_video(n_frame, 'pixel')


def pixel_domain_block_matching(ref_block, frame, last_match, R=8):
    last_match_top, last_match_left = last_match
    min_mad = float('inf')
    arg_min_mad = None
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
            m = mad(ref_block, block)
            if m < min_mad:
                min_mad = m
                arg_min_mad = (last_match_top + i, last_match_left + j)
    return arg_min_mad


def compression_domain_block_matching(ref, frame, R=16):
    pass
