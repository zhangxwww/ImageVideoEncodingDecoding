from imageVideoProcessor import draw_rectangle

BLOCK_SIZE = 16
REFERENCE_BLOCK_LEFT = 182
REFERENCE_BLOCK_TOP = 165


def motion_estimation_experiment(video):
    draw_reference_block(video[0])


def draw_reference_block(frame):
    draw_rectangle(frame,
                   REFERENCE_BLOCK_LEFT,
                   REFERENCE_BLOCK_TOP,
                   BLOCK_SIZE, BLOCK_SIZE)
