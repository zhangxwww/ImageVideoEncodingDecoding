import sys

from imageVideoProcessor import read_image, convert_to_gray, read_video
from transform import transform_experiment
from quantization import quantization_experiment
from estimation import motion_estimation_experiment


def main():
    exp = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else [1, 2, 3]
    if 1 in exp:
        img = read_image('./lena.bmp')
        img = convert_to_gray(img)
        transform_experiment(img)
    if 2 in exp:
        img = read_image('./lena.bmp')
        img = convert_to_gray(img)
        quantization_experiment(img)
    if 3 in exp:
        video = read_video('./cars.avi', (0, 60))
        motion_estimation_experiment(video)


if __name__ == '__main__':
    main()
