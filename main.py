import sys

from imageProcesser import read_image, convert_to_gray
from transform import transform_experiment
from quantization import quantization_experiment


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
        pass


if __name__ == '__main__':
    main()
