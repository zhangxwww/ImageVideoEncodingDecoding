from imageProcesser import read_image, convert_to_gray
from transform import transform_experiment


def main():
    img = read_image('./lena.bmp')
    img = convert_to_gray(img)
    transform_experiment(img)


if __name__ == '__main__':
    main()
