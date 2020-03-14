from PIL import Image
import numpy as np


def read_image(filename):
    img = Image.open(filename)
    return img


def convert_to_gray(img):
    img = img.convert('L')
    return np.array(img)

def convert_to_image(img):
    img = Image.fromarray(img.astype('uint8'))
    img.show()
    return img
