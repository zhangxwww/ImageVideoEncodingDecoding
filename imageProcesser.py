import os
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
    return img

def save_image(img, filename):
    save_dir = './img'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, filename)
    img.save(save_path)
