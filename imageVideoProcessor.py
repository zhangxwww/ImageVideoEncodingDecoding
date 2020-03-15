import os
from PIL import Image
import numpy as np
import cv2


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


def read_video(filename, frame_range):
    cap = cv2.VideoCapture(filename)
    start, end = frame_range
    cur = 0
    video = []
    while True:
        if cur == end:
            break
        elif cur == start:
            ret, frame = cap.read()
            if not ret:
                break
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video.append(img_gray)
        cur += 1
    return np.stack(video, axis=0)


def draw_rectangle(frame, x, y, w, h):
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    save_image(convert_to_image(frame), 'reference_block.png')
