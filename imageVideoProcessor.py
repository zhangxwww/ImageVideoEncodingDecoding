import os
from PIL import Image
import numpy as np
import cv2

from util import get_save_dir


def read_image(filename):
    img = Image.open(filename)
    return img


def convert_to_gray(img):
    img = img.convert('L')
    return np.array(img)


def convert_to_image(img):
    img = Image.fromarray(img.astype('uint8'))
    return img


def save_image(img, filename, exp=None):
    save_dir = get_save_dir(exp)
    save_path = os.path.join(save_dir, filename)
    img.save(save_path)


def read_video(filename, frame_range):
    cap = cv2.VideoCapture(filename)
    start, end = frame_range
    cur = 0
    video = []
    s = False
    while True:
        if cur == end:
            break
        elif cur == start:
            s = True
        ret, frame = cap.read()
        if not ret:
            break
        if s:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video.append(img_gray)
        cur += 1
    return np.stack(video, axis=0)


def draw_rectangle(frame, x, y, w, h, index):
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    save_image(convert_to_image(frame), '{}.png'.format(index))
    return frame


def save_video(n_frame, type_, exp=None):
    save_dir = get_save_dir(exp)
    save_path = os.path.join(save_dir, '{}.avi'.format(type_))
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (352, 288))
    for i in range(n_frame):
        img_name = './result/{}_{}.png'.format(type_, i)
        f = cv2.imread(img_name)
        os.remove(img_name)
        video_writer.write(f)
    video_writer.release()
