import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(color_codes=True)


def timing(f):
    def g(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        delta = end - start
        print('Time: {:.3}s'.format(delta))
        return res

    g.__doc__ = f.__doc__
    return g


def mse(p1, p2):
    return np.mean((p1 - p2) ** 2)


def psnr(p1, p2):
    if p1.dtype != np.uint8:
        p1 = p1.astype(np.uint8)
    if p2.dtype != np.uint8:
        p2 = p2.astype(np.uint8)
    res = 2 * np.log10(255) - np.log10(mse(p1, p2))
    return res * 10


def generate_image_name(test_name, retains):
    template = '{name}_{retain}.png'
    fmt_dict = {}
    if test_name[0] == '1':
        fmt_dict['name'] = '1D_whole'
    elif 'whole' in test_name:
        fmt_dict['name'] = '2D_whole'
    else:
        fmt_dict['name'] = '2D_block'
    fmt_dict['retain'] = retains
    return template.format(**fmt_dict)


def plot_curve(x, y, x_label, y_label, title):
    x = list(map('{:.1f}'.format, x))
    ax = sns.pointplot(x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

def mad(p1, p2):
    return np.abs(p1.astype(float) - p2.astype(float)).mean()
