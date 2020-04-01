import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)


def timing(f):
    def g(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        delta = end - start
        print('Time: {}s'.format(delta))
        return res

    g.__doc__ = f.__doc__
    return g


def mse(p1, p2):
    return np.mean((p1.astype(float) - p2.astype(float)) ** 2)


def psnr(p1, p2):
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


def plot_curve(x, y, x_label, y_label, title, legend=None, major=1, filename=''):
    ax = plt.gca()
    color = ['#1F77B4', '#FF7F0E', '#FFBB78', '#D62728']
    if isinstance(y, list) or y.ndim > 1:
        for yy, c in zip(y, color):
            sns.lineplot(x=x, y=yy, color=c)
    else:
        ax = sns.lineplot(x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(plt.MultipleLocator(major))
    if legend:
        ax.legend(legend)
    save_dir = get_save_dir()
    plt.savefig(os.path.join(save_dir, filename))
    # plt.show()


def mad(p1, p2):
    return np.abs(p1.astype(float) - p2.astype(float)).mean()

def get_save_dir(exp=None):
    save_dir = './result'
    if exp:
        save_dir = os.path.join(save_dir, exp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
