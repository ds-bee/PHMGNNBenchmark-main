import os

import numpy as np
from PIL import Image


def load_imgs(npzfile, path, savepath):
    images = np.load(npzfile)
    i = 0
    for file in images:
        i += 1
        image = np.load(path + '/' +path + '/' + file + '.npy')
        image = Image.fromarray(image)
        image = image.convert('L')
        # image.show()
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        image.save(savepath + '/' + 'array_%d.jpg' % (i))



load_imgs('normal_imgs.npz', 'normal_imgs', 'GCN_imgs/normal_imgs')
load_imgs('ball_18_imgs.npz', 'ball_18_imgs', 'GCN_imgs/ball_18_imgs')
load_imgs('ball_36_imgs.npz', 'ball_36_imgs', 'GCN_imgs/ball_36_imgs')
load_imgs('ball_54_imgs.npz', 'ball_54_imgs', 'GCN_imgs/ball_54_imgs')
load_imgs('inner_18_imgs.npz', 'inner_18_imgs', 'GCN_imgs/inner_18_imgs')
load_imgs('inner_36_imgs.npz', 'inner_36_imgs', 'GCN_imgs/inner_36_imgs')
load_imgs('inner_54_imgs.npz', 'inner_54_imgs', 'GCN_imgs/inner_54_imgs')
load_imgs('outer_18_imgs.npz', 'outer_18_imgs', 'GCN_imgs/outer_18_imgs')
load_imgs('outer_36_imgs.npz', 'outer_36_imgs', 'GCN_imgs/outer_36_imgs')
load_imgs('outer_54_imgs.npz', 'outer_54_imgs', 'GCN_imgs/outer_54_imgs')

