import cv2
import numpy as np
import matplotlib.image as mi
import matplotlib.pyplot as plt
import os
from matplotlib.colors import rgb_to_hsv
from skimage import io, color
from skimage.transform import resize
from PIL import Image
import imageio

# A --> image
# B --> mask
# C --> removed
dir = '../data/ISTD_Dataset/test/test_A'
save_path = '../data/ISTD_Dataset/test/test_A_hsv/'


def rgb2hsv(dir):
    image_names = os.listdir(dir)
    os.makedirs(save_path, exist_ok=True)
    for name in image_names:
        image = mi.imread(dir + name)
        lab = color.rgb2lab(image)
        plt.imshow(lab)
        plt.show()


dirs = [
    '../data/ISTD_Dataset/train/train_A/',
    '../data/ISTD_Dataset/train/train_B',
    '../data/ISTD_Dataset/train/train_C',
    '../data/ISTD_Dataset/train/train_A_hsv',
    '../data/ISTD_Dataset/test/test_A',
    '../data/ISTD_Dataset/test/test_B',
    '../data/ISTD_Dataset/test/test_C',
    '../data/ISTD_Dataset/test/test_A_hsv'
]

rgb2hsv(dirs[0])
