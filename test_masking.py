import matplotlib.pyplot as plt
from skimage.transform import resize
import os

def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap='gray')
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional


test_mask_path = './data/SBU/SBU-Test/aug_shadow_mask/'
for image_name in os.listdir(test_mask_path):
    mask = plt.imread(test_mask_path + image_name)
    m2 = resize(mask,(112,112))
    m3 = resize(mask, (56, 56))
    m4 = resize(mask, (28, 28))
    m5 = resize(mask, (14, 14))
    m6 = resize(mask, (7, 7))
    plot_figures({'image':mask,'112':m2,'56':m3,'28':m4,'14':m5,'7':m6},3,3)
    plt.show()