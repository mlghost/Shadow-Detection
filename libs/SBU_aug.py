import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mp
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
from skimage.transform import rotate,resize

from PIL import Image
import cv2
from scipy.misc import imsave
from PIL import Image

base = '../data/SBU/SBU-Test/ShadowImages/'
for image_name in os.listdir(base):
        print (image_name)
        image = Image.open(base + image_name)
        image = image.resize((224,224))
        r90 = image.rotate(90)
        r180 = image.rotate(180)
        r270 = image.rotate(270)
        # plt.hist(np.array(image).ravel())
        # plt.show()
        # plt.hist(np.array(r90).ravel())
        # plt.show()
        base_name = image_name[:-4]
        # if 'B' not in dir:
        # mp.imsave('../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_image/' + base_name + '.JPG', image, format='JPG')
        # mp.imsave('../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_image/' + base_name + '_90' + '.JPG', r90,
        #               format='JPG')
        # mp.imsave('../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_image/' + base_name + '_180' + '.JPG', r180,
        #               format='JPG')
        # mp.imsave('../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_image/' + base_name + '_270' + '.JPG', r270,
        #               format='JPG')
        # else:
        image.save('../data/SBU/SBU-Test/aug_shadow_image/' +  base_name + '.JPG')
        r90.save('../data/SBU/SBU-Test/aug_shadow_image/' + base_name + '_90.JPG')
        r180.save('../data/SBU/SBU-Test/aug_shadow_image/' +  base_name + '_180.JPG')
        r270.save('../data/SBU/SBU-Test/aug_shadow_image/' + base_name + '_270.JPG')
# plt.imread