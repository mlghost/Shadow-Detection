import tensorflow as tf
import layers
import os
import matplotlib.image as mp
import numpy as np
import random
import termcolor
from matplotlib.colors import rgb_to_hsv
from models import *
from losses import *


class IlluminatorNet:
    graph_save = ''
    weight_save = ''
    conditional_detection = True
    DIM = 224
    block = None
    detection_mode = 0
    EPOCH = 50000000
    sd_lr = 1e-4
    BATCH_SIZE = 4
    INIT_LEARNING_RATE = 1e-4

    # mask_path = '../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_mask/'
    # image_path = '../data/SBU/SBUTrain4KRecoveredSmall/aug_shadow_image/'
    # test_image_path = '../data/SBU/SBU-Test/aug_shadow_image/'
    # test_mask_path = '../data/SBU/SBU-Test/aug_shadow_mask/'
    mask_path = '../data/ISTD_Dataset/augmented/train/train_B_224/'
    image_path = '../data/ISTD_Dataset/augmented/train/train_A_224/'
    test_image_path = '../data/ISTD_Dataset/augmented/test/test_A_224/'
    test_mask_path = '../data/ISTD_Dataset/augmented/test/test_B_224/'

    def __init__(self, graph_save='../graph_save/', weight_save='../results/weight/', activation='prelu',
                 block='conv', agg='sum', EPOCH=50000000, BATCH_SIZE=3,
                 conditional_detection=True, color_space='HSV', main_input='RGB',
                 use_gpu=False, DIM=224, detection_mode=0,
                 sd_lr=1e-5, beta1=0.9, beta2=0.999, optimizer='Adam'):

        tf.set_random_seed(1111)
        tf.reset_default_graph()

        self.summary_path = '../results/summary/'
        self.agg = agg
        self.activation = activation
        self.block_type = block
        self.conditional_detection = conditional_detection
        self.color_space = color_space
        self.main_input = main_input
        self.use_gpu = use_gpu
        self.weight_save = weight_save
        self.graph_save = graph_save
        self.DIM = DIM
        self.optimizer = optimizer
        self.sd_lr = sd_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.BATCH_SIZE = BATCH_SIZE

        self.himage = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 3],
                                     name='Input_1')
        self.mask = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 2],
                                   name='Mask')
        self.removed = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 3],
                                      name='Removed')
self    self.net = VGG_ILSVRC_16_layers({'data': self.himage}).netself.net = VGG_ILSVRC_16_layers({'data': self.himage})self.net = VGG_ILSVRC_16_layers({'data': self.himage}) = VGG_ILSVRC_16_layers({'data': self.himage})































































































































