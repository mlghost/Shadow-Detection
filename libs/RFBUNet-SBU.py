import tensorflow as tf
import layers
from vgg16 import VGG_ILSVRC_16_layers
import os
import matplotlib.image as mp
import numpy as np
import random
import termcolor
from matplotlib.colors import rgb_to_hsv
from models import *


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
        self.net = VGG_ILSVRC_16_layers({'data': self.himage})
        self.EPOCH = EPOCH

    @staticmethod
    def _process_labels(label):
        labels = np.zeros((224, 224, 2), dtype=np.float32)
        for i in range(224):
            for j in range(224):
                if label[i][j] == 0:
                    labels[i, j, 0] = 1
                else:
                    labels[i, j, 1] = 1
        return labels

    @staticmethod
    def to_hsv(image):
        # hsv = rgb_to_hsv(image)
        # hsv[..., 2] /= 255
        import matplotlib.pyplot as plt
        # plt.hist(hsv.ravel())
        # plt.show()
        hsv = np.array(image,dtype=np.float32)/255
        # plt.hist(hsv[...,1].ravel())
        # plt.show()
        return hsv

    def build_net(self):

        with tf.name_scope('IlluminatorNet'):
            f1 = self.net.layers['conv1_2']  # 224 * 224 * 64
            f2 = self.net.layers['conv2_2']  # 112 * 112 * 128
            f3 = self.net.layers['conv3_3']  # 56 * 56 * 256
            f4 = self.net.layers['conv4_3']  # 28 * 28 * 512
            mask = RFUNet([f1, f2, f3, f4], 2, 'ShadowDetection')
            tf.summary.image('Logits', tf.reshape(mask[..., 1], [-1, 224, 224, 1]))
            tf.summary.image('Shadow Mask', tf.reshape(self.mask[..., 1], [-1, 224, 224, 1]))

            # matte = RFUNet([f1, f2, f3, f4], 1, 'ShadowRemoval')
        return mask

    def train(self):

        with tf.name_scope('Model'):
            mask_log = self.build_net()
            tf.train.MomentumOptimizer
            sc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=mask_log, labels=self.mask))
            # mask_log = tf.nn.softmax(mask_log, -1)
            # positive_shadow_mask = tf.cast(self.mask[..., 1], tf.bool)
            # negative_shadow_mask = tf.logical_not(positive_shadow_mask)

            # light_obj = tf.reduce_mean(mask_log[..., 0] * self.mask[..., 1])
            #
            # dark_obj = tf.reduce_mean(mask_log[..., 1] * self.mask[..., 0])

            # loss = light_obj * 2 + dark_obj * 5
            # light_obj = tf.boolean_mask(mask_log[..., 0], positive_shadow_mask)
            # light_cost = tf.reduce_mean(light_obj * tf.log(tf.boolean_mask(self.mask[..., 0], positive_shadow_mask)))
            #
            # dark_obj = tf.boolean_mask(mask_log[..., 1], negative_shadow_mask)
            # dark_cost = tf.reduce_mean(dark_obj * tf.log(tf.boolean_mask(self.mask[..., 1], negative_shadow_mask)))

            # softmax_loss = loss
            # shadow_detection = tf.reduce_sum(tf.subtract(1, tf.boolean_mask(mask_log[..., 1], positive_shadow_mask)))
            # background_detection = tf.reduce_sum(tf.subtract(1, tf.boolean_mask(mask_log[..., 0], negative_shadow_mask)))
            # soft_shadows = tf.reduce_sum(tf.boolean_mask(mask_log[..., 0], positive_shadow_mask))
            # dark_objects = tf.reduce_sum(tf.boolean_mask(mask_log[..., 1], negative_shadow_mask))
            # softmax_loss = shadow_detection * 2. + background_detection * 1 + soft_shadows * 2 + dark_objects * 5
            reg_loss = tf.add_n(
                [0.00005 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'conv5' not in var.name and
                 'fc' not in var.name]
            )
            loss = sc + reg_loss
            tf.summary.scalar('Loss', loss)
            with tf.name_scope('Optimization'):
                sd_lr = tf.placeholder(tf.float32, name='learn_rate_sd')
                tf.summary.scalar('Learning Rate', sd_lr)
                vars = [var for var in tf.trainable_variables() if 'conv5' not in var.name and
                        'fc' not in var.name]

                optr = tf.train.AdamOptimizer(learning_rate=sd_lr)
                train_opt = optr.minimize(loss, var_list=vars)

            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

        epoch = 28001
        with tf.Session() as sess:

            writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

            # \

            # sess.run([global_init])
            # self.net.load('../results/weights/vgg16.npy', sess)
            saver.restore(sess,self.weight_save + 'RFBUNET-ISTD-' + str(28000)+'.ckpt')
            names = os.listdir(self.image_path)

            for ep in range(self.EPOCH):
                print termcolor.colored('Epoch:', 'green'), ep
                random.shuffle(names)
                for bn in range(int(len(names) / self.BATCH_SIZE)):

                    loc_name = names[bn * self.BATCH_SIZE: (bn + 1) * self.BATCH_SIZE]
                    masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in loc_name]

                    hsv = [self.to_hsv(mp.imread(self.image_path + name)) for name in loc_name]
                    _, sd = sess.run([train_opt, loss], feed_dict={
                        self.himage: hsv,
                        self.mask: masks,
                        sd_lr: self.sd_lr
                    })
                    print 'Epoch:', ep, 'Batch:', bn, 'Learning Rate:', self.sd_lr, 'Cost:', sd,'step:',epoch

                    if ep <= 10:
                        self.sd_lr = 0.00001/3
                    if 60 > ep >= 10:
                        self.sd_lr = 0.00001/3
                    if 60 <= ep < 90:
                        self.sd_lr = 0.000001
                    if ep >= 90:
                        self.sd_lr = 0.000001

                    if (epoch % 30) == 0:
                        test_names = os.listdir(self.test_image_path)
                        random.shuffle(test_names)
                        test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png')) for name
                                      in
                                      test_names[:self.BATCH_SIZE]]
                        test_hsv = [self.to_hsv(mp.imread(self.test_image_path + name)) for name in
                                    test_names[:self.BATCH_SIZE]]
                        summary, tl = sess.run([merged_summary, loss], feed_dict={
                            self.himage: test_hsv,
                            self.mask: test_masks,
                            sd_lr: self.sd_lr})
                        writer.add_summary(summary,epoch)
                        print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Test Error:',
                                                                            'blue'), tl, 'Learning Rate:', self.sd_lr
                        print 'summary was updated.'

                    if (epoch % 1000) == 0:
                        saver.save(sess, self.weight_save + 'RFBUNET-ISTD-' + str(epoch) + '.ckpt')
                        print 'Model was saved successfully in epoch ' + str(epoch)
                    epoch += 1


if __name__ == '__main__':
    model = IlluminatorNet(BATCH_SIZE=3,sd_lr=0.00001/3)
    model.train()
