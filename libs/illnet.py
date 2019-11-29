import tensorflow as tf
import layers
from vgg16 import VGG_ILSVRC_16_layers
import os
import matplotlib.pyplot as plt
import matplotlib.image as mp
import numpy as np
import random
import termcolor
from matplotlib.colors import rgb_to_hsv


class IlluminatorNet:
    graph_save = ''
    weight_save = ''
    activation = 'prelu'
    block_type = 'conv'
    agg = 'sum'
    conditional_detection = True
    color_space = 'HSV'
    main_input = 'RGB'
    use_gpu = False
    DIM = 224
    block = None
    detection_mode = 0
    EPOCH = 50000000
    sd_lr = 1e-4  # 1E-4
    beta1 = 0.9
    beta2 = 0.999
    optimizer = 'Adam'
    BATCH_SIZE = 4
    INIT_LEARNING_RATE = 1e-4

    mask_path = '../data/ISTD_Dataset/augmented/train/train_B_224/'
    image_path = '../data/ISTD_Dataset/augmented/train/train_A_224/'
    test_image_path = '../data/ISTD_Dataset/augmented/test/test_A_224/'
    removed_path = '../data/ISTD_Dataset/augmented/train/train_C_224/'
    test_mask_path = '../data/ISTD_Dataset/augmented/test/test_B_224/'
    test_removed_path = '../data/ISTD_Dataset/augmented/test/test_C_224/'

    def __init__(self, graph_save='../graph_save/', weight_save='../results/weight/', activation='prelu',
                 block='conv', agg='sum', EPOCH=50000000, BATCH_SIZE=3,
                 conditional_detection=True, color_space='HSV', main_input='RGB',
                 use_gpu=False, DIM=224, detection_mode=0,
                 sd_lr=1e-2, beta1=0.9, beta2=0.999, optimizer='Adam'):

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
        self.image_value = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 1],
                                          name='image_value')
        self.mask = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 2],
                                   name='Mask')
        self.value = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 1],
                                    name='value')
        self.net = VGG_ILSVRC_16_layers({'data': self.himage})
        self.detection_mode = detection_mode
        self.EPOCH = EPOCH

    @staticmethod
    def lr_decay(lr_input, decay_rate, num_epoch):
        return lr_input / (1 + decay_rate * num_epoch)

    @staticmethod
    def _process_labels(label):
        labels = np.zeros((224, 224, 2), dtype=np.float32)
        for i in range(224):
            for j in range(224):
                if label[i][j] == 0:
                    labels[i, j, 0] = 1
                else:
                    labels[i, j, 1] = 1
        # labels[..., 1] = label
        # labels[..., 0] = np.invert(label.astype(np.int))
        return labels

    def build_net(self):
        with tf.name_scope('IlluminatorNet'):

            # ======================= Main Stream =========================

            f1 = self.net.layers['pool1']
            f2 = self.net.layers['pool2']
            f3 = self.net.layers['pool3']
            f4 = self.net.layers['pool4']
            f5 = self.net.layers['pool5']

            # ======================= Shadow Prior Map =========================

            # f1 --> (112,112,64)
            l11 = layers.conv_layer(f1, [3, 3, 64, 64], 1, name='sp_conv_11')
            l121 = layers.conv_layer(l11, [3, 3, 64, 128], 1, name='sp_conv_12')
            l12 = layers.conv_layer(l121, [3, 3, 128, 1], 1, name='sp_conv_13')
            # l13 = layers.deconv_layer(l12, [3, 3, 1, 1], [self.BATCH_SIZE, 224, 224, 1], 2, "deconv1")
            l13 = tf.image.resize_images(l12, [224, 224])
            # tf.image.resize_bilinear()
            # tf.summary.image('SP_1', tf.reshape(l13, [-1, 224, 224, 1]))
            # f2 --> (56,56,128)
            l21 = layers.conv_layer(f2, [3, 3, 128, 64], 1, name='sp_conv_21')
            l211 = layers.conv_layer(l21, [3, 3, 64, 128], 1, name='sp_conv_22')
            l22 = layers.conv_layer(l211, [3, 3, 128, 1], 1, name='sp_conv_23')
            # l23 = layers.deconv_layer(l22, [3, 3, 1, 1], [self.BATCH_SIZE, 224, 224, 1], 4, "deconv2")
            l23 = tf.image.resize_images(l22, [224, 224])
            # tf.summary.image('SP_2', tf.reshape(l23, [-1, 224, 224, 1]))
            # f3 --> (28,28,256)
            l31 = layers.conv_layer(f3, [3, 3, 256, 128], 1, name='sp_conv_31')
            l311 = layers.conv_layer(l31, [3, 3, 128, 128], 1, name='sp_conv_32')
            l32 = layers.conv_layer(l311, [3, 3, 128, 1], 1, name='sp_conv_33')
            # l33 = layers.deconv_layer(l32, [3, 3, 1, 1], [self.BATCH_SIZE, 224, 224, 1], 8, "deconv3")
            l33 = tf.image.resize_images(l32, [224, 224])
            # tf.summary.image('SP_3', tf.reshape(l33, [-1, 224, 224, 1]))
            # f4 --> (14,14,512)
            l41 = layers.conv_layer(f4, [3, 3, 512, 512], 1, name='sp_conv_41')
            l411 = layers.conv_layer(l41, [3, 3, 512, 128], 1, name='sp_conv_42')
            l42 = layers.conv_layer(l411, [3, 3, 128, 1], 1, name='sp_conv_43')
            # l43 = layers.deconv_layer(l42, [3, 3, 1, 1], [self.BATCH_SIZE, 224, 224, 1], 16, "deconv4")
            l43 = tf.image.resize_images(l42, [224, 224])
            # tf.summary.image('SP_4', tf.reshape(l43, [-1, 224, 224, 1]))
            # f5 --> (7,7,512)
            l51 = layers.conv_layer(f5, [3, 3, 512, 128], 1, name='sp_conv_51')
            l511 = layers.conv_layer(l51, [3, 3, 128, 128], 1, name='sp_conv_52')
            l52 = layers.conv_layer(l511, [3, 3, 128, 1], 1, name='sp_conv_53')
            # l53 = layers.deconv_layer(l52, [3, 3, 1, 1], [self.BATCH_SIZE, 224, 224, 1], 32, "deconv5")
            l53 = tf.image.resize_images(l52, [224, 224])
            # tf.summary.image('SP_5', tf.reshape(l53, [-1, 224, 224, 1]))
            # ======================= Shadow Remover =========================

            SR1 = layers.conv_layer(
                tf.concat([self.image_value, l13, l23, l33, l43, l53], axis=3, name='concat_2'), [1, 1, 6, 128],
                1,
                name='SR_conv_1')
            SR2 = layers.conv_layer(SR1, [1, 1, 128, 128], 1, name='SR_conv_2')
            SF = layers.conv_layer(SR2, [1, 1, 128, 1], 1, name='SR_conv_3', activation='linear')
            # tf.summary.image('SF', tf.reshape(SF, [-1, 224, 224, 1]))
            # tf.summary.image('Hue ', tf.reshape(self.image_value, [-1, 224, 224, 1]))

            # ======================= Shadow Detector =========================

            SD1 = layers.conv_layer(tf.concat([l13, l23, l33, l43, l53], axis=3, name='concat_1'), [1, 1, 5, 128],
                                    1,
                                    name='SD_conv_1')
            SD2 = layers.conv_layer(SD1, [1, 1, 128, 128], 1, name='SD_conv_2')
            SD = layers.conv_layer(SD2, [1, 1, 128, 2], 1, name='SD_conv_3', activation='linear')
            # tf.summary.image('SD', tf.reshape(SD[..., 1], [-1, 224, 224, 1]))
            # tf.summary.image('Mask', tf.reshape(self.mask[..., 1], [-1, 224, 224, 1]))
            tf.summary.image('Shadow Removal',
                             tf.concat(axis=2,
                                       values=[self.value, self.image_value, tf.reshape(SF, [-1, 224, 224, 1])]),
                             max_outputs=3)
            tf.summary.image('Shadow Detection',
                             tf.concat(axis=2, values=[tf.reshape(self.mask[..., 1], [-1, 224, 224, 1]),
                                                       tf.reshape(SD[..., 1], [-1, 224, 224, 1])]),
                             max_outputs=3)
            tf.summary.image('Shadow Prior',
                             tf.concat(axis=2, values=[l13, l23, l33, l43, l53]),
                             max_outputs=3)  #
            return SF, SD

    def train(self):

        with tf.name_scope('Model'):

            r_hat, m_hat = self.build_net()
            inverse_class_weights = tf.divide(1.0,
                                              tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                            tf.nn.softmax(m_hat))))
            decode_logits_weighted = tf.multiply(m_hat, inverse_class_weights)
            #
            sd_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=decode_logits_weighted, labels=self.mask, name='entropy_loss'))
            tf.summary.scalar('SD Cost', sd_cost)
            # sd_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_hat, labels=self.mask))
            # sr_cost = tf.reduce_mean(tf.losses.absolute_difference(self.value, r_hat))

            real_shadow_area = self.value[..., 0] * self.mask[..., 1]
            real_non_shadow_area = self.value[..., 0] * self.mask[..., 0]
            tf.summary.image('Real Shadow Non-Shadow',
                             tf.concat(axis=2, values=[tf.reshape(real_shadow_area, [-1, 224, 224, 1]),
                                                       tf.reshape(real_non_shadow_area, [-1, 224, 224, 1])]),
                             max_outputs=3)

            shadow_area = r_hat[..., 0] * self.mask[..., 1]
            non_shadow_area = r_hat[..., 0] * self.mask[..., 0]
            tf.summary.image('Shadow Non-Shadow',
                             tf.concat(axis=2, values=[tf.reshape(shadow_area, [-1, 224, 224, 1]),
                                                       tf.reshape(non_shadow_area, [-1, 224, 224, 1])]),
                             max_outputs=3)
            T1 = tf.reduce_mean(tf.losses.mean_squared_error(real_shadow_area, shadow_area))
            tf.summary.scalar('Shadow_Removal_Cost', T1)
            T2 = tf.reduce_mean(tf.losses.mean_squared_error(real_non_shadow_area, non_shadow_area))
            tf.summary.scalar('Non-Shadow_Removal_Cost', T2)
            sr_cost = 5 * T1 + 3 * T2
            tf.summary.scalar('SR Cost',sr_cost)
            cost = sr_cost + sd_cost
            tf.summary.scalar('Total Loss', cost)

            with tf.name_scope('Optimization'):
                sd_lr = tf.placeholder(tf.float32, name='learn_rate_sd')
                tf.summary.scalar('Learning Rate', sd_lr)
                optr = tf.train.AdamOptimizer(learning_rate=sd_lr)
                train_opt = optr.minimize(cost)

            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

            # config = tf.ConfigProto()
            # config.gpu_options.allocator_type = 'BFC'
            # config.gpu_options.per_process_gpu_memory_fraction = 0.80
        epoch = 1
        with tf.Session() as sess:
            test_names = os.listdir(self.test_image_path)
            test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png'))
                          for name in test_names[:self.BATCH_SIZE]]
            test_hsv = [mp.imread(self.test_image_path + name) for name in
                        test_names[:self.BATCH_SIZE]]
            test_im_value = [image / 255 for image in test_hsv]
            test_removed = [
                np.reshape(rgb_to_hsv(mp.imread(self.test_removed_path + name))[:, :, -1] / 255,
                           (224, 224, 1))
                for name in test_names[:self.BATCH_SIZE]]

            writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

            sess.run([global_init])
            print 'Loading VGG-16...'
            self.net.load('../results/weights/vgg16.npy', sess)
            print 'VGG-16 was loaded.'
            names = os.listdir(self.image_path)

            for ep in range(self.EPOCH):
                print termcolor.colored('Epoch:', 'green'), ep
                random.shuffle(names)
                for bn in range(int(len(names) / self.BATCH_SIZE)):

                    loc_name = names[bn * self.BATCH_SIZE: (bn + 1) * self.BATCH_SIZE]
                    masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in loc_name]
                    hsv = [mp.imread(self.image_path + name) for name in loc_name]
                    im_value = [np.reshape(rgb_to_hsv(image)[:, :, -1] / 255, (224, 224, 1)) for image in hsv]
                    removed = [
                        np.reshape(rgb_to_hsv(mp.imread(self.removed_path + name))[:, :, -1] / 255, (224, 224, 1))
                        for name in loc_name]
                    print 'Batch', bn + 1, 'loaded.'
                    _, l, sr, sd = sess.run([train_opt, cost, sd_cost, sr_cost], feed_dict={
                        self.himage: hsv,
                        self.mask: masks,
                        self.image_value: im_value,
                        self.value: removed,
                        sd_lr: self.sd_lr
                    })
                    print 'SR:', sr, "SD:", sd
                    print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Train Error:', 'green'), l, 'Learning Rate:', \
                        self.sd_lr

                    if (epoch % 3000) == 0:
                        self.sd_lr = self.lr_decay(0.001, 1, epoch)
                        print 'Learning rate Decay. LR=', self.sd_lr

                    if (epoch % 30) == 0:
                        summary, tl = sess.run([merged_summary, cost], feed_dict={
                            self.himage: test_hsv,
                            self.mask: test_masks,
                            self.image_value: test_im_value,
                            self.value: test_removed,
                            sd_lr: self.sd_lr})
                        writer.add_summary(summary, epoch)
                        print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Test Error:',
                                                                            'blue'), tl, 'Learning Rate:', self.sd_lr
                        print 'summary was updated.'

                    if (epoch % 3000) == 0:
                        saver.save(sess, self.weight_save, epoch, latest_filename='Model-' + str(epoch))
                        print 'Model was saved successfully in epoch ' + str(epoch)
                    epoch += 1


if __name__ == '__main__':
    model = IlluminatorNet(BATCH_SIZE=10,EPOCH=40,sd_lr=0.001)
    model.train()
