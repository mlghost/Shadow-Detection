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
        # self.image_value = tf.placeholder(dtype='float', shape=[None, self.DIM, self.DIM, 1],
        #                                   name='image_value')
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

    def feature_extractor(self, m1, reuse=False, name='feature_extractor'):
        with tf.name_scope(name):
            l1 = layers.conv_layer(m1, [3, 3, 2, 64], 1, 'sdl1', reuse=reuse)
            l2 = layers.conv_layer(l1, [3, 3, 64, 128], 1, 'sdl2', reuse=reuse)
            l3 = layers.conv_layer(l2, [3, 3, 128, 128], 1, 'sdl3', reuse=reuse)
            l4 = layers.conv_layer(l3, [3, 3, 128, 256], 1, 'sdl4', reuse=reuse)
            return l4

    def shadow_refiner(self, m1, m2, name='Mask_Refiner', reuse=False):
        with tf.name_scope(name):
            f1 = self.feature_extractor(m1, reuse=reuse)
            f2 = self.feature_extractor(m2, reuse=True)
            ff = tf.concat([f1, f2], axis=3)
            ff1 = layers.conv_layer(ff, [3, 3, 512, 512], 1, 'sdmr_1', reuse=reuse)
            ff2 = layers.conv_layer(ff1, [3, 3, 512, 2], 1, 'sdmr_2', activation='linear', reuse=reuse)
            return ff2

    def feature_extractor1(self, m1, reuse=False, name='feature_extractor1'):
        with tf.name_scope(name):
            l1 = layers.conv_layer(m1, [3, 3, 1, 64], 1, 'srl1', reuse=reuse)
            l2 = layers.conv_layer(l1, [3, 3, 64, 128], 1, 'srl2', reuse=reuse)
            l3 = layers.conv_layer(l2, [3, 3, 128, 128], 1, 'srl3', reuse=reuse)
            l4 = layers.conv_layer(l3, [3, 3, 128, 256], 1, 'srl4', reuse=reuse)
            return l4

    def shadow_remover(self, m1, m2, name='Shadow_Remover', reuse=False):
        with tf.name_scope(name) as mr:
            f1 = self.feature_extractor1(m1, reuse=reuse)
            f2 = self.feature_extractor1(m2, reuse=True)
            ff = tf.concat([f1, f2], axis=3)
            ff1 = layers.conv_layer(ff, [3, 3, 512, 512], 1, 'sri_1', reuse=reuse)
            ff2 = layers.conv_layer(ff1, [3, 3, 512, 1], 1, 'sri_2', activation='linear', reuse=reuse)
            return ff2

    def build_net(self):
        with tf.name_scope('IlluminatorNet'):
            # ======================= Main Stream =========================

            fm1 = layers.conv_layer(self.himage, [9, 9, 3, 64], 1, name='m_conv_1')
            fm2 = layers.conv_layer(fm1, [7, 7, 64, 64], 1, name='m_conv_2')

            fm3 = layers.conv_layer(fm2, [5, 5, 64, 128], 1, name='m_conv_3')
            fm4 = layers.conv_layer(fm3, [3, 3, 128, 128], 1, name='m_conv_4')

            fm5 = layers.conv_layer(fm4, [3, 3, 128, 256], 1, name='m_conv_5')
            fm6 = layers.conv_layer(fm5, [3, 3, 256, 256], 1, name='m_conv_6')

            fm7 = layers.conv_layer(fm6, [3, 3, 256, 512], 1, name='m_conv_7')
            fm8 = layers.conv_layer(fm7, [3, 3, 512, 512], 1, name='m_conv_8')

            # ======================= Shadow Prior Map =========================

            with tf.name_scope('Shadow_Detection'):
                sd1 = layers.conv_layer(fm2, [3, 3, 64, 128], 1, 'sd_1')
                sd2 = layers.conv_layer(sd1, [3, 3, 128, 2], 1, 'sd_2')

                sd3 = layers.conv_layer(fm4, [3, 3, 128, 128], 1, 'sd_3')
                sd4 = layers.conv_layer(sd3, [3, 3, 128, 2], 1, 'sd_4')

                sd5 = layers.conv_layer(fm6, [3, 3, 256, 256], 1, 'sd_5')
                sd6 = layers.conv_layer(sd5, [3, 3, 256, 2], 1, 'sd_6')

                sd7 = layers.conv_layer(fm8, [3, 3, 512, 512], 1, 'sd_7')
                sd8 = layers.conv_layer(sd7, [3, 3, 512, 2], 1, 'sd_8')

                SD1 = self.shadow_refiner(sd2, sd4, name='Mask_Refiner')
                SD2 = self.shadow_refiner(SD1, sd6, reuse=True, name='Mask_Refiner')
                SD = self.shadow_refiner(SD2, sd8, reuse=True, name='Mask_Refiner')
                tf.summary.image('Shadow Mask',
                                 tf.concat(axis=2, values=[tf.reshape(SD1[..., 1], [-1, 224, 224, 1]),tf.reshape(SD2[..., 1], [-1, 224, 224, 1]),
                                                           tf.reshape(SD[..., 1], [-1, 224, 224, 1]),tf.reshape(self.mask[..., 1], [-1, 224, 224, 1])]),
                                 max_outputs=4)
            with tf.name_scope('Shadow_Removal'):
                sr1 = layers.conv_layer(fm2, [3, 3, 64, 128], 1, 'sr_1')
                sr2 = layers.conv_layer(sr1, [3, 3, 128, 1], 1, 'sr_2')

                sr3 = layers.conv_layer(fm4, [3, 3, 128, 128], 1, 'sr_3')
                sr4 = layers.conv_layer(sr3, [3, 3, 128, 1], 1, 'sr_4')

                sr5 = layers.conv_layer(fm6, [3, 3, 256, 256], 1, 'sr_5')
                sr6 = layers.conv_layer(sr5, [3, 3, 256, 1], 1, 'sr_6')

                sr7 = layers.conv_layer(fm8, [3, 3, 512, 512], 1, 'sr_7')
                sr8 = layers.conv_layer(sr7, [3, 3, 512, 1], 1, 'sr_8')

                SF1 = self.shadow_remover(sr2, sr4, name='Illumination_Refiner')
                SF2 = self.shadow_remover(SF1, sr6, reuse=True, name='Illumination_Refiner')
                SF = self.shadow_remover(SF2, sr8, reuse=True, name='Illumination_Refiner')
                tf.summary.image('Shadow Free',
                             tf.concat(axis=2, values=[tf.reshape(SF1, [-1, 224, 224, 1]), tf.reshape(SF2, [-1, 224, 224, 1]),
                                                       tf.reshape(SF, [-1, 224, 224, 1]), tf.reshape(self.value, [-1, 224, 224, 1])]),
                             max_outputs=4)
            return SF1, SF2, SF, SD1, SD2, SD

    def train(self):

        with tf.name_scope('Model'):

            sf1, sf2, sf, sd1, sd2, sd = self.build_net()
            # SD1
            inverse_class_weights1 = tf.divide(1.0,
                                               tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                             tf.nn.softmax(sd1))))
            decode_logits_weighted1 = tf.multiply(sd1, inverse_class_weights1)
            #
            sd_cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=decode_logits_weighted1, labels=self.mask, name='entropy_loss'))

            # SD2
            inverse_class_weights2 = tf.divide(1.0,
                                               tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                             tf.nn.softmax(sd2))))
            decode_logits_weighted2 = tf.multiply(sd2, inverse_class_weights2)
            #
            sd_cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=decode_logits_weighted2, labels=self.mask, name='entropy_loss'))

            # SD
            inverse_class_weights3 = tf.divide(1.0,
                                               tf.log(tf.add(tf.constant(1.02, tf.float32),
                                                             tf.nn.softmax(sd))))
            decode_logits_weighted3 = tf.multiply(sd, inverse_class_weights3)
            #
            sd_cost_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=decode_logits_weighted3, labels=self.mask, name='entropy_loss'))

            sd_cost = 1 * sd_cost_1 + 3 * sd_cost_2 + 6 *sd_cost_3

            tf.summary.scalar('SD Cost', sd_cost)
            # sd_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_hat, labels=self.mask))
            # sr_cost = tf.reduce_mean(tf.losses.absolute_difference(self.value, r_hat))

            real_shadow_area = self.value[..., 0] * self.mask[..., 1]
            real_non_shadow_area = self.value[..., 0] * self.mask[..., 0]

            shadow_area1 = sd1[..., 0] * self.mask[..., 1]
            non_shadow_area1 = sd1[..., 0] * self.mask[..., 0]

            shadow_area2 = sd2[..., 0] * self.mask[..., 1]
            non_shadow_area2 = sd2[..., 0] * self.mask[..., 0]

            shadow_area3 = sd[..., 0] * self.mask[..., 1]
            non_shadow_area3 = sd[..., 0] * self.mask[..., 0]

            T1 = tf.reduce_mean(tf.losses.mean_squared_error(real_shadow_area, shadow_area1))
            T2 = tf.reduce_mean(tf.losses.mean_squared_error(real_non_shadow_area, non_shadow_area1))
            sr_cost_1 = 1 * T1 + 30 * T2

            T3 = tf.reduce_mean(tf.losses.mean_squared_error(real_shadow_area, shadow_area2))
            T4 = tf.reduce_mean(tf.losses.mean_squared_error(real_non_shadow_area, non_shadow_area2))
            sr_cost_2 = 1 * T3 + 30 * T4

            T5 = tf.reduce_mean(tf.losses.mean_squared_error(real_shadow_area, shadow_area3))
            T6 = tf.reduce_mean(tf.losses.mean_squared_error(real_non_shadow_area, non_shadow_area3))
            sr_cost_3 = 1 * T5 + 30 * T6

            sr_cost = 1 * sr_cost_1 + 3 * sr_cost_2 + 6 * sr_cost_3
            tf.summary.scalar('SR Cost', sr_cost)

            cost = sr_cost + sd_cost
            tf.summary.scalar('Total Loss', cost)

            with tf.name_scope('Optimization'):
                sd_lr = tf.placeholder(tf.float32, name='learn_rate')
                tf.summary.scalar('Learning Rate', sd_lr)
                optr = tf.train.AdamOptimizer(learning_rate=sd_lr)
                train_opt = optr.minimize(cost)
            # with tf.name_scope('Optimization'):
            #     sd_lr = tf.placeholder(tf.float32, name='learn_rate')
            #
            #     opt1 = tf.train.AdamOptimizer(learning_rate=sd_lr)
            #     opt2 = tf.train.AdamOptimizer(learning_rate=sd_lr)
            #     opt3 = tf.train.AdamOptimizer(learning_rate=sd_lr)
            #     sd_vars = [var for var in tf.trainable_variables() if 'sd' in var.name]
            #     sr_vars = [var for var in tf.trainable_variables() if 'sr' in var.name]
            #     main_vars = [var for var in tf.trainable_variables() if 'm_conv' in var.name]
            #     train1 = opt1.minimize(sd_cost,var_list=sd_vars)
            #     train2 = opt2.minimize(sr_cost,var_list=sr_vars)
            #     train3 = opt3.minimize(cost,var_list=main_vars)
            #     # train_opt = optr.minimize(cost)
            # #
            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

            # config = tf.ConfigProto()
            # config.gpu_options.allocator_type = 'BFC'
            # config.gpu_options.per_process_gpu_memory_fraction = 0.80
        epoch = 1
        with tf.Session() as sess:

            writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

            sess.run([global_init])
            # print 'Loading VGG-16...'
            # self.net.load('../results/weights/vgg16.npy', sess)
            # print 'VGG-16 was loaded.'
            names = os.listdir(self.image_path)

            for ep in range(self.EPOCH):
                print termcolor.colored('Epoch:', 'green'), ep
                random.shuffle(names)
                for bn in range(int(len(names) / self.BATCH_SIZE)):

                    loc_name = names[bn * self.BATCH_SIZE: (bn + 1) * self.BATCH_SIZE]
                    masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in loc_name]
                    hsv = [rgb_to_hsv(mp.imread(self.image_path + name)) for name in loc_name]
                    # im_value = [np.reshape(image[:, :, -1] / 255, (224, 224, 1)) for image in hsv]
                    removed = [
                        np.reshape(rgb_to_hsv(mp.imread(self.removed_path + name))[:, :, -1] / 255, (224, 224, 1))
                        for name in loc_name]
                    print 'Batch', bn + 1, 'loaded.'
                    _,l, sr, sd = sess.run([train_opt, cost, sd_cost, sr_cost], feed_dict={
                        self.himage: hsv,
                        self.mask: masks,
                        # self.image_value: im_value,
                        self.value: removed,
                        sd_lr: self.sd_lr
                    })
                    print 'SR:', sr, "SD:", sd
                    print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Train Error:', 'green'), l, 'Learning Rate:', \
                        self.sd_lr

                    if (ep % 2) == 1:
                        self.sd_lr = self.lr_decay(0.001, 1, epoch)
                        print 'Learning rate Decay. LR=', self.sd_lr

                    if (epoch % 30) == 0:
                        test_names = os.listdir(self.test_image_path)
                        random.shuffle(test_names)
                        test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png'))
                                      for name in test_names[:self.BATCH_SIZE]]
                        test_hsv = [rgb_to_hsv(mp.imread(self.test_image_path + name)) for name in test_names[:self.BATCH_SIZE]]
                        # test_im_value = [np.reshape(image[:, :, -1] / 255, (224, 224, 1)) for image in test_hsv]
                        test_removed = [np.reshape(rgb_to_hsv(mp.imread(self.test_removed_path + name))[:, :, -1] / 255,
                                                   (224, 224, 1))
                                        for name in test_names[:self.BATCH_SIZE]]
                        summary, tl = sess.run([merged_summary, cost], feed_dict={
                            self.himage: test_hsv,
                            self.mask: test_masks,
                            # self.image_value: test_im_value,
                            self.value: test_removed,
                            sd_lr: self.sd_lr})
                        writer.add_summary(summary, epoch)
                        print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Test Error:',
                                                                            'blue'), tl, 'Learning Rate:', self.sd_lr
                        print 'summary was updated.'

                    if (epoch % 3000) == 0:
                        saver.save(sess, self.weight_save, epoch, latest_filename='Model-v2-' + str(epoch)+'.ckpt')
                        print 'Model was saved successfully in epoch ' + str(epoch)
                    epoch += 1


if __name__ == '__main__':
    model = IlluminatorNet(BATCH_SIZE=1, EPOCH=40, sd_lr=0.001)
    model.train()
