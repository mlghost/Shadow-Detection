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

            f1 = self.net.layers['conv1_2']
            f2 = self.net.layers['conv2_2']
            f3 = self.net.layers['conv3_3']
            f4 = self.net.layers['conv4_3']
            f5 = self.net.layers['conv5_3']

            # ======================= Shadow Prior Map =========================

            # f1 --> (112,112,64)
            l11 = layers.conv_layer(f1, [3, 3, 64, 128], 1, name='sd_conv_11')
            l12 = layers.conv_layer(l11, [3, 3, 128, 256], 1, name='sd_conv_12')

            # f2 --> (56,56,128)
            l21 = layers.deconv_layer(f2, [3, 3, 256, 128], [self.BATCH_SIZE, 224, 224, 256], 2, "sd_deconv2")
            l22 = layers.conv_layer(l21, [3, 3, 256, 256], 1, name='sd_conv_21')
            # l221 = layers.conv_layer(l22, [3, 3, 256, 2], 1, name='sp_conv_22')

            l31 = layers.deconv_layer(f3, [3, 3, 256, 256], [self.BATCH_SIZE, 224, 224, 256], 4, "sd_deconv3")
            l32 = layers.conv_layer(l31, [3, 3, 256, 256], 1, name='sd_conv_31')
            # l321 = layers.conv_layer(l32, [3, 3, 512, 2], 1, name='sp_conv_32')
            import tensorflow as tf
            from tensorflow.contrib.layers import xavier_initializer_conv2d

            def conv_layer(x, filtershape, stride, name, activation='prelu'):
                with tf.variable_scope(name):
                    w = tf.get_variable(name='w',
                                        shape=filtershape,
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0, stddev=0.001))
                    b = tf.get_variable(name='b',
                                        shape=[filtershape[-1]],
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
                    conv = tf.nn.conv2d(input=x,
                                        filter=w,
                                        strides=[1, stride, stride, 1],
                                        padding='SAME')
                    if activation == 'prelu':
                        output = prelu(conv + b)
                    elif activation == 'linear':
                        output = conv + b
                    else:
                        output = tf.nn.leaky_relu(conv + b)
                    return output

            def deconv_layer(x, filtershape, output_shape, stride, name):
                with tf.variable_scope(name):
                    filters = tf.get_variable(
                        name='weight',
                        shape=filtershape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
                        trainable=True)
                    deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding='SAME')
                    # deconv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
                    #                        trainable=True, name ='bias')
                    # bias = tf.nn.bias_add(deconv, deconv_biases)
                    # output = prelu(bias)
                    # output = tf.nn.dropout(prelu, keep_prob=keep_prob)
                    return prelu(deconv)

            def max_pool_layer(x, filtershape, stride, name):
                return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding='SAME', name=name)

            def prelu(x):
                with tf.variable_scope('prelu'):
                    alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                             initializer=tf.constant_initializer(0.0),
                                             dtype=tf.float32)
                    pos = tf.nn.relu(x)
                    neg = alphas * (x - tf.abs(x)) * 0.5
                    return pos + neg

            def resblock():
                pass

            l41 = layers.deconv_layer(f4, [3, 3, 512, 512], [self.BATCH_SIZE, 224, 224, 512], 8, "sd_deconv4")
            l42 = layers.conv_layer(l41, [3, 3, 512, 512], 1, name='sd_conv_41')
            # l421 = layers.conv_layer(l42, [3, 3, 256, 2], 1, name='sp_conv_42')

            l51 = layers.deconv_layer(f5, [3, 3, 512, 512], [self.BATCH_SIZE, 224, 224, 512], 16, "sd_deconv5")
            l52 = layers.conv_layer(l51, [3, 3, 512, 512], 1, name='sd_conv_51')
            # l521 = layers.conv_layer(l52, [3, 3, 256, 2], 1, name='sp_conv_52')

            SD = layers.conv_layer(tf.concat([l12, l22, l32, l42, l52], 3), [1, 1, 1792, 2], 1, 'sd_conv_61',activation='linear')

            # ======================= Shadow Remover =========================

            # f1 --> (112,112,64)
            sr11 = layers.conv_layer(f1, [3, 3, 64, 128], 1, name='sr_conv_11')
            sr12 = layers.conv_layer(sr11, [3, 3, 128, 256], 1, name='sr_conv_12')

            # f2 --> (56,56,128)
            sr21 = layers.deconv_layer(f2, [3, 3, 256, 128], [self.BATCH_SIZE, 224, 224, 256], 2, "sr_deconv2")
            sr22 = layers.conv_layer(sr21, [3, 3, 256, 256], 1, name='sr_conv_21')
            # l221 = layers.conv_layer(l22, [3, 3, 256, 2], 1, name='sp_conv_22')

            sr31 = layers.deconv_layer(f3, [3, 3, 256, 256], [self.BATCH_SIZE, 224, 224, 256], 4, "sr_deconv3")
            sr32 = layers.conv_layer(sr31, [3, 3, 256, 256], 1, name='sr_conv_31')
            # l321 = layers.conv_layer(l32, [3, 3, 512, 2], 1, name='sp_conv_32')

            sr41 = layers.deconv_layer(f4, [3, 3, 512, 512], [self.BATCH_SIZE, 224, 224, 512], 8, "sr_deconv4")
            sr42 = layers.conv_layer(sr41, [3, 3, 512, 512], 1, name='sr_conv_41')
            # l421 = layers.conv_layer(l42, [3, 3, 256, 2], 1, name='sp_conv_42')

            sr51 = layers.deconv_layer(f5, [3, 3, 512, 512], [self.BATCH_SIZE, 224, 224, 512], 16, "sr_deconv5")
            sr52 = layers.conv_layer(sr51, [3, 3, 512, 512], 1, name='sr_conv_51')
            # l521 = layers.conv_layer(l52, [3, 3, 256, 2], 1, name='sp_conv_52')

            SF = layers.conv_layer(tf.concat([sr12, sr22, sr32, sr42, sr52], 3), [1, 1, 1792, 1], 1, 'sr_conv_61',activation='linear')

            # ======================= Shadow Detector =========================

            tf.summary.image('Shadow Removal',
                             tf.concat(axis=2,
                                       values=[self.value, tf.reshape(SF, [-1, 224, 224, 1])]),
                             max_outputs=3)
            tf.summary.image('Shadow Detection',
                             tf.concat(axis=2, values=[tf.reshape(self.mask[..., 1], [-1, 224, 224, 1]),
                                                       tf.reshape(SD[..., 1], [-1, 224, 224, 1])]),
                             max_outputs=3)
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


            shadow_area = r_hat[..., 0] * self.mask[..., 1]
            non_shadow_area = r_hat[..., 0] * self.mask[..., 0]

            T1 = tf.reduce_mean(tf.losses.mean_squared_error(real_shadow_area, shadow_area))
            tf.summary.scalar('Shadow_Removal_Cost', T1)
            T2 = tf.reduce_mean(tf.losses.mean_squared_error(real_non_shadow_area, non_shadow_area))
            tf.summary.scalar('Non-Shadow_Removal_Cost', T2)
            sr_cost = 1 * T1 + 30 * T2
            tf.summary.scalar('SR Cost', sr_cost)
            cost = sr_cost + sd_cost
            tf.summary.scalar('Total Loss', cost)

            with tf.name_scope('Optimization'):
                sd_lr = tf.placeholder(tf.float32, name='learn_rate_sd')
                tf.summary.scalar('Learning Rate', sd_lr)
                optr = tf.contrib.opt.AdamWOptimizer(learning_rate=sd_lr,weight_decay=1e-5)
                # optr = tf.train.MomentumOptimizer(sd_lr,0.9)
                train_opt = optr.minimize(cost)

            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

            # config = tf.ConfigProto()
            # config.gpu_options.allocator_type = 'BFC'
            # config.gpu_options.per_process_gpu_memory_fraction = 0.80
            epoch = 1
            with tf.Session() as sess:

                writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

                # \

                sess.run([global_init])
                self.net.load('../results/weights/vgg16.npy', sess)
                # saver.restore(sess, self.weight_save + model_name + '-' + str(169000) + '.ckpt')
                names = os.listdir(self.image_path)

                for ep in range(self.EPOCH):
                    print termcolor.colored('Epoch:', 'green'), ep
                    random.shuffle(names)
                    for bn in range(int(len(names) / self.BATCH_SIZE)):

                        loc_name = names[bn * self.BATCH_SIZE: (bn + 1) * self.BATCH_SIZE]
                        masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in
                                 loc_name]

                        hsv = [self.to_hsv(mp.imread(self.image_path + name)) for name in loc_name]
                        _, sd = sess.run([train_opt, loss], feed_dict={
                            self.himage: hsv,
                            self.mask: masks,
                            sd_lr: self.sd_lr
                        })
                        print 'Epoch:', ep, 'Batch:', bn, 'Learning Rate:', self.sd_lr, 'Cost:', sd, 'step:', epoch

                        if epoch <= 30000:
                            self.sd_lr = 0.00001
                        if 60000 > epoch > 30000:
                            self.sd_lr = 0.00001 / 3
                        if 60000 < ep < 90000:
                            self.sd_lr = 0.000001

                        if (epoch % 30) == 0:
                            test_names = os.listdir(self.test_image_path)
                            random.shuffle(test_names)
                            test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png')) for
                                          name
                                          in
                                          test_names[:self.BATCH_SIZE]]
                            test_hsv = [self.to_hsv(mp.imread(self.test_image_path + name)) for name in
                                        test_names[:self.BATCH_SIZE]]
                            summary, tl = sess.run([merged_summary, loss], feed_dict={
                                self.himage: test_hsv,
                                self.mask: test_masks,
                                sd_lr: self.sd_lr})
                            writer.add_summary(summary, epoch)
                            print 'Epoch:', ep, 'Batch:', bn, termcolor.colored('Test Error:',
                                                                                'blue'), tl, 'Learning Rate:', self.sd_lr
                            print 'summary was updated.'

                        if (epoch % 3000) == 0:
                            saver.save(sess, self.weight_save + model_name + '-' + str(epoch) + '.ckpt')
                            print 'Model was saved successfully in epoch ' + str(epoch)
                        epoch += 1

    if __name__ == '__main__':
        model = IlluminatorNet(BATCH_SIZE=10, sd_lr=0.000001)
        model.train()

