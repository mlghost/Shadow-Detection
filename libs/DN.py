import tensorflow as tf
from termcolor import colored
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mp
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('/content/gdrive/My Drive/SD/')
ROOT = '/content/gdrive/My Drive/SD/'
from libs.vgg16 import VGG_ILSVRC_16_layers
from libs.layers import *

class DynamicNet:

    def __init__(self, learning_rate=0.0001, BATCH_SIZE=10, activation_number=36, num_of_task=2):

        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], 'SupervisorImages')
        self.activation_number = activation_number
        self.num_of_task = num_of_task
        self.mask = tf.placeholder(tf.float32, [None, 224, 224, 2], 'image_mask')
        self.shadow_free = tf.placeholder(tf.float32, [None, 224, 224, 3], 'shadow_free_images')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.EPOCH = 500000
        self.summary_path = os.path.join(ROOT,'results/summary/')
        self.mask_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/train/train_B_224/')
        self.image_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/train/train_A_224/')
        self.test_image_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/test/test_A_224/')
        self.removed_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/train/train_C_224/')
        self.test_mask_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/test/test_B_224/')
        self.test_removed_path = os.path.join(ROOT,'data/ISTD_Dataset/augmented/test/test_C_224/')
        self.BATCH_SIZE = BATCH_SIZE
        self.supervisor = VGG_ILSVRC_16_layers({'data': self.input})

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
    def normalize_image(image):
        hsv = np.array(image, dtype=np.float32) / 255.
        return hsv

    def build_net(self):
        model_name = 'DynamicNet'
        with tf.name_scope('SupervisorNet'):
            output = self.supervisor.layers['fc7']
            activation_layer = fc(output, [4096, 2 * 63], 'ActivationLayer', 'relu')
            activation_layer = tf.sigmoid(activation_layer)
            task1 = activation_layer[:, :63]
            task2 = activation_layer[:, 63:]
            t1ea = task1[:, :36]
            t1da = task1[:, 36:]
            t2ea = task2[:, :36]
            t2da = task2[:, 36:]

        with tf.name_scope('GeneralEncoder'):
            t1o = encoder(self.input, t1ea)
            t2o = encoder(self.input, t2ea, reuse=True)

        with tf.name_scope('GeneralDecoder'):
            t1r = decoder(t1o, t1da)
            t2r = decoder(t2o, t2da, reuse=True)

        with tf.name_scope('FinalOutput'):
            mask = conv_layer(t1r, [1, 1, 64, 2], 1, name='mask', activation='linear')
            shadow_free = conv_layer(t2r, [1, 1, 64, 3], 1, name='shadow_free', activation='relu')

        with tf.name_scope('Optimization'):
            detection_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.mask, logits=mask))
            removal_cost = tf.reduce_mean(tf.losses.mean_squared_error(self.shadow_free,shadow_free))

            tf.summary.scalar('Detection Cost', detection_cost)
            tf.summary.scalar('Removal Cost', removal_cost)

            mask_preds = tf.expand_dims(tf.argmax(mask, 3), -1)
            mask_preds = tf.cast(mask_preds * 255 / 2, tf.uint8)

            t_mask_preds = tf.expand_dims(tf.argmax(self.mask, 3), -1)
            t_mask_preds = tf.cast(t_mask_preds * 255 / 2, tf.uint8)

            tf.summary.image('Shadow Mask', tf.concat([mask_preds, t_mask_preds], 2))

            sf_preds = tf.cast(shadow_free * 255, tf.uint8)
            t_sf_preds = tf.cast(self.shadow_free * 255, tf.uint8)

            tf.summary.image('Shadow Free', tf.concat([sf_preds, t_sf_preds], 2))

            detection_vars = [var for var in tf.trainable_variables() if
                              'fc8' not in var.name and 'shadow_free' not in var.name]
            removal_vars = [var for var in tf.trainable_variables() if 'fc8' not in var.name and 'mask' not in var.name]
            # for var in detection_vars:
            #     print(var.name)
            # print('=============================')
            # for var in removal_vars:
            #     print(var.name)
            train_opt1 = tf.train.AdamOptimizer(0.0001).minimize(detection_cost, var_list=detection_vars)
            train_opt2 = tf.train.AdamOptimizer(0.0001).minimize(removal_cost, var_list=removal_vars)
            # loss = detection_cost + removal_cost
            # train_opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
            saver = tf.train.Saver()
            global_init = tf.global_variables_initializer()
            merged_summary = tf.summary.merge_all()

            epoch = 1
            istrain = True
            with tf.Session() as sess:

                writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

                sess.run([global_init])

                names = os.listdir(self.image_path)
                if istrain:
                    for ep in range(self.EPOCH):
                        print colored('Epoch:', 'green'), ep
                        random.shuffle(names)
                        for bn in range(int(len(names) / self.BATCH_SIZE)):

                            loc_name = names[bn * self.BATCH_SIZE: (bn + 1) * self.BATCH_SIZE]

                            masks = [self._process_labels(mp.imread(self.mask_path + name[:-4] + '.png')) for name in
                                     loc_name]
                            hsv = [self.normalize_image(mp.imread(self.image_path + name)) for name in loc_name]
                            shadow_free_inputs = [self.normalize_image(mp.imread(self.removed_path + name)) for name in
                                                  loc_name]
                            print(colored('============================== Step '+ str(epoch),'red'))
                            _,_,  dc, rc = sess.run([train_opt1,train_opt2, detection_cost, removal_cost], feed_dict={
                                self.input: hsv,
                                self.mask: masks,
                                self.shadow_free: shadow_free_inputs,
                                self.lr: self.learning_rate
                            })
                            print colored('Epoch', 'green'), colored(ep, 'blue'), colored('LR:', 'green'), colored(
                                self.learning_rate, 'blue'), colored('Detection Cost:', 'green'), colored(dc, 'blue'), colored('Removal Cost:',
                                                                                              'green'), colored(rc,
                                                                                                                'blue')
                            if epoch <= 30000:
                                self.learning_rate = 0.0001
                            if 60000 > epoch > 30000:
                                self.learning_rate = 0.00001
                            if 60000 < ep < 90000:
                                self.learning_rate = 0.00001 / 3

                            if (epoch % 30) == 0:
                                test_names = os.listdir(self.test_image_path)
                                random.shuffle(test_names)
                                test_masks = [self._process_labels(mp.imread(self.test_mask_path + name[:-4] + '.png'))
                                              for name in test_names[:self.BATCH_SIZE]]
                                test_hsv = [self.normalize_image(mp.imread(self.test_image_path + name)) for name in
                                            test_names[:self.BATCH_SIZE]]
                                test_shadow_free_inputs = [
                                    self.normalize_image(mp.imread(self.test_removed_path + name)) for name in
                                    test_names[:self.BATCH_SIZE]]
                                summary, dc,rc = sess.run([merged_summary, detection_cost,removal_cost], feed_dict={
                                    self.input: test_hsv,
                                    self.mask: test_masks,
                                    self.shadow_free: test_shadow_free_inputs,
                                    self.lr: self.learning_rate})
                                writer.add_summary(summary, epoch)
                                print colored('Epoch', 'green'), colored(ep, 'blue'), colored('LR:', 'green'), colored(
                                    self.learning_rate, 'blue'), colored('Detection Cost:', 'green'), colored(dc,
                                                                                                              'blue'), colored(
                                    'Removal Cost:', 'green'), colored(rc, 'blue')
                                print 'summary was updated.'

                            if (epoch % 3000) == 0:
                                saver.save(sess, self.weight_save + model_name + '-' + str(epoch) + '.ckpt')
                                print 'Model was saved successfully in epoch ' + str(epoch)
                            epoch += 1

model = DynamicNet(BATCH_SIZE=1, learning_rate=0.0001)
model.build_net()