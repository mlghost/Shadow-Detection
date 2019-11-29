from layers import *
import tensorflow as tf
import numpy as np


def RFUNet(f, output_dim, name='RFB_Decoder'):
    with tf.name_scope(name):
        o = tf.image.resize_bilinear(f[3], (56, 56))
        o = conv_layer(o, [1, 1, 512, 256], 1, name + '_up_conv_1', activation='relu')
        o = tf.concat([o, f[2]], -1)
        o = RFB(o, 512, 256, 1, name=name + '_RFB_1')

        o = tf.image.resize_bilinear(o, (112, 112))
        o = conv_layer(o, [1, 1, 256, 128], 1, name + '_up_conv_2', activation='relu')
        o = tf.concat([o, f[1]], -1)
        o = RFB(o, 256, 128, 1, name=name + '_RFB_2')

        o = tf.image.resize_bilinear(o, (224, 224))
        o = conv_layer(o, [1, 1, 128, 64], 1, name + '_up_conv_3', activation='relu')
        o = tf.concat([o, f[0]], -1)
        o = RFB(o, 128, 64, 1, name=name + '_RFB_3')

        o = conv_layer(o, [1, 1, 64, output_dim], 1, name + '_output', activation='linear')

    return o


def FullyRFUNet(image, output_dim, name='RFB_Decoder'):
    with tf.name_scope(name):
        f1 = RFB(image, 3, 64, 1, name=name + '_RFB_1')  # 224 * 64
        m1 = max_pool_layer(f1, [1,2, 2,1], 2,'p1')  # 112

        f2 = RFB(m1, 64, 128, 1, name=name + '_RFB_2')  # 112 * 128
        m2 = max_pool_layer(f2, [1,2, 2,1], 2,'p2')  # 56

        f3 = RFB(m2, 128, 256, 1, name=name + '_RFB_3')  # 56 * 256
        m3 = max_pool_layer(f3, [1,2, 2,1], 2,'p3')  # 28

        f4 = RFB(m3, 256, 512, 1, name=name + '_RFB_4')  # 28 *512
        m4 = max_pool_layer(f4, [1,2, 2,1], 2,'p4')  # 14

        m = conv_layer(m4, [1, 1, 512, 1024], 1, name + 'mconv_1', activation='relu')  # 14 * 1024
        mm = conv_layer(m, [1, 1, 1024, 1024], 1, name + 'mconv_2', activation='relu')  # 14 * 1024


        o = tf.image.resize_bilinear(mm, (28, 28))  # 28
        o = conv_layer(o, [1, 1, 1024, 512], 1, name + '_up_conv_1', activation='relu')  # 28 * 512
        o = tf.concat([o, f4], -1)  # 28 * 1024
        o = RFB(o, 1024, 512, 1, name=name + '_RFB_6')  # 28 * 512

        o = tf.image.resize_bilinear(o, (56, 56))  # 56 * 512
        o = conv_layer(o, [1, 1, 512, 256], 1, name + '_up_conv_2', activation='relu')  # 56 * 256
        o = tf.concat([o, f3], -1)  # 56 * 512
        o = RFB(o, 512, 256, 1, name=name + '_RFB_7')  # 56 256

        o = tf.image.resize_bilinear(o, (112, 112))  # 112 256
        o = conv_layer(o, [1, 1, 256, 128], 1, name + '_up_conv_3', activation='relu')  # 112 128
        o = tf.concat([o, f2], -1)  # 112 256
        o = RFB(o, 256, 128, 1, name=name + '_RFB_8') # 112 128

        o = tf.image.resize_bilinear(o, (224, 224))  # 224
        o = conv_layer(o, [1, 1, 128, 64], 1, name + '_up_conv_4', activation='relu')  # 224 64
        o = tf.concat([o, f1], -1)  # 224 128
        o = RFB(o, 128, 64, 1, name=name + '_RFB_9')

        o = conv_layer(o, [1, 1, 64, output_dim], 1, name + '_output', activation='linear')

    return o

def multi_level_RFUNet(f, output_dim, name='RFB_Decoder'):
    with tf.name_scope(name):
        o = tf.image.resize_bilinear(f[3], (56, 56))
        extra_inf1  = tf.image.resize_bilinear(f[0], (56, 56))
        o = conv_layer(o, [1, 1, 512, 256], 1, name + '_up_conv_1', activation='relu')
        o = tf.concat([o, f[2],extra_inf1], -1)
        o = RFB(o, 64+512, 256, 1, name=name + '_RFB_1')

        o = tf.image.resize_bilinear(o, (112, 112))
        o = conv_layer(o, [1, 1, 256, 128], 1, name + '_up_conv_2', activation='relu')
        o = tf.concat([o, f[1]], -1)
        o = RFB(o, 256, 128, 1, name=name + '_RFB_2')

        o = tf.image.resize_bilinear(o, (224, 224))
        extra_inf2 = tf.image.resize_bilinear(f[3], (224, 224))
        o = conv_layer(o, [1, 1, 128, 64], 1, name + '_up_conv_3', activation='relu')
        o = tf.concat([o, f[0],extra_inf2], -1)
        o = RFB(o, 512+128, 64, 1, name=name + '_RFB_3')

        o = conv_layer(o, [1, 1, 64, output_dim], 1, name + '_output', activation='linear')

    return o