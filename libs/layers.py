import tensorflow as tf


def prelu(x):
    with tf.variable_scope('prelu'):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5
        return pos + neg


def conv_layer(x, filtershape, stride, name, activation='prelu', reuse=False):
    with tf.variable_scope(name) as layer:
        if reuse:
            layer.reuse_variables()
        w = tf.get_variable(name='w',
                            shape=filtershape,
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.001))
        conv = tf.nn.conv2d(input=x,
                            filter=w,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        if activation == 'prelu':
            output = prelu(conv)
        elif activation == 'linear':
            output = conv
        else:
            output = tf.nn.leaky_relu(conv)
        return output


def fc(x, filtershape, name, activation='prelu', reuse=False):
    with tf.variable_scope(name) as layer:
        if reuse:
            layer.reuse_variables()
        w = tf.get_variable(name='w',
                            shape=filtershape,
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        b = tf.get_variable(name='b',
                            shape=[filtershape[-1]],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        xw = tf.nn.xw_plus_b(x, w,b)
        if activation == 'prelu':
            output = prelu(xw)
        elif activation == 'linear':
            output = xw
        else:
            output = tf.nn.leaky_relu(xw)
        return output


def atrous_conv_layer(x, filtershape, rate, name, activation='linear', reuse=False):
    with tf.variable_scope(name) as layer:
        if reuse:
            layer.reuse_variables()
        w = tf.get_variable(name='w',
                            shape=filtershape,
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        conv = tf.nn.atrous_conv2d(value=x, filters=w, rate=rate, padding='SAME', name='atr_conv')
        if activation == 'prelu':
            output = prelu(conv)
        elif activation == 'linear':
            output = conv
        else:
            output = tf.nn.leaky_relu(conv)
        return output


def deconv_layer(x, filtershape, output_shape, stride, name, act='linear'):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name='weight',
            shape=filtershape,
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            trainable=True)

        deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding='SAME')

        if act == 'linear':
            return deconv
        elif act == 'relu':
            return tf.nn.relu(deconv)
        elif act == 'prelu':
            return prelu(deconv)
        else:
            return tf.nn.leaky_relu(deconv)


def max_pool_layer(x, filtershape, stride, name):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding='SAME', name=name)


def RFB(input, in_dim, output_dim, stride, name='RFB'):
    with tf.name_scope(name):
        l1 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l1', activation='linear')
        l2 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l2', activation='linear')
        l3 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l3', activation='linear')
        l1 = conv_layer(l1, [5, 5, output_dim, output_dim], stride, name + '_l11', activation='linear')
        l2 = conv_layer(l2, [3, 3, output_dim, output_dim], stride, name + '_l22', activation='linear')
        l1 = atrous_conv_layer(l1, [3, 3, output_dim, output_dim], 5, name + 'atr_l1', activation='linear')
        l2 = atrous_conv_layer(l2, [3, 3, output_dim, output_dim], 3, name + 'atr_l2', activation='linear')
        l3 = atrous_conv_layer(l3, [3, 3, output_dim, output_dim], 1, name + 'atr_l3', activation='linear')
        concat = tf.concat([l1, l2, l3], -1)
        output = conv_layer(concat, [1, 1, 3 * output_dim, output_dim], 1, name + '_conv_final', activation='linear')
        if output_dim == in_dim:
            shortcut = input
        else:
            shortcut = conv_layer(input, [1, 1, in_dim, output_dim], 1, '_shortcut', activation='linear')

        return tf.nn.relu(output + shortcut)


def atrous_block(input, in_dim, output_dim, kernel, rate, name='RFB', reuse=False):
    with tf.name_scope(name):
        l1 = conv_layer(input, [kernel, kernel, in_dim, output_dim], 1, name + '_l1', activation='linear', reuse=reuse)
        l1 = atrous_conv_layer(l1, [3, 3, output_dim, output_dim], rate, name + 'atr_l1', activation='linear',
                               reuse=reuse)
        return l1


def DRFB(input, in_depth, out_depth, weights, name='DRFB', reuse=False):
    with tf.name_scope(name):
        # ====================  Layer 1  =======================

        b11 = atrous_block(input, in_depth, out_depth, 1, 1, name + '_b1', reuse=reuse)
        b12 = atrous_block(input, in_depth, out_depth, 1, 3, name + '_b2', reuse=reuse)
        b13 = atrous_block(input, in_depth, out_depth, 1, 5, name + '_b3', reuse=reuse)
        b14 = atrous_block(input, in_depth, out_depth, 3, 1, name + '_b4', reuse=reuse)
        b15 = atrous_block(input, in_depth, out_depth, 3, 3, name + '_b5', reuse=reuse)
        b16 = atrous_block(input, in_depth, out_depth, 3, 5, name + '_b6', reuse=reuse)
        b17 = atrous_block(input, in_depth, out_depth, 5, 1, name + '_b7', reuse=reuse)
        b18 = atrous_block(input, in_depth, out_depth, 5, 3, name + '_b8', reuse=reuse)
        b19 = atrous_block(input, in_depth, out_depth, 5, 5, name + '_b9', reuse=reuse)
        lo = tf.concat([
            b11 * weights[:, 0],
            b12 * weights[:, 1],
            b13 * weights[:, 2],
            b14 * weights[:, 3],
            b15 * weights[:, 4],
            b16 * weights[:, 5],
            b17 * weights[:, 6],
            b18 * weights[:, 7],
            b19 * weights[:, 8]
        ],
            axis=3,
            name=name + '_ec')

        lo = conv_layer(lo, [1, 1, 9 * out_depth, out_depth],stride=1,activation= 'relu', name=name + '_lo', reuse=reuse,)
        return lo


def encoder(input, weight, reuse=False):
    with tf.name_scope('GeneralEncoder'):
        l1 = DRFB(input, 3, 64, weight[:, :9], name='DRFB_1', reuse=reuse)
        l1 = tf.layers.max_pooling2d(inputs=l1, pool_size=[2, 2], strides=2)
        l2 = DRFB(l1, 64, 128, weight[:, 9:18], name='DRFB_2', reuse=reuse)
        l2 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2, 2], strides=2)
        l3 = DRFB(l2, 128, 256, weight[:, 18:27], name='DRFB_3', reuse=reuse)
        l3 = tf.layers.max_pooling2d(inputs=l3, pool_size=[2, 2], strides=2)
        l4 = DRFB(l3, 256, 512, weight[:, 27:], name='DRFB_4', reuse=reuse)
    return l4


def decoder(input, weight, reuse=False):
    with tf.name_scope('GeneralEncoder'):
        # ====================  Layer 1  =======================

        l1 = DRFB(input, 512,256, weight[:, :9], name='DRFB_5', reuse=reuse)
        print(l1.get_shape())
        l1 = tf.layers.conv2d_transpose(l1, 256, (1, 1), 2, padding='SAME', name='up1',reuse=reuse)
        print(l1.get_shape())
        l2 = DRFB(l1, 256,128, weight[:, 9:18], name='DRFB_6', reuse=reuse)
        print(l2.get_shape())
        l2 = tf.layers.conv2d_transpose(l2, 128, (1, 1), 2, padding='SAME', name='up2',reuse=reuse)
        print(l2.get_shape())
        l3 = DRFB(l2, 128,64, weight[:, 18:], name='DRFB_7', reuse=reuse)
        print(l3.get_shape())
        l3 = tf.layers.conv2d_transpose(l3, 64, (1, 1), 2, padding='SAME', name='up3',reuse=reuse)
        print(l3.get_shape())
    return l3