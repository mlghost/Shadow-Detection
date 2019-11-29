import tensorflow as tf
x=tf.constant(3)
with tf.Session() as sess:
    print(sess.run(x))
