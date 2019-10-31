import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

x = np.random.uniform(-2.0, 2.0, size=[200]).reshape([200, 1]).astype(np.float32)
y = (np.square(x) + x).reshape([200, 1])

tf.logging.set_verbosity(tf.logging.INFO)

input = tf.constant(value=x)
y_target = tf.constant(value=y)

with slim.arg_scope([slim.fully_connected], normalizer_fn=slim.batch_norm):
    net = slim.fully_connected(input, num_outputs=32)
    net = slim.fully_connected(net, num_outputs=1, activation_fn=None)

total_loss = tf.reduce_mean(tf.square(y_target - net))

optimizer = tf.train.AdamOptimizer(learning_rate=0.03)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    gradients = optimizer.compute_gradients(loss=total_loss)
    grad_updates = optimizer.apply_gradients(gradients)
