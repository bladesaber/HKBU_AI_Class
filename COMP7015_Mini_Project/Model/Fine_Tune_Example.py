# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.

import tensorflow as tf
import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project import Data_Manager
from COMP7015_Mini_Project.Model.Tool import nets_factory
from COMP7015_Mini_Project.Model.Tool import preprocessing_factory
import os

# Where the pre-trained InceptionV3 checkpoint is saved to.
pretrained_checkpoint_dir = '/tmp/checkpoints'

# Where the training (fine-tuned) checkpoint and logs will be saved to.
train_dir = '/tmp/flowers-models/inception_v3'

# Where the dataset is saved to.
dataset_dir = 'D:/DataSet/flower_dataset/'

num_preprocessing_threads = 4
split_name = 'train'
labels_offset = 0
weight_decay = 0.00004
use_grayscale = False

num_readers = 4
batch_size = 32

learning_rate = 0.01
num_epochs_per_decay = 2.0
learning_rate_decay_type = 'exponential'
learning_rate_decay_factor = 0.94
end_learning_rate = 0.0001

trainable_scopes = None

def _get_variables_to_train():
    if trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def _configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch * num_epochs_per_decay / batch_size)

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

    elif learning_rate_decay_type == 'fixed':
        return tf.constant(learning_rate, name='fixed_learning_rate')

    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' % learning_rate_decay_type)

def _configure_optimizer(learning_rate, optimizer='adam'):
    if optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.99,
            epsilon=1.0)
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0)
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % optimizer)
    return optimizer


def get_init_fn(checkpoints_dir, model_name):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, '%s.ckpt'%model_name), variables_to_restore)

def main(model_name, checkpoints_dir):
    if not dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # Create global_step
    global_step = tf.train.create_global_step()

    # Select the dataset #
    dataset = Data_Manager.get_split(split_name, dataset_dir, file_pattern=None, reader=None)

    # Select the network #
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=(dataset.num_classes - labels_offset),
                                             weight_decay=weight_decay, is_training=True)

    # Select the preprocessing function #
    preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True,
        use_grayscale=use_grayscale)

    # Create a dataset provider that loads data from the dataset #
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    train_image_size = network_fn.default_image_size

    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)

    labels = slim.one_hot_encoding(labels, dataset.num_classes - labels_offset)
    logits, end_points = network_fn(images)

    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
    total_loss = entropy_loss + regularization_loss

    learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
    optimizer = _configure_optimizer(learning_rate)

    # ------------------------------------------------------------------------------------------------
    # variables_to_train = _get_variables_to_train()
    #
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     gradients = optimizer.compute_gradients(loss=total_loss, var_list=variables_to_train)
    #     grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)

    # -----------------------------------------------------------------------------------------------
    tf.logging.set_verbosity(tf.logging.INFO)
    # Specify the optimizer and create the train op:
    train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)

    final_loss = slim.learning.train(
        train_op,
        logdir=train_dir,
        init_fn=get_init_fn(checkpoints_dir, model_name),
        number_of_steps=2)

def dataset_test(model_name, image_size):
    import matplotlib.pyplot as plt
    import numpy as np

    if not dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # Select the dataset #
    dataset = Data_Manager.get_split(split_name, dataset_dir, file_pattern=None, reader=None)

    # Select the preprocessing function #
    preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True,
        use_grayscale=use_grayscale)

    # Create a dataset provider that loads data from the dataset #
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    image = image_preprocessing_fn(image, image_size, image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("D:/HKBU_AI_Classs/COMP7015_Mini_Project/Model/Log/", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())

        imgs, signs = sess.run(fetches=[images, labels])

        coord.request_stop()
        coord.join(threads)

    for i in range(len(signs)):
        print('label %d : ' % i, signs[i])

        imgs[i] = (imgs[i] - np.min(imgs[i])) / (np.max(imgs[i]) - np.min(imgs[i]))
        plt.imshow(imgs[i])
        plt.show()

def model_test(model_name, num_classes):
    if not dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # Create global_step
    global_step = tf.train.create_global_step()

    # Select the network #
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=(num_classes - labels_offset),
                                             weight_decay=weight_decay, is_training=True)

    train_image_size = network_fn.default_image_size

    images = tf.placeholder(dtype=tf.float32, shape=[32, train_image_size, train_image_size, 3])
    labels = tf.placeholder(dtype=tf.float32, shape=[32, num_classes])

    logits, end_points = network_fn(images)

    # Print endpoint
    # for key in end_points:
    #     print(key, end_points[key])

    entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
    total_loss = entropy_loss + regularization_loss

    # write the graph
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     writer = tf.summary.FileWriter("D:/HKBU_AI_Classs/COMP7015_Mini_Project/Model/Log/", sess.graph)

    # tensorboard --logdir="log"

if __name__ == '__main__':
    # main(model_name='vgg_16')

    # model_test('vgg_16', num_classes=1000)

    pass
