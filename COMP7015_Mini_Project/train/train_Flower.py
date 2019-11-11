import tensorflow as tf
from COMP7015_Mini_Project.Data.Data_Loader import DataLoader_Flower
from COMP7015_Mini_Project.Data.Flower import Flower_Data_Manager
from COMP7015_Mini_Project.preprocessing import flower_preprocessing
from COMP7015_Mini_Project.train import nets_factory
import tensorflow.contrib.slim as slim
import time
import numpy as np

num_preprocessing_threads = 4
split_name = 'train'
labels_offset = 0
weight_decay = 0.00004
use_grayscale = False

num_readers = 4
batch_size = 32

def get_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return config

def run(model_name, image_size=128, checkpoints_dir=None, dataset_dir='D:/DataSet/flower_dataset/'):
    if not dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    dataset = Flower_Data_Manager.get_split(split_name, dataset_dir, file_pattern=None, reader=None)

    network_fn, network = nets_factory.get_network_fn(model_name, num_classes=Flower_Data_Manager._NUM_CLASSES,
                                                      weight_decay=weight_decay, is_training=True)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    image = flower_preprocessing.preprocess_image(image, image_size, image_size,
                                                  is_training=True,
                                                  resize_side_min=image_size,
                                                  resize_side_max=int(image_size*1.5), )

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)

    labels = slim.one_hot_encoding(labels, dataset.num_classes - labels_offset)
    logits = network_fn(images,
                        num_classes=5,
                        is_training=True,
                        batch_size=batch_size,
                        dtype=tf.float32,
                        weight_decay=weight_decay)

    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float16))

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # ------------------------------------------------------------------------------------------------
    variables_to_train = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(loss=total_loss, var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(gradients)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(100):
            err, acc, _ = sess.run(fetches=[total_loss, accuracy, grad_updates])
            print('episode: %d'%i, ' err: ',err, ' acc: ', acc)

        coord.request_stop()
        coord.join(threads)

def run_mini(model_name, image_size=128, checkpoints_dir=None):
    dataLoader = DataLoader_Flower(output_size=image_size, mini_side=int(image_size*1.3), dtype=np.float32)

    images = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
    labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, dataLoader.get_class_num()])

    network_fn = nets_factory.get_network_fn(model_name, num_classes=dataLoader.get_class_num(),
                                             weight_decay=weight_decay, is_training=True)

    logits, endpoints = network_fn(images,
                        num_classes=5,
                        is_training=True,
                        batch_size = batch_size,
                        dtype=tf.float32,
                        weight_decay=weight_decay)

    logits_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / batch_size
    total_loss = l2_loss + logits_loss

    predict = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, axis=1)), tf.float32))

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # ------------------------------------------------------------------------------------------------
    variables_to_train = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(loss=total_loss, var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(gradients)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            t = time.time()
            images_batch, labels_batch = dataLoader.get_batch(batch_size)
            _, acc, err, predict_value = sess.run(fetches=[grad_updates, accuracy, total_loss, predict], feed_dict={
                images: images_batch,
                labels: labels_batch
            })
            print('eposch: %d' % i, ' accuracy: ', acc, ' loss: ', err, ' time: ', time.time() - t)
            # print('predict_v:', predict_value.reshape([-1]))
            # print('label_v:', np.argmax(labels_batch, axis=1).reshape([-1]))
            # print('\n')

if __name__ == '__main__':
    # run('cnn')
    run_mini('mini_cnn')
