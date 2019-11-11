from COMP7015_Mini_Project.train import nets_factory
from COMP7015_Mini_Project.Data import Data_Loader
import time
import tensorflow as tf
import math
import numpy as np

num_preprocessing_threads = 4
split_name = 'train'
labels_offset = 0
weight_decay = 0.00004
use_grayscale = False

num_readers = 4
batch_size = 64

def get_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return config

def run(model_name, epoch, checkpoints_dir=None):
    dataLoader = Data_Loader.DataLoader_Cifar10()

    images = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])
    labels = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 1])
    is_trainning_placeholder = tf.placeholder(dtype=tf.bool, name='is_trainning_bool')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learn_rate')

    network_fn, network = nets_factory.get_network_fn(model_name, weight_decay=weight_decay,
                                             num_classes=dataLoader.get_class_num(), is_training=is_trainning_placeholder)

    logits, endpoints = network_fn(images,
                                   num_classes=dataLoader.get_class_num(),
                                   is_training=is_trainning_placeholder,
                                   batch_size=batch_size,
                                   dtype=tf.float32)

    # -------------------------------------------  summaries trigger
    # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # for end_point in endpoints:
    #     x = endpoints[end_point]
    #     summaries.add(tf.summary.histogram(end_point+'_distribution', x))
    #     summaries.add(tf.summary.scalar(end_point+'_sparsity', tf.nn.zero_fraction(x)))
    # ----------------------------------------------------------------------------------------

    target_labels = tf.one_hot(labels, depth=dataLoader.get_class_num(), dtype=tf.float32)
    target_labels = tf.reshape(target_labels, shape=[batch_size, dataLoader.get_class_num()])
    predict = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)

    logits_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target_labels))
    # logits_loss = tf.reduce_sum(tf.abs(tf.nn.softmax(logits, axis=1) - target_labels))
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / batch_size
    total_loss = l2_loss + logits_loss

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predict, tf.argmax(target_labels, axis=1)), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # ------------------------------------------------------------------------------------------------
    variables_to_train = tf.trainable_variables()

    # -------------------------------------------  summaries trigger
    # for variable in slim.get_model_variables():
    #     summaries.add(tf.summary.histogram(variable.op.name, variable))
    # summary_op = tf.summary.merge(list(summaries), name='summary_op')
    # ---------------------------------------------------------------------------

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(loss=total_loss, var_list=variables_to_train)

        # gradient clip
        # for i, (g, v) in enumerate(gradients):
        #     if g is not None:
        #         gradients[i] = (tf.clip_by_norm(g, 5.), v)
        #         # gradients[i] = (tf.clip_by_value(g, 0.001, 5.0), v)

        grad_updates = optimizer.apply_gradients(gradients)

        # claculate_grads = tf.gradients(total_loss, variables_to_train)

    # -----------------------    evaluation
    train_accuracy_record, time_record, train_cost_record = [], [], []
    test_accuracy_record, test_cost_record = [], []

    batch_num_per_epoch = math.ceil(dataLoader.train_num/batch_size)
    print('train step: %d'%(epoch*batch_num_per_epoch))

    saver = tf.train.Saver()

    learn_rate = 0.001

    with tf.Session(config=get_config()) as sess:
        sess.run(tf.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # train_writer = tf.summary.FileWriter('D:/HKBU_AI_Classs/COMP7015_Mini_Project/PersonNetwork/Log', sess.graph)
        # images_batch, labels_batch = dataLoader.get_batch(batch_size)
        #
        # for k in range(4):
        #     _, summury_operation, err = sess.run(fetches=[grad_updates, summary_op, total_loss],
        #                                     feed_dict={
        #                                         images: images_batch,
        #                                         labels: labels_batch
        #                                     })
        #     train_writer.add_summary(summury_operation, k)
        #
        #     # grads, logits_err, l2_err, endpoints_dict, predict_value, grads_compute = sess.run(
        #     #     fetches=[claculate_grads, logits_loss, l2_loss, endpoints, predict, gradients],
        #     #     feed_dict={
        #     #         images: images_batch,
        #     #         labels: labels_batch
        #     #     })
        #     #
        #     # for m in range(len(grads_compute)):
        #     #     for n in range(len(grads_compute[m])):
        #     #         value = grads_compute[m][n]
        #     #         name = gradients[m][n].name
        #     #         print(name, ' shape:', value.shape, ' mean:', np.mean(value),
        #     #               ' max:', np.max(value), ' min:', np.min(value))
        #
        #     # print('logits loss:', logits_err, ' l2 loss:', l2_err, ' predice:', predict_value)
        #     # for i in range(len(grads)):
        #     #     g = grads[i]
        #     #     print('index: ', claculate_grads[i].name, ' shape:', g.shape,
        #     #           ' grad mean:', np.mean(g), ' grad std:', np.std(g),
        #     #           ' grad max:', np.max(g), ' grad min:', np.min(g))
        #
        #     # for i in endpoints_dict.keys():
        #     #     print(i, ' shape:', endpoints_dict[i].shape, ' max:', np.max(endpoints_dict[i]), ' min:',
        #     #           np.min(endpoints_dict[i]),
        #     #           ' mean:', np.mean(endpoints_dict[i]))
        #
        #     print('---------------------------------------')
        # --------------------------------------------------------------------------------------------------------------

        t = time.time()
        for i in range(epoch*batch_num_per_epoch):

            if (i+1)%batch_num_per_epoch==0:
                learn_rate = learn_rate*0.1

            images_batch, labels_batch = dataLoader.get_batch(batch_size)
            _, acc, err, predict_value = sess.run(fetches=[grad_updates, accuracy, total_loss, predict],
                     feed_dict={
                         images: images_batch,
                         labels: labels_batch,
                         is_trainning_placeholder:True,
                         learning_rate: learn_rate
                     })
            cost_time = time.time()-t
            print('eposch: %d'%i, ' accuracy: ',acc,' loss: ', err, ' time: ', cost_time)

            train_accuracy_record.append(acc)
            time_record.append(cost_time)
            train_cost_record.append(err)

            if (i+1)%100==0:
                test_images, test_labels = dataLoader.get_test_batch(512)
                test_acc, test_err = sess.run(fetches=[accuracy, total_loss],
                                    feed_dict={images: test_images,labels: test_labels, is_trainning_placeholder:False})
                test_accuracy_record.append(test_acc)
                test_cost_record.append(test_err)

            if (i+1)%50==0:
                saver.save(sess, save_path='D:\HKBU_AI_Classs\COMP7015_Mini_Project\Log/mini_cnn.ckpt',
                           write_state=False, write_meta_graph=False)

        save_dict = {
            'train_acc':np.array(train_accuracy_record),
            'time': np.array(time_record),
            'train_cost': np.array(train_cost_record),
            'test_acc': np.array(test_accuracy_record),
            'test_cost': np.array(test_cost_record)
        }
        np.save('D:/HKBU_AI_Classs/COMP7015_Mini_Project/Log/mini_cnn.npy', save_dict)

if __name__ == '__main__':
    run('mini_cnn', epoch=2)
