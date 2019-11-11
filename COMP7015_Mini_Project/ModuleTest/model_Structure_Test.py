import tensorflow as tf
from COMP7015_Mini_Project.train import nets_factory

labels_offset = 0
weight_decay = 0.00004

learning_rate = 0.01
num_epochs_per_decay = 2.0
learning_rate_decay_type = 'exponential'
learning_rate_decay_factor = 0.94
end_learning_rate = 0.0001

trainable_scopes = None

# ----------------------------------------------------------------------------------------------------------------------
def model_Structure_Test(model_name, num_classes, dataset_dir='D:/DataSet/flower_dataset/'):
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