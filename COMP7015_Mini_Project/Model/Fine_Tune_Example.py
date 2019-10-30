# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.

import tensorflow as tf
import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project import Data_Manager
from COMP7015_Mini_Project.Model.Tool import nets_factory
from COMP7015_Mini_Project.Model.Tool import preprocessing_factory

# Where the pre-trained InceptionV3 checkpoint is saved to.
pretrained_checkpoint_dir = '/tmp/checkpoints'

# Where the training (fine-tuned) checkpoint and logs will be saved to.
train_dir = '/tmp/flowers-models/inception_v3'

# Where the dataset is saved to.
dataset_dir = '/tmp/flowers'

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

    if sync_replicas:
        decay_steps /= replicas_to_aggregate

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


def main(model_name):
    if not dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = Data_Manager.get_split(split_name, dataset_dir, file_pattern=None, reader=None)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(model_name,
                                                 num_classes=(dataset.num_classes - labels_offset),
                                                 weight_decay=weight_decay, is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True,
            use_grayscale=use_grayscale)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size)

            [image, label] = provider.get(['image', 'label'])
            label -= labels_offset

            train_image_size = train_image_size or network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocessing_threads,
                capacity=5 * batch_size)

            labels = slim.one_hot_encoding(labels, dataset.num_classes - labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)
            return end_points

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            optimizer = _configure_optimizer(learning_rate)

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                total_num_replicas=FLAGS.worker_replicas,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            init_fn=_get_init_fn(),
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            sync_optimizer=optimizer if FLAGS.sync_replicas else None)
