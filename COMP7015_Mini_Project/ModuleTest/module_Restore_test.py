import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project.train import nets_factory
from COMP7015_Mini_Project.tool.Ckpt_utils import *

labels_offset = 0
weight_decay = 0.00004
use_grayscale = False

num_readers = 4
batch_size = 32

trainable_scopes = None

def get_init_fn(checkpoints_dir, model_name):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = []

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    # init_op= slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, '%s.ckpt'%model_name), variables_to_restore)
    restorer = tf.train.Saver(variables_to_restore)
    return restorer

def model_Restore_Test(model_name, num_classes):
    # Select the network #
    network_fn, network = nets_factory.get_network_fn(model_name,
                                                      num_classes=(num_classes - labels_offset),
                                                      weight_decay=weight_decay, is_training=True)

    train_image_size = network_fn.default_image_size
    images = tf.placeholder(dtype=tf.float16, shape=[32, train_image_size, train_image_size, 3])
    labels = tf.placeholder(dtype=tf.float16, shape=[32, num_classes])

    logits, end_points = network_fn(images)

    t_vars = tf.trainable_variables()

    # restorer = get_init_fn('D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/', model_name=model_name)
    # ckpt_Reader = Ckpt_reader('D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/vgg_16.ckpt')

    with tf.Session() as sess:
        # restorer.restore(sess, save_path='D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/vgg_16.ckpt')
        network.load_npy(data_path='D:\HKBU_AI_Classs\COMP7015_Mini_Project\Pretrain/vgg_16.npy',
                         sess=sess, t_vars=t_vars)

        # graph = tf.get_default_graph()
        #
        # for var in t_vars:
        #     print(var.name)
        #     op = get_tensor_by_graph(graph, var.name)
        #
        #     tensor_v = sess.run(op)
        #     ckpt_v = ckpt_Reader.get_tensor_from_ckpt(var.name.replace(':0', ''))
        #
        #     print((tensor_v==ckpt_v).all())

if __name__ == '__main__':
    model_Restore_Test('vgg_16', num_classes=1000)

