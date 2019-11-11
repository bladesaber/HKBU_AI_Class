from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np

class Ckpt_reader:
    def __init__(self, checkpoint_path):
        self.reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        self.var_to_shape_map = self.reader.get_variable_to_shape_map()

    def keys(self):
        return self.var_to_shape_map.keys()

    def get_tensor_from_ckpt(self, tensor_name):
        return self.reader.get_tensor(tensor_name)

def get_tensor_by_graph(graph, tensor_name):
    return graph.get_tensor_by_name(tensor_name)

def adjust_ckpt(checkpoint_path):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

def paras_count(t_vars):
    return np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])

# if __name__ == '__main__':
#     adjust_ckpt('D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/vgg_16.ckpt')
