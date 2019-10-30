import functools
from COMP7015_Mini_Project.Model import InceptionV3
from COMP7015_Mini_Project.Model import ResnetV1_50
from COMP7015_Mini_Project.Model import Vgg_16
import tensorflow.contrib.slim as slim

networks_map = {'vgg_16': Vgg_16.vgg_16,
                'inception_v3': InceptionV3.inception_v3,
                'resnet_v1_50': ResnetV1_50.resnet_v1_50,}

arg_scopes_map = {'vgg_16': Vgg_16.vgg_arg_scope,
                  'inception_v3': InceptionV3.inception_v3_arg_scope,
                  'resnet_v1_50': ResnetV1_50.resnet_arg_scope,}

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes=num_classes, is_training=is_training, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
