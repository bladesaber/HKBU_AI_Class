import tensorflow as tf
import os

LABELS_FILENAME = 'labels.txt'

def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    tf.train.Feature()
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_list_feature(values):
    """Returns a TF-Feature of list of bytes.
    Args:
      values: A string or list of strings.
    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_list_feature(values):
    """Returns a TF-Feature of list of floats.
    Args:
      values: A float or list of floats.
    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))

def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names
