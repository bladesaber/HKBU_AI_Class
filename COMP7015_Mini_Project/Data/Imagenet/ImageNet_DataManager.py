import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project.Data.Imagenet.ImageNet_convert import get_label_file

_FILE_PATTERN = 'tiny_imagenet_%s_*.tfrecord'

_SPLITS_TO_SIZES = {
    'train': 20580,
    'validation': 0,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

# If set to false, will not try to set label_to_names in dataset
# by reading them from labels.txt or github.
LOAD_READABLE_NAMES = True

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet."""
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names, class_names_to_ids = get_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=len(labels_to_names),
        labels_to_names=labels_to_names)

# Test
if __name__ == '__main__':
    dataset = get_split('train', 'D:/DataSet/tiny_imagenet/')
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])
