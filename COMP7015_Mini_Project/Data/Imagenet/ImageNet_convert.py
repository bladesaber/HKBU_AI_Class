import os
import tensorflow as tf
import COMP7015_Mini_Project.Data.data_utils as data_utils
import tarfile
import glob
import random
import math

_RANDOM_SEED = 0

_NUM_SHARDS = 20

_file_pattern = '%s_boxes.txt'

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'tiny_imagenet_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

def un_tar(file_name):
    # untar zip file"""
    tar = tarfile.open(file_name)
    names = tar.getnames()
    directory = file_name.replace('.tar','')
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)
    for name in names:
        tar.extract(name, directory)
    tar.close()
    os.remove(file_name)

def extract_data(data_dir):
    files = glob.glob(os.path.join(data_dir,'*.tar'))
    for file in files:
        un_tar(file)

def image_to_tfexample(image_data, image_format, height, width, class_id, label_text):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': data_utils.bytes_feature(image_data),
        'image/format': data_utils.bytes_feature(image_format),
        'image/height': data_utils.int64_feature(height),
        'image/width': data_utils.int64_feature(width),
        'image/class/label': data_utils.int64_feature(class_id),
        'image/class/text': data_utils.bytes_feature(label_text),
    }))

def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string."""

    with open('D:/DataSet/tiny_imagenet/snyset.txt', 'r') as f:
        synset_to_human = {}
        for s in f.readlines():
            parts = s.strip().split(',')
            # print(parts[0], parts[1])
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human

    label_index = 1
    # labels_to_names = {0: 'background'}
    labels_to_names = {}
    for synset in synset_to_human:
        # name = synset_to_human[synset]
        labels_to_names[label_index] = synset
        label_index += 1

    return labels_to_names

def get_label_file(dataset_dir):
    if data_utils.has_labels(dataset_dir):
        labels_to_names = data_utils.read_label_file(dataset_dir)
    else:
        labels_to_names = create_readable_names_for_imagenet_labels()
        data_utils.write_label_file(labels_to_names, dataset_dir=dataset_dir)

    if data_utils.has_labels(dataset_dir, filename='name_to_label.txt'):
        with open(os.path.join(dataset_dir, 'name_to_label.txt'), 'r') as f:
            name_to_labels = eval(f.read())
            f.close()
    else:
        name_to_labels = {}
        for key in labels_to_names:
            name_to_labels[labels_to_names[key]] = key
        with open(os.path.join(dataset_dir, 'name_to_label.txt'), 'w') as f:
            f.write(str(name_to_labels))
            f.close()

    return labels_to_names, name_to_labels

def _get_filenames_and_classes(dataset_dir):
    root_directory = os.path.join(dataset_dir, 'tiny_imagenet')
    root = os.path.join(root_directory, 'image')
    directories = []
    class_names = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset."""
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        print('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))

                        # Read the filename:
                        image_data = tf.gfile.GFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = image_to_tfexample(image_data, b'jpg',height, width, class_id, bytes(class_name, encoding = "utf8"))
                        tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir):
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)

    root_directory = os.path.join(dataset_dir, 'tiny_imagenet')
    labels_to_names, class_names_to_ids = get_label_file(root_directory)

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    _convert_dataset('train', photo_filenames, class_names_to_ids, dataset_dir)

if __name__ == '__main__':
    # extract_data('D:\DataSet/timy_imagenet/image')
    # get_label_file('D:\DataSet/timy_imagenet')
    # run('D:\DataSet/')

    pass