#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import tensorflow as tf
from zoo.orca.data.image.imagenet_dataset import *
from zoo.orca.data.image.parquet_dataset import _check_arguments


def write_imagenet(imagenet_path: str,
                   output_path: str, **kwargs):
    """
    Write ImageNet data to TFRecords file format. The train and validation data will be
    converted into 1024 and 128 TFRecord files, respectively. Each train TFRecord file
    contains ~1250 records. Each validation TFRecord file contains ~390 records.

    Each record within the TFRecord file is a serialized Example proto. The Example proto
    contains the following fields:

    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/colorspace: string, specifying the colorspace, always 'RGB'
    image/channels: integer, specifying the number of channels, always 3
    image/class/label: integer, identifier for the ground truth for the network
    image/class/synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    image/format: string, specifying the format, always'JPEG'
    image/filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image/encoded: string containing JPEG encoded image in RGB colorspace

    Args:
    imagenet_path: ImageNet raw data path. Download raw ImageNet data from
                   http://image-net.org/download-images
                   e.g, if you use ImageNet 2012, please extract ILSVRC2012_img_train.tar and
                   ILSVRC2012_img_val.tar. Download the validation image labels file
                   https://github.com/tensorflow/models/blob/master/research/slim/datasets/
                   imagenet_2012_validation_synset_labels.txt and rename as synset_labels.txt
                   provide imagenet path in this format:
                   - Training images: train/n03062245/n03062245_4620.JPEG
                   - Validation Images: validation/ILSVRC2012_val_00000001.JPEG
                   - Validation Labels: synset_labels.txt
    output_path: Output data directory

    """
    if not imagenet_path:
        raise AssertionError('ImageNet data path should not be empty. Please download '
                             'from http://image-net.org/download-images and extract .tar '
                             'and provide raw data directory path')
    return convert_imagenet_to_tf_records(imagenet_path, output_path, **kwargs)


def read_imagenet(path: str,
                  is_training: bool):
    """
    Convert ImageNet TFRecords files to tf.data.Dataset

    Args:
    data_dir: ImageNet TFRecords data path. It supports local path or hdfs path. If you use
              hdfs path, please make sure set environment variables LD_LIBRARY_PATH within PATH.
            - Training images: train/train-00000-of-01024
            - Validation Images: validation/validation-00000-of-00128
    is_training: True or False. train dataset or val dataset

    """
    filenames = get_filenames(is_training, path)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Convert to individual records
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""

    _NUM_IMAGENET_TRAIN_FILES = 1024
    _NUM_IMAGENET_VAL_FILES = 128
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_IMAGENET_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(_NUM_IMAGENET_VAL_FILES)]


def write_tfrecord(format, output_path, *args, **kwargs):
    """
    Convert input dataset to TFRecords

    Args:
    format: String. Support "imagenet" format.
    output_path: String. output path.

    """
    supported_format = {"imagenet"}
    if format not in supported_format:
        raise ValueError(format + " is not supported, should be 'imagenet'. ")

    format_to_function = {"imagenet": (write_imagenet, ["imagenet_path"])}
    func, required_args = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    func(output_path=output_path, *args, **kwargs)


def read_tfrecord(format, path, *args, **kwargs):
    """
    Read TFRecords files

    Args:
    format: String. Support "imagenet" format.
    path: String. TFRecords files path.

    """
    supported_format = {"imagenet"}
    if format not in supported_format:
        raise ValueError(format + " is not supported, should be 'imagenet'. ")

    format_to_function = {"imagenet": (read_imagenet, ["is_training"])}
    func, required_args = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    return func(path=path, *args, **kwargs)
