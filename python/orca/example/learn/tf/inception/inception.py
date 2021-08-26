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
import argparse
from optparse import OptionParser
from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.metrics import *
from zoo.pipeline.nnframes import *
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator
from zoo.orca.learn.trigger import EveryEpoch, SeveralIteration
from zoo.orca.data.image import write_tfrecord, read_tfrecord
from inception_preprocessing import preprocess_for_train, \
    preprocess_for_eval
from nets import inception_v1
import tensorflow as tf
from math import ceil

slim = tf.contrib.slim

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_NUM_VAL_FILES = 128
_SHUFFLE_BUFFER = 2112

_NUM_EXAMPLES_NAME = "num_examples"

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256


def preprocess_image(image_buffer, output_height, output_width,
                     num_channels=_NUM_CHANNELS, is_training=False):
    if is_training:
        return preprocess_for_train(
            image_buffer, output_height, output_width, num_channels)
    else:
        return preprocess_for_eval(
            image_buffer, output_height, output_width, num_channels)


def _parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
    }

    # Sparse features in Example proto.

    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label


def parse_record(raw_record, is_training, dtype):
    image_buffer, label = _parse_example_proto(raw_record)

    image = preprocess_image(
        image_buffer=image_buffer,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)
    image = tf.cast(image, dtype)

    label = tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1

    return image, label


def process_record_dataset(dataset, is_training, shuffle_buffer,
                           parse_record_fn, dtype=None):
    if dtype is None:
        dtype = tf.float32

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.map(lambda value: parse_record_fn(value, is_training, dtype))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def input_fn(is_training, data_dir):
    dataset = read_tfrecord(format="imagenet", path=data_dir, is_training=is_training)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
    )


def train_data_creator(config):
    train_dataset = input_fn(is_training=True,
                             data_dir=config["data_dir"])

    return train_dataset


def val_data_creator(config):
    val_dataset = input_fn(is_training=False,
                           data_dir=config["data_dir"])

    return val_dataset


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                      help="raw ImageNet data, it includes train and val folder with format of"
                           "train/n03062245/n03062245_4620.JPEG,"
                           "validation/ILSVRC2012_val_00000001.JPEG")
    parser.add_option("--imagenet", type=str, dest="imagenet", default="/tmp/imagenet",
                      help="generated imagenet TFRecord path")
    parser.add_option("--model", type=str, dest="model", default="",
                      help="model snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="",
                      help="where to cache the model")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=1,
                      help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=3100,
                      help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01,
                      help="learning rate")
    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0,
                      help="warm up epoch numbers")
    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0,
                      help="max Lr after warm up")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001,
                      help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")
    parser.add_option("--resumeTrainingCheckpoint", type=str, dest="resumeTrainingCheckpoint",
                      default=None,
                      help="an analytics zoo checkpoint path for resume training, usually contains"
                           + "a file named model.$iter_num and a file named"
                           + " optimMethod-TFParkTraining.$iter_num")
    parser.add_option("--resumeTrainingVersion", type=int, dest="resumeTrainingVersion",
                      default=None,
                      help="the version of checkpoint file, should be the $iter_num"
                           + " in model.$iter_num")
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The mode for the Spark cluster.')
    parser.add_option("--worker_num", type=int, default=1,
                      help="The number of slave nodes to be used in the cluster."
                           "You can change it depending on your own cluster setting.")
    parser.add_option("--cores", type=int, default=4,
                      help="The number of cpu cores you want to use on each node. "
                           "You can change it depending on your own cluster setting.")
    parser.add_option("--memory", type=str, default="10g",
                      help="The memory you want to use on each node. "
                           "You can change it depending on your own cluster setting.")

    return parser


if __name__ == "__main__":
    parser = config_option_parser()
    (options, args) = parser.parse_args(sys.argv)

    write_tfrecord(format="imagenet", imagenet_path=options.folder, output_path=options.imagenet)

    train_data = train_data_creator(
        config={"data_dir": os.path.join(options.imagenet, "train")})
    val_data = val_data_creator(
        config={"data_dir": os.path.join(options.imagenet, "validation")})

    num_nodes = 1 if options.cluster_mode == "local" else options.worker_num
    init_orca_context(cluster_mode=options.cluster_mode, cores=options.cores, num_nodes=num_nodes,
                      memory=options.memory)

    images = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    labels = tf.placeholder(dtype=tf.int32, shape=(None))
    is_training = tf.placeholder(dtype=tf.bool, shape=())

    with slim.arg_scope(inception_v1.inception_v1_arg_scope(weight_decay=0.0,
                                                            use_batch_norm=False)):
        logits, end_points = inception_v1.inception_v1(images,
                                                       dropout_keep_prob=0.6,
                                                       num_classes=1000,
                                                       is_training=is_training)
        probabilities = tf.nn.softmax(logits)
        print("probabilities", probabilities)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                                 labels=labels))

    from zoo.orca.learn.optimizers.schedule import SequentialSchedule, Warmup, Poly
    from zoo.orca.learn.optimizers import SGD

    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
    if options.maxEpoch:
        maxIteration = iterationPerEpoch * options.maxEpoch
    else:
        maxIteration = options.maxIteration
    warmup_iteration = options.warmupEpoch * iterationPerEpoch

    if warmup_iteration == 0:
        warmupDelta = 0.0
    else:
        if options.maxLr:
            maxlr = options.maxLr
        else:
            maxlr = options.learningRate
        warmupDelta = (maxlr - options.learningRate) / warmup_iteration
    polyIteration = maxIteration - warmup_iteration
    lrSchedule = SequentialSchedule(iterationPerEpoch)
    lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
    lrSchedule.add(Poly(0.5, maxIteration), polyIteration)
    optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0,
                weightdecay=options.weightDecay, momentum=0.9, dampening=0.0,
                nesterov=False, learningrate_schedule=lrSchedule)

    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)

    def calculate_top_k_accuracy(logits, targets, k=1):
        values, indices = tf.math.top_k(logits, k=k, sorted=True)
        y = tf.reshape(targets, [-1, 1])
        correct = tf.cast(tf.equal(y, indices), tf.float32)
        top_k_accuracy = tf.reduce_mean(correct) * k
        return top_k_accuracy

    acc = calculate_top_k_accuracy(logits, targets=labels)

    est = Estimator.from_graph(inputs=images,
                               outputs=logits,
                               labels=labels,
                               loss=loss,
                               optimizer=optim,
                               model_dir="/tmp/logs",
                               metrics={"acc": acc})

    if options.resumeTrainingCheckpoint is not None:
        assert options.resumeTrainingVersion is not None, \
            "--resumeTrainingVersion must be specified when --resumeTrainingCheckpoint is."
        est.load_orca_checkpoint(options.resumeTrainingCheckpoint,
                                 options.resumeTrainingVersion)

    est.fit(data=train_data,
            batch_size=options.batchSize,
            epochs=options.maxEpoch,
            validation_data=val_data,
            feed_dict={is_training: [True, False]},
            checkpoint_trigger=checkpoint_trigger)

    if options.checkpoint:
        saver = tf.train.Saver()
        saver.save(est.sess, options.checkpoint)

    stop_orca_context()
