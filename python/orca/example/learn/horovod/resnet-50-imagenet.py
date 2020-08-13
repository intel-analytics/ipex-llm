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

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray import RayContext
import os
import argparse

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500

_NUM_EXAMPLES_NAME = "num_examples"

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256


def _decode_crop_and_flip(image_buffer, num_channels):
    min_object_covered = 0.1
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32, shape=[1, 1, 4])  # From the entire image
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        image_size=tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(
        image_buffer, crop_window, channels=num_channels)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    return tf.image.resize(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR)


def preprocess_image(image_buffer, output_height, output_width,
                     num_channels, is_training=False):
    if is_training:
        # For training, we want to randomize some of the distortions.
        image = _decode_crop_and_flip(image_buffer, num_channels)

        image = _resize_image(image, output_height, output_width)
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(128)]


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
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label


def parse_record(raw_record, is_training, dtype):
    image_buffer, label = _parse_example_proto(raw_record)

    image = preprocess_image(
        image_buffer=image_buffer,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    return image, label


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, dtype=None):
    if dtype is None:
        dtype = tf.float32
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat()
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=tf.data.experimental.AUTOTUNE))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def input_fn(is_training, data_dir, batch_size):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
    )


def model_creator(config):
    wd = config["wd"]
    import tensorflow as tf
    import tensorflow.keras as keras
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_bfloat16')
    # policy = mixed_precision.Policy('float32')
    # mixed_precision.set_policy(policy)

    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=1001)
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(wd)
            regularizer_config = {'class_name': regularizer.__class__.__name__,
                                  'config': regularizer.get_config()}
            layer_config['config']['kernel_regularizer'] = regularizer_config
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = tf.keras.models.Model.from_config(model_config)
    return model


def compile_args_creator(config):
    momentum = config["momentum"]
    import tensorflow.keras as keras
    opt = keras.optimizers.SGD(momentum=momentum)
    param = dict(loss=keras.losses.sparse_categorical_crossentropy, optimizer=opt,
                 metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])
    return param


def train_data_creator(config):
    train_dataset = input_fn(is_training=True,
                             data_dir=config["data_dir"],
                             batch_size=config["batch_size"])

    return train_dataset


def val_data_creator(config):
    val_dataset = input_fn(is_training=False,
                           data_dir=config["data_dir"],
                           batch_size=config["batch_size"])

    return val_dataset


def schedule(epoch, lr_multiplier):
    if epoch < 5:
        return 0.1 + 0.1 * (lr_multiplier - 1) * epoch / 5

    if 5 <= epoch < 30:
        return 0.1 * lr_multiplier

    if 30 <= epoch < 60:
        return 0.1 * 0.1 * lr_multiplier

    if 60 <= epoch < 80:
        return 0.1 * 0.01 * lr_multiplier

    return 0.1 * 0.001 * lr_multiplier


parser = argparse.ArgumentParser()
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                         " configuration folder. Otherwise, turn on local mode.")
parser.add_argument("--worker_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--conda_name", type=str,
                    help="The name of conda environment.")
parser.add_argument("--executor_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--executor_memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_memory", type=str, default="2g",
                    help="The size of driver's memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--extra_executor_memory_for_ray", type=str, default="20g",
                    help="The extra executor memory to store some data."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--object_store_memory", type=str, default="4g",
                    help="The memory to store data on local."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--batch_size_per_worker", type=int, default=256)
parser.add_argument("--data_dir", type=str, help="the directory of tfrecords of imagenet, follow"
                                                 " https://github.com/IntelAI/models/blob/"
                                                 "v1.6.1/docs/image_recognition/tensorflow/"
                                                 "Tutorial.md#initial-setup for generating"
                                                 " data file")

if __name__ == "__main__":

    args = parser.parse_args()
    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name=args.conda_name,
            num_executors=args.worker_num,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            driver_cores=args.driver_cores,
            extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()
    else:
        sc = init_spark_on_local()
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()

    assert args.data_dir is not None, "--data_dir must be provided"

    from zoo.orca.learn.tf2 import Estimator
    import tensorflow as tf

    global_batch_size = args.worker_num * args.batch_size_per_worker

    base_batch_size = 256

    lr_multiplier = global_batch_size // base_batch_size

    lr_schdule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: schedule(epoch, lr_multiplier), verbose=1)

    config = {
        "momentum": 0.9,
        "wd": 0.00005,
        "batch_size": args.batch_size_per_worker,
        "val_batch_size": args.batch_size_per_worker,
        "warmup_epoch": 5,
        "num_worker": args.worker_num,
        "data_dir": args.data_dir,
    }

    trainer = Estimator(
        model_creator=model_creator,
        compile_args_creator=compile_args_creator,
        verbose=True,
        config=config, backend="horovod")

    results = trainer.fit(
        data_creator=train_data_creator,
        epochs=90,
        validation_data_creator=val_data_creator,
        steps_per_epoch=_NUM_IMAGES['train'] // global_batch_size,
        callbacks=[lr_schdule],
        validation_steps=_NUM_IMAGES['validation'] // global_batch_size,
    )

    print(results)
