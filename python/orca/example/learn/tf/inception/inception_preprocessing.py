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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.python.ops import control_flow_ops


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256


def distorted_bounding_box_crop(image_buffer,
                                num_channels,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
    image_buffer: 3-D Tensor of image.
    num_channels: num of channels
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
    Returns:
    A tuple, a 3-D Tensor cropped_image
    """
    # bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
    #       where each coordinate is [0, 1) and the coordinates are arranged
    #       as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
    #       image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32, shape=[1, 1, 4])  # From the entire image

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped_image = tf.image.decode_and_crop_jpeg(
        image_buffer, crop_window, channels=num_channels)

    return cropped_image


def _mean_image_subtraction(image, means, num_channels):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    # means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    means = tf.broadcast_to(means, tf.shape(image))

    return image - means


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


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

    return tf.image.resize(
        image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)


def preprocess_for_train(image_buffer, output_height, output_width, num_channels):

    image = distorted_bounding_box_crop(image_buffer, num_channels)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(
        image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)


def preprocess_for_eval(image_buffer, output_height, output_width, num_channels):
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])
    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
