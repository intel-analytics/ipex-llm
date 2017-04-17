#
# Copyright 2016 The BigDL Authors.
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


from util.common import Sample
import random
import numpy as np


def normalizer(mean, std):
    """
    Normalize features by standard deviation
    """
    return lambda sample: Sample.from_ndarray((sample.features - mean) / std,
                                              sample.label, sample.bigdl_type)


def pixel_normalizer(mean):
    """
    Normalize with pixel mean array
    :param mean: mean ndarray for each pixel
    :return:
    """
    return lambda sample: Sample((sample.features - mean),
                                              sample.label, sample.bigdl_type)


def crop(crop_width, crop_height, crop_method='random'):
    """
    Crop image sample to specified width and height
    :param crop_width: width after cropped
    :param crop_height: height after cropped
    :param crop_method: crop method, should be random or center
    :return: cropped image sample
    """
    def func(sample):
        h = sample.features.shape[0]
        w = sample.features.shape[1]
        if crop_method == 'random':
            x1 = random.randint(0, w - crop_width)
            y1 = random.randint(0, h - crop_height)
        elif crop_method == 'center':
            x1 = (w - crop_width)/2
            y1 = (h - crop_height)/2
        cropped = sample.features[y1:y1+crop_height, x1:x1+crop_width]
        return Sample(cropped, sample.label, sample.bigdl_type)
    return func


def channel_normalizer(mean_r, mean_g, mean_b, std_r, std_g, std_b):
    """
    Normalize image sample by means and std of each channel
    :param mean_r: mean for red channel
    :param mean_g: mean for green channel
    :param mean_b: mean for blue channel
    :param std_r: std for red channel
    :param std_g: std for green channel
    :param std_b: std for blue channel
    :return: normalized image sample
    """
    def func(sample):
        mean = np.array([mean_b, mean_g, mean_r])
        std = np.array([std_b, std_g, std_r])
        mean_sub = sample.features[:, :] - mean
        result = mean_sub[:, :] / std
        return Sample(result, sample.label, sample.bigdl_type)
    return func


def flip(threshold):
    """
    Flip image sample horizontally
    :param threshold: if random number is over the threshold, we need to flip image, otherwise, do not filp
    :return: flipped image sample or original image sample
    """
    def func(sample):
        if random.random() > threshold:
            # flip with axis 1 which is horizontal flip
            return Sample(np.flip(sample.features, 1), sample.label, sample.bigdl_type)
        else:
            return sample
    return func


def transpose(to_rgb=True):
    """
    Transpose the shape of image sample from (height, width, channel) to (channel, height, width)
    :param to_rgb: whether need to change channel from bgr to rgb
    :return: transposed image sample
    """
    def func(sample):
        if to_rgb:
            result = sample.features.transpose(2, 0, 1)[(2, 1, 0), :, :]
        else:
            result = sample.features.transpose(2, 0, 1)
        return Sample(result, sample.label,sample.bigdl_type)
    return func

