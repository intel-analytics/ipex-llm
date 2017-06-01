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

import random
import numpy as np
from scipy import misc


class ImgTransformer(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> ImgTransformer([
        >>>     Crop(10, 10),
        >>>     Normalizer(0.3, 0.8)
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Crop(object):
    """
    Crop image which is an numpy array to specified width and height. The shape of image is (height, width, channel).
    :param crop_width: width after cropped
    :param crop_height: height after cropped
    :param crop_method: crop method, should be random or center
    :return: cropped image. The shape of image is (height, width, channel)
    """

    def __init__(self, crop_width, crop_height, crop_method='random'):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_method = crop_method

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        if self.crop_method == 'random':
            x1 = random.randint(0, w - self.crop_width)
            y1 = random.randint(0, h - self.crop_height)
        elif self.crop_method == 'center':
            x1 = (w - self.crop_width) / 2
            y1 = (h - self.crop_height) / 2
        cropped = img[y1:y1 + self.crop_height, x1:x1 + self.crop_width]
        return cropped


class Normalizer(object):
    """
    Normalize image which is an numpy array by mean and standard deviation The shape of image is (height, width, channel)
    :param mean: mean
    :param std: standard deviation
    :return: normalized image. The shape of image is (height, width, channel)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std


class PixelNormalizer(object):
    """
    Normalize image which is an numpy array by means of every pixel. The shape of image is (height, width, channel)
    :param mean: mean
    :param std: standard deviation
    :return: normalized image. The shape of image is (height, width, channel)
    """

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img):
        return img - self.mean


class ChannelNormalizer(object):
    """
    Normalize image which is an numpy array by means and std of each channel. The shape of image is (height, width, channel)
    :param mean_r: mean for red channel
    :param mean_g: mean for green channel
    :param mean_b: mean for blue channel
    :param std_r: std for red channel
    :param std_g: std for green channel
    :param std_b: std for blue channel
    :return: normalized image. The shape of image is (height, width, channel)
    """
    def __init__(self, mean_r, mean_g, mean_b, std_r, std_g, std_b):
        self.mean_r = mean_r
        self.mean_g = mean_g
        self.mean_b = mean_b
        self.std_r = std_r
        self.std_g = std_g
        self.std_b = std_b

    def __call__(self, img):
        mean = np.array([self.mean_b, self.mean_g, self.mean_r])
        std = np.array([self.std_b, self.std_g, self.std_r])
        mean_sub = img[:, :] - mean
        return mean_sub[:, :] / std


class Flip(object):
    """
    Flip image which is an numpy array horizontally.  The shape of image is (height, width, channel)
    :param threshold: if random number is over the threshold, we need to flip image, otherwise, do not filp
    :return: flipped image or original image. The shape of image is (height, width, channel)
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        if random.random() > self.threshold:
            # flip with axis 1 which is horizontal flip
            return np.flip(img, 1)
        else:
            return img


class TransposeToTensor(object):
    """
    Transpose the shape of image which is an numpy array from (height, width, channel) to (channel, height, width)
    :param to_rgb: whether need to change channel from bgr to rgb
    :return: transposed image
    """
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, img):
        if self.to_rgb:
            return img.transpose(2, 0, 1)[(2, 1, 0), :, :]
        else:
            return img.transpose(2, 0, 1)


class Resize(object):
    """
    Resize image to specified width and height. The type of image should be uint8.
    The shape of image is (height, width, channel)
    :param resize_width: the width resized to
    :param resize_height: the height resized to
    :return: resized image.
    """
    def __init__(self, resize_width, resize_height):
        self.resize_width = resize_width
        self.resize_height = resize_height

    def __call__(self, img):
        return misc.imresize(img, (self.resize_width, self.resize_height))
