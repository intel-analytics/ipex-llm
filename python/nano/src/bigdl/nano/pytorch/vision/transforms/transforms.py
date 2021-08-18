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


import torchvision.transforms as tv_t
import opencv_transforms.transforms as cv_t
import cv2
import numpy as np
import numbers
import random
import warnings
import torch
from typing import Tuple, List, Optional
import collections
from torchvision.transforms.functional import InterpolationMode

_cv_strToModes_mapping = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}
_torch_intToModes_mapping = {
    0: InterpolationMode.NEAREST,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
    4: InterpolationMode.BOX,
    5: InterpolationMode.HAMMING,
    1: InterpolationMode.LANCZOS,
}

_torch_strToModes_mapping = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'box': InterpolationMode.BOX,
    'hamming': InterpolationMode.HAMMING,
    'lanczos': InterpolationMode.LANCZOS,
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int."
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_intToModes_mapping[interpolation]

        if isinstance(interpolation, str):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of str."
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_strToModes_mapping[interpolation]

        if interpolation in _torch_intToModes_mapping.values():
            self.tv_F = tv_t.Resize(size, interpolation)
        else:
            self.tv_F = tv_t.Resize(size, InterpolationMode.BILINEAR)
        if interpolation in _cv_strToModes_mapping:
            self.cv_F = cv_t.Resize(size, _cv_strToModes_mapping[interpolation])
        else:
            self.cv_F = cv_t.Resize(size, cv2.INTER_LINEAR)
        self.interpolation = interpolation

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        interpolation_str = self.__class__.__name__
        interpolation_str += '(size={0}, interpolation={1})'.format(self.size, self.interpolation)
        return interpolation_str


class ToTensor(object):
    def __init__(self):
        self.tv_F = tv_t.ToTensor()
        self.cv_F = cv_t.ToTensor()

    def __call__(self, pic):
        if type(pic) == np.ndarray:
            return self.cv_F.__call__(pic)
        else:
            return self.tv_F.__call__(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.tv_F = tv_t.RandomHorizontalFlip(self.p)
        self.cv_F = cv_t.RandomHorizontalFlip(self.p)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        self.tv_F = tv_t.RandomCrop(size, padding, pad_if_needed, fill, padding_mode)
        self.cv_F = cv_t.RandomCrop(size, padding, pad_if_needed, fill, padding_mode)

    @staticmethod
    def get_params(img, output_size):
        if type(img) == np.ndarray:
            return self.cv_F.get_params(img, output_size)
        else:
            return self.tv_F.get_params(img, output_size)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
            self.size, self.padding)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.tv_F = tv_t.ColorJitter(brightness, contrast, saturation, hue)
        self.cv_F = cv_t.ColorJitter(brightness, contrast, saturation, hue)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        return self.tv_F.get_params(brightness, contrast, saturation, hue)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class Normalize(tv_t.Normalize):
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__(mean, std, inplace)
