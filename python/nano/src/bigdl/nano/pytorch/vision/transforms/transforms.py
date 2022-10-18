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


import cv2
import collections
import numbers
import random
import torch
import types
import warnings

import numpy as np
import torchvision.transforms as tv_t
from torchvision.transforms.functional import InterpolationMode
import opencv_transforms.transforms as cv_t
from bigdl.nano.utils.log4Error import invalidInputError


__all__ = [
    'Compose',
    'Resize',
    'Scale',
    'ToTensor',
    'RandomHorizontalFlip',
    'RandomCrop',
    'ColorJitter',
    'Normalize',
    'PILToTensor',
    'ConvertImageDtype',
    'ToPILImage',
    'CenterCrop',
    'Pad',
    'Lambda',
    'RandomApply',
    'RandomOrder',
    'RandomChoice',
    'RandomVerticalFlip',
    'RandomResizedCrop',
    'RandomSizedCrop',
    'FiveCrop',
    'TenCrop',
    'LinearTransformation',
    'RandomRotation',
    'RandomAffine',
    'Grayscale',
    'RandomGrayscale',
    'RandomPerspective',
    'RandomErasing',
    'GaussianBlur',
    'RandomInvert',
    'RandomPosterize',
    'RandomSolarize',
    'RandomAdjustSharpness',
    'RandomAutocontrast',
    'RandomEqualize',
    'InterpolationMode'
]

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

_modes_torchToCV2_mapping = {
    InterpolationMode.NEAREST: cv2.INTER_NEAREST,
    InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
    InterpolationMode.BICUBIC: cv2.INTER_CUBIC,
    InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4
}

_torch_modesToStr_mapping = {
    v: k
    for k, v in _torch_strToModes_mapping.items()
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
            format_string += '    {}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(
            self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None
    ):
        self.size = size
        self.max_size = max_size
        self.antialias = antialias
        self.cv_F = None
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int."
                "Please, use InterpolationMode enum."
            )
            self.cv_F = cv_t.Resize(self.size, interpolation)
            interpolation = _torch_intToModes_mapping[interpolation]

        if isinstance(interpolation, str):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of str."
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_strToModes_mapping[interpolation]
        self.interpolation = interpolation

        self.tv_F = tv_t.Resize(self.size, self.interpolation, self.max_size, self.antialias)

        if self.cv_F is None:
            if self.interpolation in _modes_torchToCV2_mapping:
                self.cv_F = cv_t.Resize(self.size, _modes_torchToCV2_mapping[self.interpolation])
            else:
                self.cv_F = cv_t.Resize(self.size, cv2.INTER_LINEAR)

    def __call__(self, img):
        if type(img) == np.ndarray:
            if self.max_size or self.antialias:
                warnings.warn(
                    "Parameters \'max_size\' and \'antialias\' will be "
                    "ignored for np.ndarray image."
                )
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        interpolation_str = self.__class__.__name__
        interpolation_str += '(size={}, interpolation={}, max_size={}, antialias={}'\
            .format(self.size, self.interpolation, self.max_size, self.antialias)

        return interpolation_str


class Scale(Resize):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of the transforms.Scale transform is deprecated, "
            "please use transforms.Resize instead."
        )
        super(Scale, self).__init__(*args, **kwargs)


class ToTensor(object):
    def __init__(self):
        self.tv_F = tv_t.ToTensor()
        self.cv_F = cv_t.ToTensor()

    def __call__(self, pic):
        if type(pic) == torch.Tensor:
            return pic
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

        self.tv_F = tv_t.RandomCrop(self.size, self.padding, self.pad_if_needed,
                                    self.fill, self.padding_mode)
        self.cv_F = cv_t.RandomCrop(self.size, self.padding, self.pad_if_needed,
                                    self.fill, self.padding_mode)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={}, padding={})'.format(
            self.size, self.padding)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.tv_F = tv_t.ColorJitter(self.brightness, self.contrast,
                                     self.saturation, self.hue)
        self.cv_F = cv_t.ColorJitter(self.brightness, self.contrast,
                                     self.saturation, self.hue)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={}'.format(self.brightness)
        format_string += ', contrast={}'.format(self.contrast)
        format_string += ', saturation={}'.format(self.saturation)
        format_string += ', hue={})'.format(self.hue)
        return format_string


class Normalize(tv_t.Normalize):
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__(mean, std, inplace)


class PILToTensor(tv_t.PILToTensor):
    def __init__(self):
        super().__init__()


class ConvertImageDtype(tv_t.ConvertImageDtype):
    def __init__(self, dtype: torch.dtype):
        super().__init__(dtype)


class ToPILImage(tv_t.ToPILImage):
    def __init__(self, mode=None):
        super().__init__(mode)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.tv_F = tv_t.CenterCrop(self.size)
        self.cv_F = cv_t.CenterCrop(self.size)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        invalidInputError(isinstance(padding, (numbers.Number, tuple, list)),
                          "padding is expected to be (numbers.Number, tuple, list)")
        invalidInputError(isinstance(fill, (numbers.Number, str, tuple)),
                          "fill is expected to be (numbers.Number, str, tuple)")
        invalidInputError(padding_mode in ['constant', 'edge', 'reflect', 'symmetric'],
                          "padding_mode is expected to be ['constant', 'edge',"
                          " 'reflect', 'symmetric']")
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            invalidInputError(False,
                              f"Padding must be an int or a 2, or 4 element tuple,"
                              f" not a {len(padding)} element tuple")

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

        self.tv_F = tv_t.Pad(self.padding, self.fill, self.padding_mode)
        self.cv_F = cv_t.Pad(self.padding, self.fill, self.padding_mode)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={}, fill={}, padding_mode={})'\
            .format(self.padding, self.fill, self.padding_mode)


class Lambda(object):
    def __init__(self, lambd):
        invalidInputError((lambd, types.LambdaType),
                          "lambd is expected to types.LambdaType")
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):
    def __init__(self, transforms):
        invalidInputError(isinstance(transforms, (list, tuple)),
                          "transforms is expected to be (list, tuple)")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        invalidInputError(False, "not implemented")

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):
    def __init__(self, transforms, p=None):
        super().__init__(transforms)
        if p is not None and not isinstance(p, collections.Sequence):
            invalidInputError(False,
                              "Argument p should be a sequence")
        self.p = p

    def __call__(self, *args):
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(*args)

    def __repr__(self):
        return super().__repr__() + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
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


class RandomResizedCrop(object):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=InterpolationMode.BILINEAR):

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.cv_F = None

        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int."
                "Please, use InterpolationMode enum."
            )
            # note: the RandomResizedCrop class only accepts a single number as size,
            # and uses self.size = (size, size) as the target size
            # this maybe a bug of opencv_transforms, but we have to change our code for now
            self.cv_F = cv_t.RandomResizedCrop(self.size[0], self.scale, self.ratio, interpolation)
            interpolation = _torch_intToModes_mapping[interpolation]

        if isinstance(interpolation, str):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of str."
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_strToModes_mapping[interpolation]
        self.interpolation = interpolation

        self.tv_F = tv_t.RandomResizedCrop(self.size, self.scale, self.ratio, self.interpolation)

        if self.cv_F is None:
            if self.interpolation in _modes_torchToCV2_mapping:
                self.cv_F = cv_t.RandomResizedCrop(self.size[0], self.scale, self.ratio,
                                                   _modes_torchToCV2_mapping[self.interpolation])
            else:
                self.cv_F = cv_t.RandomResizedCrop(self.size[0], self.scale, self.ratio,
                                                   cv2.INTER_LINEAR)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        interpolate_str = _torch_modesToStr_mapping[self.interpolation]
        format_string = self.__class__.__name__ + '(size={}'.format(self.size)
        format_string += ', scale={}'.format(
            tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={}'.format(
            tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={})'.format(interpolate_str)
        return format_string


class RandomSizedCrop(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of the transforms.RandomSizedCrop transform is deprecated, "
            "please use transforms.RandomResizedCrop instead."
        )
        super(RandomSizedCrop, self).__init__(*args, **kwargs)


class FiveCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            invalidInputError(len(size) == 2,
                              "Please provide only two dimensions (h, w) for size.")
            self.size = size

        self.tv_F = tv_t.FiveCrop(self.size)
        self.cv_F = cv_t.FiveCrop(self.size)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class TenCrop(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            invalidInputError(len(size) == 2,
                              "Please provide only two dimensions (h, w) for size.")
            self.size = size
        self.vertical_flip = vertical_flip

        self.tv_F = tv_t.TenCrop(self.size, self.vertical_flip)
        self.cv_F = cv_t.TenCrop(self.size, self.vertical_flip)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={}, vertical_flip={})'.format(
            self.size, self.vertical_flip)


class LinearTransformation(object):
    def __init__(self, transformation_matrix, mean_vector=None):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            invalidInputError(False,
                              "transformation_matrix should be square. Got [{} x {}]"
                              " rectangular matrix.".format(*transformation_matrix.size()))

        if mean_vector is not None:
            if mean_vector.size(0) != transformation_matrix.size(0):
                invalidInputError(False,
                                  "mean_vector should have the same length {}"
                                  "as any one of the dimensions of the transformation_matrix"
                                  " [{}]"
                                  .format(mean_vector.size(0), tuple(transformation_matrix.size()))
                                  )

            if transformation_matrix.device != mean_vector.device:
                invalidInputError(False,
                                  "Input tensors should be on the same device. Got {} and {}"
                                  .format(transformation_matrix.device, mean_vector.device)
                                  )

        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

        self.cv_F = cv_t.LinearTransformation(self.transformation_matrix)
        self.tv_F = tv_t.LinearTransformation(self.transformation_matrix, self.mean_vector)

    def __call__(self, tensor):
        if self.mean_vector is None:
            return self.cv_F.__call__(tensor)
        else:
            return self.tv_F.__call__(tensor)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(transformation_matrix={}, mean_vector={})'\
            .format(self.transformation_matrix.numpy().tolist(), self.mean_vector)
        return format_string


class RandomRotation(object):
    def __init__(
        self, degrees, interpolation=InterpolationMode.NEAREST,
            expand=False, center=None, fill=0, resample=None
    ):
        if resample is not None:
            warnings.warn(
                "The parameter 'resample' is deprecated"
                "Please use 'interpolation' instead."
            )
            self.resample = resample
            interpolation = _torch_intToModes_mapping(resample)
        else:
            self.resample = False

        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_intToModes_mapping(interpolation)

        self.degrees = degrees
        self.center = center
        self.interpolation = interpolation
        self.expand = expand
        self.fill = fill

        self.cv_F = cv_t.RandomRotation(
            degrees=self.degrees,
            resample=self.resample,
            expand=self.expand,
            center=self.center
        )

        self.tv_F = tv_t.RandomRotation(
            degrees=self.degrees,
            interpolation=self.interpolation,
            expand=self.expand,
            center=self.center,
            fill=self.fill
        )

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string


class RandomAffine(object):
    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 interpolation=cv2.INTER_LINEAR,
                 fill=0,
                 fillcolor=0,
                 center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                invalidInputError(False,
                                  "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            invalidInputError(isinstance(degrees, (tuple, list)) and len(degrees) == 2,
                              "degrees should be a list or tuple and it must be of length 2.")
            self.degrees = degrees

        if translate is not None:
            invalidInputError(isinstance(translate, (tuple, list)) and len(translate) == 2,
                              "translate should be a list or tuple and it must be of length 2.")
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    invalidInputError(False,
                                      "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            invalidInputError(isinstance(scale, (tuple, list)) and len(scale) == 2,
                              "scale should be a list or tuple and it must be of length 2.")
            for s in scale:
                if s <= 0:
                    invalidInputError(False, "scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    invalidInputError(False,
                                      "If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                invalidInputError(isinstance(shear, (tuple, list)) and len(shear) == 2,
                                  "shear should be a list or tuple and it must be of length 2.")
                self.shear = shear
        else:
            self.shear = shear

        if fillcolor is not None:
            warnings.warn(
                "The parameter 'fillcolor' is deprecated. "
                "Please use 'fill' instead."
            )
            fill = fillcolor

        if fill is None:
            fill = 0
        elif not isinstance(fill, (collections.Sequence, numbers.Number)):
            invalidInputError(False,
                              "Fill should be either a sequence or a number.")
        self.fill = fill

        if center is not None:
            invalidInputError(isinstance(center, (tuple, list)) and len(center) == 2,
                              "center should be a list or tuple and it must be of length 2.")

        # self.resample = resample
        self.center = center

        self.cv_F = None
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int."
                "Please, use InterpolationMode enum."
            )
            self.cv_F = cv_t.RandomAffine(
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                interpolation=interpolation,
                fillcolor=self.fill
            )
            interpolation = _torch_intToModes_mapping[interpolation]

        if isinstance(interpolation, str):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of str."
                "Please, use InterpolationMode enum."
            )
            interpolation = _torch_strToModes_mapping[interpolation]
        self.interpolation = interpolation

        self.tv_F = tv_t.RandomAffine(degrees=self.degrees,
                                      translate=self.translate,
                                      scale=self.scale,
                                      shear=self.shear,
                                      interpolation=self.interpolation,
                                      fill=self.fill,
                                      center=self.center)
        if self.cv_F is None:
            if self.interpolation in _modes_torchToCV2_mapping:
                self.cv_F = cv_t.RandomAffine(degrees=self.degrees,
                                              translate=self.translate,
                                              scale=self.scale,
                                              shear=self.shear,
                                              interpolation=_modes_torchToCV2_mapping[
                                                  self.interpolation],
                                              fillcolor=self.fill)
            else:
                self.cv_F = cv_t.RandomAffine(degrees=self.degrees,
                                              translate=self.translate,
                                              scale=self.scale,
                                              shear=self.shear,
                                              interpolation=cv2.INTER_LINEAR,
                                              fillcolor=self.fill)

    def __call__(self, img):
        if type(img) == np.ndarray:
            if self.center is None:
                warnings.warn(
                    "Parameters \'center\' will be ignored for np.ndarray image."
                )
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        if self.interpolation != InterpolationMode.NEAREST:
            s += f", interpolation={self.interpolation.value}"
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += ")"

        return s


class Grayscale(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

        self.tv_F = tv_t.Grayscale(self.num_output_channels)
        self.cv_F = cv_t.Grayscale(self.num_output_channels)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={})'.format(
            self.num_output_channels)


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

        self.tv_F = tv_t.Grayscale(self.size, self.vertical_flip)
        self.cv_F = cv_t.Grayscale(self.size, self.vertical_flip)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomPerspective(tv_t.RandomPerspective):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomPerspective, self).__call__(img)


class RandomErasing(tv_t.RandomErasing):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomErasing, self).__call__(img)


class GaussianBlur(tv_t.GaussianBlur):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(GaussianBlur, self).__call__(img)


class RandomInvert(tv_t.RandomInvert):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomInvert, self).__call__(img)


class RandomPosterize(tv_t.RandomPosterize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomPosterize, self).__call__(img)


class RandomSolarize(tv_t.RandomSolarize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomSolarize, self).__call__(img)


class RandomAdjustSharpness(tv_t.RandomAdjustSharpness):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomAdjustSharpness, self).__call__(img)


class RandomAutocontrast(tv_t.RandomAutocontrast):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomAutocontrast, self).__call__(img)


class RandomEqualize(tv_t.RandomEqualize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            img = tv_t.ToTensor()(img)

        return super(RandomEqualize, self).__call__(img)
