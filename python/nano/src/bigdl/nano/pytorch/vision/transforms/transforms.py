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
import types
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
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    # TODO: Support cv2.INTER_AREA
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        self.size = size
        self.max_size = max_size
        self.antialias = antialias
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
        self.interpolation = interpolation

        self.tv_F = tv_t.Resize(self.size, self.interpolation, self.max_size, self.antialias)

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
        interpolation_str += '(size={0}, interpolation={1}, max_size={2}, antialias={3}'\
            .format(self.size, self.interpolation, self.max_size, self.antialias)

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
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
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
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
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
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        assert isinstance(padding, (numbers.Number, tuple, list))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError(
                "Padding must be an int or a 2, or 4 element tuple, not a " +
                "{} element tuple".format(len(padding))
            )

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
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
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
            format_string += '    {0}'.format(t)
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
            raise TypeError("Argument p should be a sequence")
        self.p = p

    def __call__(self, *args):
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(*args)

    def __repr__(self):
        return super().__repr__() + '(p={0})'.format(self.p)


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
    # TODO: Support cv2.INTER_AREA
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=InterpolationMode.BILINEAR):

        self.size = size
        self.scale = scale
        self.ratio = ratio

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
        self.interpolation = interpolation

        self.tv_F = tv_t.RandomResizedCrop(self.size, self.scale, self.ratio, self.interpolation)

        if self.interpolation in _modes_torchToCV2_mapping:
            self.cv_F = cv_t.RandomResizedCrop(size, self.scale, self.ratio,
                                               _modes_torchToCV2_mapping[self.interpolation])
        else:
            self.cv_F = cv_t.RandomResizedCrop(size, self.scale, self.ratio, cv2.INTER_LINEAR)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        interpolate_str = _torch_modesToStr_mapping[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class FiveCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(
                size
            ) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

        self.tv_F = tv_t.FiveCrop(self.size)
        self.cv_F = cv_t.FiveCrop(self.size)

    def __call__(self, img):
        if type(img) == np.ndarray:
            return self.cv_F.__call__(img)
        else:
            return self.tv_F.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class TenCrop(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(
                size
            ) == 2, "Please provide only two dimensions (h, w) for size."
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
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(
            self.size, self.vertical_flip)


class LinearTransformation(object):
    def __init__(self, transformation_matrix, mean_vector=None):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(
                                 *transformation_matrix.size()))

        if mean_vector is not None:
            if mean_vector.size(0) != transformation_matrix.size(0):
                raise ValueError(
                    "mean_vector should have the same length {} " +
                    "as any one of the dimensions of the transformation_matrix [{}]"
                    .format(mean_vector.size(0), tuple(transformation_matrix.size()))
                )

            if transformation_matrix.device != mean_vector.device:
                raise ValueError(
                    "Input tensors should be on the same device. Got {0} and {1}"
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
        format_string = self.__class__.__name__ + '('
        format_string += (
                "transformation_matrix=" +
                str(self.transformation_matrix.numpy().tolist()) + ', '
                + "mean_vector=" + str(self.mean_vector) + ')')
        return format_string


class RandomRotation(object):
    # TODO: missing parameter "interpolation"
    # Conflict: resample indexes of torchvision and opencv are different.
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center


class RandomAffine(object):
    # TODO: Support cv2.INTER_AREA
    def __init__(self,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 interpolation=cv2.INTER_LINEAR,
                 fill=0,
                 center=None):
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
        self.interpolation = interpolation

        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError(
                        "If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        # if fillcolor is not None:
        #     warnings.warn(
        #         "The parameter 'fillcolor' is deprecated since 0.12 and will be removed in 0.14. "
        #         "Please use 'fill' instead."
        #     )
        #     fill = fillcolor

        if fill is None:
            fill = 0
        elif not isinstance(fill, (collections.Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")
        self.fill = fill

        if center is not None:
            assert isinstance(center, (tuple, list)) and len(center) == 2, \
                "center should be a list or tuple and it must be of length 2."

        # self.resample = resample
        self.center = center

        self.tv_F = tv_t.RandomAffine(degrees=self.degrees,
                                      translate=self.translate,
                                      scale=self.scale,
                                      shear=self.shear,
                                      interpolation=self.interpolation,
                                      fill=self.fill,
                                      center=self.center)

        if self.interpolation in _modes_torchToCV2_mapping:
            self.cv_F = cv_t.RandomAffine(degrees=self.degrees,
                                          translate=self.translate,
                                          scale=self.scale,
                                          shear=self.shear,
                                          interpolation=_modes_torchToCV2_mapping[
                                              self.interpolation],
                                          fillcolor=self.fill,
                                          center=self.center)
        else:
            self.cv_F = cv_t.RandomAffine(degrees=self.degrees,
                                          translate=self.translate,
                                          scale=self.scale,
                                          shear=self.shear,
                                          interpolation=cv2.INTER_LINEAR,
                                          fillcolor=self.fill,
                                          center=self.center)

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
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
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
        return self.__class__.__name__ + '(num_output_channels={0})'.format(
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
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class RandomPerspective(tv_t.RandomPerspective):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomPerspective, self).__call__(img)


class RandomErasing(tv_t.RandomErasing):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomErasing, self).__call__(img)


class GaussianBlur(tv_t.GaussianBlur):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(GaussianBlur, self).__call__(img)


class RandomInvert(tv_t.RandomInvert):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomInvert, self).__call__(img)


class RandomPosterize(tv_t.RandomPosterize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomPosterize, self).__call__(img)


class RandomSolarize(tv_t.RandomSolarize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomSolarize, self).__call__(img)


class RandomAdjustSharpness(tv_t.RandomAdjustSharpness):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomAdjustSharpness, self).__call__(img)


class RandomAutocontrast(tv_t.RandomAutocontrast):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomAutocontrast, self).__call__(img)


class RandomEqualize(tv_t.RandomEqualize):

    def __call__(self, img):
        if type(img) == np.ndarray:
            raise NotImplementedError("Input image must be PIL image or Tensor image for {0}".
                                      format(self.__class__.__name__))
        else:
            return super(RandomEqualize, self).__call__(img)
