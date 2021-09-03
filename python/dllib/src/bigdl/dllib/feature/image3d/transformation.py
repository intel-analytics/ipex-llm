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

import sys
import numpy as np

from bigdl.util.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.image.imageset import *

if sys.version >= '3':
    long = int
    unicode = str


class ImagePreprocessing3D(ImagePreprocessing):
    """
    ImagePreprocessing3D is a transformer that transform ImageFeature for 3D image
    """
    def __init__(self, bigdl_type="float", *args):
        super(ImagePreprocessing3D, self).__init__(bigdl_type, *args)


class Crop3D(ImagePreprocessing3D):
    """
    Crop a patch from a 3D image from 'start' of patch size.
    The patch size should be less than the image size.

    :param start start point list[depth, height, width] for cropping
    :param patchSize patch size list[depth, height, width]
    """
    def __init__(self, start, patch_size, bigdl_type="float"):
        super(Crop3D, self).__init__(bigdl_type, start, patch_size)


class RandomCrop3D(ImagePreprocessing3D):
    """
    Random crop a `cropDepth` x `cropHeight` x `cropWidth` patch from an image.
    The patch size should be less than the image size.

    :param crop_depth depth after crop
    :param crop_height height after crop
    :param crop_width width after crop
    """
    def __init__(self, crop_depth, crop_height, crop_width, bigdl_type="float"):
        super(RandomCrop3D, self).__init__(bigdl_type, crop_depth, crop_height, crop_width)


class CenterCrop3D(ImagePreprocessing3D):
    """
    Center crop a `cropDepth` x `cropHeight` x `cropWidth` patch from an image.
    The patch size should be less than the image size.

    :param crop_depth depth after crop
    :param crop_height height after crop
    :param crop_width width after crop
    """
    def __init__(self, crop_depth, crop_height, crop_width, bigdl_type="float"):
        super(CenterCrop3D, self).__init__(bigdl_type, crop_depth, crop_height, crop_width)


class Rotate3D(ImagePreprocessing3D):
    """
    Rotate a 3D image with specified angles.

    :param rotation_angles the angles for rotation.
    Which are the yaw(a counterclockwise rotation angle about the z-axis),
    pitch(a counterclockwise rotation angle about the y-axis),
    and roll(a counterclockwise rotation angle about the x-axis).
    """
    def __init__(self, rotation_angles, bigdl_type="float"):
        super(Rotate3D, self).__init__(bigdl_type, rotation_angles)


class AffineTransform3D(ImagePreprocessing3D):
    """
    Affine transformer implements affine transformation on a given tensor.
    To avoid defects in resampling, the mapping is from destination to source.
    dst(z,y,x) = src(f(z),f(y),f(x)) where f: dst -> src
    :param affine_mat: numpy array in 3x3 shape.Define affine transformation from dst to src.
    :param translation: numpy array in 3 dimension.Default value is np.zero(3).
            Define translation in each axis.
    :param clampMode: str, default value is "clamp".
            Define how to handle interpolation off the input image.
    :param padVal: float, default is 0.0. Define padding value when clampMode="padding".
            Setting this value when clampMode="clamp" will cause an error.
    """

    def __init__(self, affine_mat, translation=np.zeros(3), clamp_mode="clamp",
                 pad_val=0.0, bigdl_type="float"):
        affine_mat_tensor = JTensor.from_ndarray(affine_mat)
        translation_tensor = JTensor.from_ndarray(translation)
        super(AffineTransform3D, self).__init__(bigdl_type, affine_mat_tensor, translation_tensor,
                                                clamp_mode, pad_val)
