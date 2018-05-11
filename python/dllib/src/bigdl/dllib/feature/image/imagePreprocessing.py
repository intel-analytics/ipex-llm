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

from bigdl.util.common import *
from zoo.feature.common import Preprocessing
from zoo.feature.image.imageset import ImageSet

if sys.version >= '3':
    long = int
    unicode = str


class ImagePreprocessing(Preprocessing):
    """
    ImagePreprocessing is a transformer that transform ImageFeature
    """
    def __init__(self, bigdl_type="float"):
        super(ImagePreprocessing, self).__init__(bigdl_type)

    def __call__(self, image_set, bigdl_type="float"):
        """
        transform ImageSet
        """
        jset = callBigDlFunc(bigdl_type,
                             "transformImageSet", self.value, image_set)
        return ImageSet(jvalue=jset)


class Resize(ImagePreprocessing):
    """
     image resize
    """
    def __init__(self, resizeH, resizeW, bigdl_type="float"):
        super(Resize, self).__init__(bigdl_type, resizeH, resizeW)


class Brightness(ImagePreprocessing):
    """
    adjust the image brightness
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgBrightness", delta_low, delta_high)


class ChannelNormalize(ImagePreprocessing):
    """
    image channel normalize
    :param mean_r mean value in R channel
    :param mean_g mean value in G channel
    :param meanB_b mean value in B channel
    :param std_r std value in R channel
    :param std_g std value in G channel
    :param std_b std value in B channel
    """
    def __init__(self, mean_r, mean_b, mean_g, std_r=1.0, std_g=1.0, std_b=1.0, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgChannelNormalizer", mean_r, mean_g, mean_b, std_r, std_g, std_b)


class MatToTensor(ImagePreprocessing):
    """
    MatToTensor
    """
    def __init__(self, bigdl_type="float"):
        super(MatToTensor, self).__init__(bigdl_type)


class CenterCrop(ImagePreprocessing):
    """
    CenterCrop
    """
    def __init__(self, cropWidth, cropHeight, bigdl_type="float"):
        super(CenterCrop, self).__init__(bigdl_type, cropWidth, cropHeight)


class Hue(ImagePreprocessing):
    """
    adjust the image hue
    :param deltaLow hue parameter: low bound
    :param deltaHigh hue parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgHue", delta_low, delta_high)


class Saturation(ImagePreprocessing):
    """
    adjust the image Saturation
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgSaturation", delta_low, delta_high)
