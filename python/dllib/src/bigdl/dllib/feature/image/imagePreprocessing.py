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

from bigdl.dllib.utils.common import *
from bigdl.dllib.feature.common import Preprocessing

if sys.version >= '3':
    long = int
    unicode = str


class ImagePreprocessing(Preprocessing):
    """
    ImagePreprocessing is a transformer that transform ImageFeature
    """
    def __init__(self, bigdl_type="float", *args):
        super(ImagePreprocessing, self).__init__(bigdl_type, *args)


class ImageBytesToMat(ImagePreprocessing):
    """
    Transform byte array(original image file in byte) to OpenCVMat
    :param byte_key key that maps byte array
    :param image_codec specifying the color type of a loaded image, same as in OpenCV.imread.
     By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
    """
    def __init__(self, byte_key="bytes", image_codec=-1, bigdl_type="float"):
        super(ImageBytesToMat, self).__init__(bigdl_type, byte_key, image_codec)


class ImagePixelBytesToMat(ImagePreprocessing):
    """
    Transform byte array(pixels in byte) to OpenCVMat
    :param byte_key key that maps byte array
    """
    def __init__(self, byte_key="bytes", bigdl_type="float"):
        super(ImagePixelBytesToMat, self).__init__(bigdl_type, byte_key)


class ImageResize(ImagePreprocessing):
    """
    Resize image
    :param resize_h height after resize
    :param resize_w width after resize
    :param resize_mode if resizeMode = -1, random select a mode from (Imgproc.INTER_LINEAR,
     Imgproc.INTER_CUBIC, Imgproc.INTER_AREA, Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
    :param use_scale_factor if true, scale factor fx and fy is used, fx = fy = 0
    note that the result of the following are different
    Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH), 0, 0, Imgproc.INTER_LINEAR)
    Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH))
    """
    def __init__(self, resize_h, resize_w, resize_mode=1, use_scale_factor=True,
                 bigdl_type="float"):
        super(ImageResize, self).__init__(bigdl_type, resize_h, resize_w,
                                          resize_mode, use_scale_factor)


class ImageBrightness(ImagePreprocessing):
    """
    adjust the image brightness
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(ImageBrightness, self).__init__(bigdl_type, float(delta_low), float(delta_high))


class ImageChannelNormalize(ImagePreprocessing):
    """
    image channel normalize
    :param mean_b mean value in B channel
    :param mean_g mean value in G channel
    :param mean_r mean value in R channel
    :param std_b std value in B channel
    :param std_g std value in G channel
    :param std_r std value in R channel
    """
    def __init__(self, mean_b, mean_g, mean_r, std_b=1.0,
                 std_g=1.0, std_r=1.0, bigdl_type="float"):
        super(ImageChannelNormalize, self).__init__(bigdl_type, float(mean_b), float(mean_g),
                                                    float(mean_r), float(std_b), float(std_g),
                                                    float(std_r))


class PerImageNormalize(ImagePreprocessing):
    """
    Normalizes the norm or value range per image, similar to opencv::normalize
    https://docs.opencv.org/ref/master/d2/de8/group__core__array.html
    #ga87eef7ee3970f86906d69a92cbf064bd
    ImageNormalize normalizes scale and shift the input features. Various normalize
    methods are supported,
    Eg. NORM_INF, NORM_L1, NORM_L2 or NORM_MINMAX
    Pleas notice it's a per image normalization.
    :param min lower range boundary in case of the range normalization or
    norm value to normalize
    :param max upper range boundary in case of the range normalization.
    It is not used for the norm normalization.
    :param norm_type normalization type, see opencv:NormTypes.
    https://docs.opencv.org/ref/master/d2/de8/group__core__array.html
    #gad12cefbcb5291cf958a85b4b67b6149f
    Default Core.NORM_MINMAX
    """
    def __init__(self, min, max, norm_type=32, bigdl_type="float"):
        super(PerImageNormalize, self).__init__(bigdl_type, float(min), float(max), norm_type)


class ImageMatToTensor(ImagePreprocessing):
    """
    MatToTensor
    :param toRGB BGR to RGB (default is BGR)
    :param tensorKey key to store transformed tensor
    :param format DataFormat.NCHW or DataFormat.NHWC
    """
    def __init__(self, to_RGB=False, tensor_key="imageTensor",
                 share_buffer=True, format="NCHW", bigdl_type="float"):
        super(ImageMatToTensor, self).__init__(bigdl_type, to_RGB, tensor_key,
                                               share_buffer, format)


class ImageSetToSample(ImagePreprocessing):
    """
    transform imageframe to samples
    :param input_keys keys that maps inputs (each input should be a tensor)
    :param target_keys keys that maps targets (each target should be a tensor)
    :param sample_key key to store sample
    """
    def __init__(self, input_keys=["imageTensor"], target_keys=["label"],
                 sample_key="sample", bigdl_type="float"):
        super(ImageSetToSample, self).__init__(bigdl_type, input_keys, target_keys, sample_key)


class ImageHue(ImagePreprocessing):
    """
    adjust the image hue
    :param deltaLow hue parameter: low bound
    :param deltaHigh hue parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(ImageHue, self).__init__(bigdl_type, float(delta_low), float(delta_high))


class ImageSaturation(ImagePreprocessing):
    """
    adjust the image Saturation
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(ImageSaturation, self).__init__(bigdl_type, float(delta_low), float(delta_high))


class ImageChannelOrder(ImagePreprocessing):
    """
    random change the channel of an image
    """
    def __init__(self, bigdl_type="float"):
        super(ImageChannelOrder, self).__init__(bigdl_type)


class ImageColorJitter(ImagePreprocessing):
    """
    Random adjust brightness, contrast, hue, saturation
    :param brightness_prob probability to adjust brightness
    :param brightness_delta brightness parameter
    :param contrast_prob probability to adjust contrast
    :param contrast_lower contrast lower parameter
    :param contrast_upper contrast upper parameter
    :param hue_prob probability to adjust hue
    :param hue_delta hue parameter
    :param saturation_prob probability to adjust saturation
    :param saturation_lower saturation lower parameter
    :param saturation_upper saturation upper parameter
    :param random_order_prob random order for different operation
    :param shuffle  shuffle the transformers
    """
    def __init__(self, brightness_prob=0.5,
                 brightness_delta=32.0,
                 contrast_prob=0.5,
                 contrast_lower=0.5,
                 contrast_upper=1.5,
                 hue_prob=0.5,
                 hue_delta=18.0,
                 saturation_prob=0.5,
                 saturation_lower=0.5,
                 saturation_upper=1.5,
                 random_order_prob=0.0,
                 shuffle=False,
                 bigdl_type="float"):
        super(ImageColorJitter, self).__init__(bigdl_type,
                                               float(brightness_prob), float(brightness_delta),
                                               float(contrast_prob), float(contrast_lower),
                                               float(contrast_upper), float(hue_prob),
                                               float(hue_delta), float(saturation_prob),
                                               float(saturation_lower), float(saturation_upper),
                                               float(random_order_prob), shuffle)


class ImageAspectScale(ImagePreprocessing):
    """
    Resize the image, keep the aspect ratio. scale according to the short edge
    :param min_size scale size, apply to short edge
    :param scale_multiple_of make the scaled size multiple of some value
    :param max_size max size after scale
    :param resize_mode if resizeMode = -1, random select a mode from
    (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
    Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
    :param use_scale_factor if true, scale factor fx and fy is used, fx = fy = 0
    :aram min_scale control the minimum scale up for image
    """

    def __init__(self, min_size, scale_multiple_of=1, max_size=1000,
                 resize_mode=1, use_scale_factor=True, min_scale=-1.0,
                 bigdl_type="float"):
        super(ImageAspectScale, self).__init__(bigdl_type,
                                               min_size, scale_multiple_of, max_size,
                                               resize_mode, use_scale_factor, min_scale)


class ImageRandomAspectScale(ImagePreprocessing):
    """
    resize the image by randomly choosing a scale
    :param scales array of scale options that for random choice
    :param scaleMultipleOf Resize test images so that its width and height are multiples of
    :param maxSize Max pixel size of the longest side of a scaled input image
    """
    def __init__(self, scales, scale_multiple_of=1, max_size=1000, bigdl_type="float"):
        super(ImageRandomAspectScale, self).__init__(bigdl_type,
                                                     scales, scale_multiple_of, max_size)


class ImagePixelNormalize(ImagePreprocessing):
    """
    Pixel level normalizer, data(i) = data(i) - mean(i)

    :param means pixel level mean, following H * W * C order
    """

    def __init__(self, means, bigdl_type="float"):
        super(ImagePixelNormalize, self).__init__(bigdl_type, means)


class ImageRandomCrop(ImagePreprocessing):
    """
    Random crop a `cropWidth` x `cropHeight` patch from an image.
    The patch size should be less than the image size.

    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip whether to clip the roi to image boundaries
    """

    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        super(ImageRandomCrop, self).__init__(bigdl_type,
                                              crop_width, crop_height, is_clip)


class ImageCenterCrop(ImagePreprocessing):
    """
    Crop a `cropWidth` x `cropHeight` patch from center of image.
    The patch size should be less than the image size.
    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip  clip cropping box boundary
    """

    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        super(ImageCenterCrop, self).__init__(bigdl_type,
                                              crop_width, crop_height, is_clip)


class ImageFixedCrop(ImagePreprocessing):
    """
    Crop a fixed area of image

    :param x1 start in width
    :param y1 start in height
    :param x2 end in width
    :param y2 end in height
    :param normalized whether args are normalized, i.e. in range [0, 1]
    :param is_clip whether to clip the roi to image boundaries
    """

    def __init__(self, x1, y1, x2, y2, normalized=True, is_clip=True, bigdl_type="float"):
        super(ImageFixedCrop, self).__init__(bigdl_type,
                                             x1, y1, x2, y2, normalized, is_clip)


class ImageExpand(ImagePreprocessing):
    """
    expand image, fill the blank part with the meanR, meanG, meanB

    :param means_r means in R channel
    :param means_g means in G channel
    :param means_b means in B channel
    :param min_expand_ratio min expand ratio
    :param max_expand_ratio max expand ratio
    """

    def __init__(self, means_r=123, means_g=117, means_b=104,
                 min_expand_ratio=1.0,
                 max_expand_ratio=4.0, bigdl_type="float"):
        super(ImageExpand, self).__init__(bigdl_type, means_r, means_g, means_b,
                                          min_expand_ratio, max_expand_ratio)


class ImageFiller(ImagePreprocessing):
    """
    Fill part of image with certain pixel value
    :param start_x start x ratio
    :param start_y start y ratio
    :param end_x end x ratio
    :param end_y end y ratio
    :param value filling value
    """

    def __init__(self, start_x, start_y, end_x, end_y, value=255, bigdl_type="float"):
        super(ImageFiller, self).__init__(bigdl_type, start_x, start_y,
                                          end_x, end_y, value)


class ImageHFlip(ImagePreprocessing):
    """
    Flip the image horizontally
    """

    def __init__(self, bigdl_type="float"):
        super(ImageHFlip, self).__init__(bigdl_type)


class ImageMirror(ImagePreprocessing):
    """
    Flip the image horizontally and vertically
    """
    def __init__(self, bigdl_type="float"):
        super(ImageMirror, self).__init__(bigdl_type)


class ImageFeatureToTensor(Preprocessing):
    """
    a Transformer that convert ImageFeature to a Tensor.
    """
    def __init__(self, bigdl_type="float"):
        super(ImageFeatureToTensor, self).__init__(bigdl_type)


class ImageFeatureToSample(Preprocessing):
    """
    A transformer that get Sample from ImageFeature.
    """
    def __init__(self, bigdl_type="float"):
        super(ImageFeatureToSample, self).__init__(bigdl_type)


class RowToImageFeature(Preprocessing):
    """
    a Transformer that converts a Spark Row to a BigDL ImageFeature.
    """
    def __init__(self, bigdl_type="float"):
        super(RowToImageFeature, self).__init__(bigdl_type)


class ImageRandomPreprocessing(Preprocessing):
    """
    Randomly apply the preprocessing to some of the input ImageFeatures, with probability specified.
    E.g. if prob = 0.5, the preprocessing will apply to half of the input ImageFeatures.
    :param preprocessing preprocessing to apply.
    :param prob probability to apply the preprocessing action.
    """

    def __init__(self, preprocessing, prob, bigdl_type="float"):
        super(ImageRandomPreprocessing, self).__init__(bigdl_type, preprocessing, float(prob))
