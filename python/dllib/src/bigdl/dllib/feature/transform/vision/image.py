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

import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *

if sys.version >= '3':
    long = int
    unicode = str


class FeatureTransformer(JavaValue):
    """
    FeatureTransformer is a transformer that transform ImageFeature
    """

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def transform(self, image_feature, bigdl_type="float"):
        """
        transform ImageFeature
        """
        callBigDlFunc(bigdl_type, "transformImageFeature", self.value, image_feature)
        return image_feature

    def __call__(self, image_frame, bigdl_type="float"):
        """
        transform ImageFrame
        """
        jframe = callBigDlFunc(bigdl_type,
                             "transformImageFrame", self.value, image_frame)
        return ImageFrame(jvalue=jframe)

class Pipeline(FeatureTransformer):
    """
    Pipeline of FeatureTransformer
    """

    def __init__(self, transformers, bigdl_type="float"):
        for transfomer in transformers:
            assert transfomer.__class__.__bases__[0].__name__ == "FeatureTransformer", "the transformer should be " \
                                                                                       "subclass of FeatureTransformer"
        super(Pipeline, self).__init__(bigdl_type, transformers)

class ImageFeature(JavaValue):
    """
    Each ImageFeature keeps information about single image,
    it can include various status of an image,
    e.g. original bytes read from image file, an opencv mat,
    pixels in float array, image label, meta data and so on.
    it uses HashMap to store all these data,
    the key is string that identify the corresponding value
    """

    def __init__(self, image=None, label=None, path=None, bigdl_type="float"):
        image_tensor = JTensor.from_ndarray(image) if image is not None else None
        label_tensor = JTensor.from_ndarray(label) if label is not None else None
        self.bigdl_type = bigdl_type
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), image_tensor, label_tensor, path)

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image as ndarray from ImageFeature
        """
        tensor = callBigDlFunc(self.bigdl_type, "imageFeatureToImageTensor", self.value,
                               float_key, to_chw)
        return tensor.to_ndarray()

    def get_label(self):
        """
        get label as ndarray from ImageFeature
        """
        label = callBigDlFunc(self.bigdl_type, "imageFeatureToLabelTensor", self.value)
        return label.to_ndarray()

    def keys(self):
        """
        get key set from ImageFeature
        """
        return callBigDlFunc(self.bigdl_type, "imageFeatureGetKeys", self.value)

class ImageFrame(JavaValue):
    """
    ImageFrame wraps a set of ImageFeature
    """

    def __init__(self, jvalue, bigdl_type="float"):
        self.value = jvalue
        self.bigdl_type = bigdl_type
        if self.is_local():
            self.image_frame = LocalImageFrame(jvalue=self.value)
        else:
            self.image_frame = DistributedImageFrame(jvalue=self.value)


    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read images as Image Frame
        if sc is defined, Read image as DistributedImageFrame from local file system or HDFS
        if sc is null, Read image as LocalImageFrame from local file system
        :param path path to read images
        if sc is defined, path can be local or HDFS. Wildcard character are supported.
        if sc is null, path is local directory/image file/image file with wildcard character
        :param sc SparkContext
        :param min_partitions A suggestion value of the minimal splitting number for input data.
        :return ImageFrame
        """
        return ImageFrame(jvalue=callBigDlFunc(bigdl_type, "read", path, sc, min_partitions))

    @classmethod
    def read_parquet(cls, path, sc, bigdl_type="float"):
        """
        Read parquet file as DistributedImageFrame
        """
        return DistributedImageFrame(jvalue=callBigDlFunc(bigdl_type, "readParquet", path, sc))

    @classmethod
    def write_parquet(cls, path, output, sc, partition_num = 1, bigdl_type="float"):
        """
        write ImageFrame as parquet file
        """
        return callBigDlFunc(bigdl_type, "writeParquet", path, output, sc, partition_num)

    def is_local(self):
        """
        whether this is a LocalImageFrame
        """
        return callBigDlFunc(self.bigdl_type, "isLocal", self.value)

    def is_distributed(self):
        """
        whether this is a DistributedImageFrame
        """
        return callBigDlFunc(self.bigdl_type, "isDistributed", self.value)

    def transform(self, transformer, bigdl_type="float"):
        """
        transformImageFrame
        """
        self.value = callBigDlFunc(bigdl_type,
                                 "transformImageFrame", transformer, self.value)
        return self

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image from ImageFrame
        """
        return self.image_frame.get_image(float_key, to_chw)

    def get_label(self):
        """
        get label from ImageFrame
        """
        return self.image_frame.get_label()

    def get_predict(self, key="predict"):
        """
        get prediction from ImageFrame
        """
        return self.image_frame.get_predict(key)

    def get_sample(self):
        """
        get sample from ImageFrame
        """
        return self.image_frame.get_sample()

    def get_uri(self):
        """
        get uri from imageframe
        """
        return self.image_frame.get_uri()

    def set_label(self, label, bigdl_type="float"):
        """
        set label for imageframe
        """
        return callBigDlFunc(bigdl_type,
                             "setLabel", label, self.value)

    def random_split(self, weights):
        """
        Random split imageframes according to weights
        :param weights: weights for each ImageFrame
        :return: 
        """
        jvalues =  self.image_frame.random_split(weights)
        return [ImageFrame(jvalue) for jvalue in jvalues]

class LocalImageFrame(ImageFrame):
    """
    LocalImageFrame wraps a list of ImageFeature
    """
    def __init__(self, image_list=None, label_list=None, jvalue=None, bigdl_type="float"):
        assert jvalue or image_list, "jvalue and image_list cannot be None in the same time"
        if jvalue:
            self.value = jvalue
        else:
            # init from image ndarray list and label rdd(optional)
            image_tensor_list = map(lambda image: JTensor.from_ndarray(image), image_list)
            label_tensor_list = map(lambda label: JTensor.from_ndarray(label), label_list) if label_list else None
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       image_tensor_list, label_tensor_list)

        self.bigdl_type = bigdl_type

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image list from ImageFrame
        """
        tensors = callBigDlFunc(self.bigdl_type,
                                   "localImageFrameToImageTensor", self.value, float_key, to_chw)
        return map(lambda tensor: tensor.to_ndarray(), tensors)

    def get_label(self):
        """
        get label list from ImageFrame
        """
        labels = callBigDlFunc(self.bigdl_type, "localImageFrameToLabelTensor", self.value)
        return map(lambda tensor: tensor.to_ndarray(), labels)

    def get_predict(self, key="predict"):
        """
        get prediction list from ImageFrame
        """
        predicts = callBigDlFunc(self.bigdl_type, "localImageFrameToPredict", self.value, key)
        return map(lambda predict: (predict[0], predict[1].to_ndarray()) if predict[1] else (predict[0], None), predicts)

    def get_sample(self,  key="sample"):
        return callBigDlFunc(self.bigdl_type, "localImageFrameToSample", self.value, key)

    def get_uri(self, key = "uri"):
        return callBigDlFunc(self.bigdl_type, "localImageFrameToUri", self.value, key)

    def random_split(self, weights):
        raise "random split not supported in LocalImageFrame"

class DistributedImageFrame(ImageFrame):
    """
    DistributedImageFrame wraps an RDD of ImageFeature
    """

    def __init__(self, image_rdd=None, label_rdd=None, jvalue=None, bigdl_type="float"):
        assert jvalue or image_rdd, "jvalue and image_rdd cannot be None in the same time"
        if jvalue:
            self.value = jvalue
        else:
            # init from image ndarray rdd and label rdd(optional)
            image_tensor_rdd = image_rdd.map(lambda image: JTensor.from_ndarray(image))
            label_tensor_rdd = label_rdd.map(lambda label: JTensor.from_ndarray(label)) if label_rdd else None
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       image_tensor_rdd, label_tensor_rdd)

        self.bigdl_type = bigdl_type

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image rdd from ImageFrame
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type,
                               "distributedImageFrameToImageTensorRdd", self.value, float_key, to_chw)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_label(self):
        """
        get label rdd from ImageFrame
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type, "distributedImageFrameToLabelTensorRdd", self.value)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_predict(self, key="predict"):
        """
        get prediction rdd from ImageFrame
        """
        predicts = callBigDlFunc(self.bigdl_type, "distributedImageFrameToPredict", self.value, key)
        return predicts.map(lambda predict: (predict[0], predict[1].to_ndarray()) if predict[1] else (predict[0], None))

    def get_sample(self,  key="sample"):
        return callBigDlFunc(self.bigdl_type, "distributedImageFrameToSample", self.value, key)

    def get_uri(self, key = "uri"):
        return callBigDlFunc(self.bigdl_type, "distributedImageFrameToUri", self.value, key)

    def random_split(self, weights):
        return callBigDlFunc(self.bigdl_type, "distributedImageFrameRandomSplit", self.value, weights)

class HFlip(FeatureTransformer):
    """
    Flip the image horizontally
    """

    def __init__(self, bigdl_type="float"):
            super(HFlip, self).__init__(bigdl_type)

class Resize(FeatureTransformer):
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

    def __init__(self, resize_h, resize_w, resize_mode = 1, use_scale_factor=True,
                 bigdl_type="float"):
        super(Resize, self).__init__(bigdl_type, resize_h, resize_w, resize_mode, use_scale_factor)

class Brightness(FeatureTransformer):
    """
    adjust the image brightness
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(Brightness, self).__init__(bigdl_type, delta_low, delta_high)

class ChannelOrder(FeatureTransformer):
    """
    random change the channel of an image
    """

    def __init__(self, bigdl_type="float"):
        super(ChannelOrder, self).__init__(bigdl_type)

class Contrast(FeatureTransformer):
    """
    Adjust the image contrast
    :param delta_low contrast parameter low bound
    :param delta_high contrast parameter high bound
    """

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(Contrast, self).__init__(bigdl_type, delta_low, delta_high)

class Saturation(FeatureTransformer):
    """
    Adjust image saturation
    """

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(Saturation, self).__init__(bigdl_type, delta_low, delta_high)

class Hue(FeatureTransformer):
    """
    Adjust image hue
    :param delta_low hue parameter: low bound
    :param delta_high hue parameter: high bound
    """

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        super(Hue, self).__init__(bigdl_type, delta_low, delta_high)

class ChannelNormalize(FeatureTransformer):
    """
    image channel normalize
    :param mean_r mean value in R channel
    :param mean_g mean value in G channel
    :param meanB_b mean value in B channel
    :param std_r std value in R channel
    :param std_g std value in G channel
    :param std_b std value in B channel
    """
    def __init__(self, mean_r, mean_g, mean_b, std_r=1.0, std_g=1.0, std_b=1.0, bigdl_type="float"):
        super(ChannelNormalize, self).__init__(bigdl_type, mean_r, mean_g, mean_b, std_r, std_g, std_b)
        
class PixelNormalize(FeatureTransformer):
    """
    Pixel level normalizer, data(i) = data(i) - mean(i)

    :param means pixel level mean, following H * W * C order
    """
    
    def __init__(self, means, bigdl_type="float"):
        super(PixelNormalize, self).__init__(bigdl_type, means)


class RandomCrop(FeatureTransformer):
    """
    Random crop a `cropWidth` x `cropHeight` patch from an image.
    The patch size should be less than the image size.
    
    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip whether to clip the roi to image boundaries
    """
    
    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        super(RandomCrop, self).__init__(bigdl_type, crop_width, crop_height, is_clip)

class CenterCrop(FeatureTransformer):
    """
    Crop a `cropWidth` x `cropHeight` patch from center of image.
    The patch size should be less than the image size.
    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip  clip cropping box boundary
    """

    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        super(CenterCrop, self).__init__(bigdl_type, crop_width, crop_height, is_clip)

class FixedCrop(FeatureTransformer):
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
        super(FixedCrop, self).__init__(bigdl_type, x1, y1, x2, y2, normalized, is_clip)

class DetectionCrop(FeatureTransformer):
    """
    Crop from object detections, each image should has a tensor detection,
    which is stored in ImageFeature
    :param roi_key key that map a tensor detection
    :param normalized whether is detection is normalized, i.e. in range [0, 1]
    """

    def __init__(self, roi_key, normalized=True, bigdl_type="float"):
        super(DetectionCrop, self).__init__(bigdl_type, roi_key, normalized)


class Expand(FeatureTransformer):
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
        super(Expand, self).__init__(bigdl_type, means_r, means_g, means_b,
                                     min_expand_ratio, max_expand_ratio)
        
class Filler(FeatureTransformer):
    """
    Fill part of image with certain pixel value
    :param start_x start x ratio
    :param start_y start y ratio
    :param end_x end x ratio
    :param end_y end y ratio
    :param value filling value
    """
    
    def __init__(self, start_x, start_y, end_x, end_y, value = 255, bigdl_type="float"):
        super(Filler, self).__init__(bigdl_type, start_x,
                                     start_y,
                                     end_x,
                                     end_y,
                                     value)

class RandomTransformer(FeatureTransformer):
    """
    It is a wrapper for transformers to control the transform probability
    :param transformer transformer to apply randomness
    :param prob max prob
    """

    def __init__(self, transformer, prob, bigdl_type="float"):
        super(RandomTransformer, self).__init__(bigdl_type, transformer, prob)


class ColorJitter(FeatureTransformer):
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
    def __init__(self, brightness_prob = 0.5,
                 brightness_delta = 32.0,
                 contrast_prob = 0.5,
                 contrast_lower = 0.5,
                 contrast_upper = 1.5,
                 hue_prob = 0.5,
                 hue_delta = 18.0,
                 saturation_prob = 0.5,
                 saturation_lower = 0.5,
                 saturation_upper = 1.5,
                 random_order_prob = 0.0,
                 shuffle = False,
                 bigdl_type="float"):
        super(ColorJitter, self).__init__(bigdl_type, brightness_prob,
                                          brightness_delta,
                                          contrast_prob,
                                          contrast_lower,
                                          contrast_upper,
                                          hue_prob,
                                          hue_delta,
                                          saturation_prob,
                                          saturation_lower,
                                          saturation_upper,
                                          random_order_prob,
                                          shuffle)

class RandomSampler(FeatureTransformer):
    """
    Random sample a bounding box given some constraints and crop the image
    This is used in SSD training augmentation
    """

    def __init__(self):
        super(RandomSampler, self).__init__(bigdl_type)

class RoiProject(FeatureTransformer):
    """
    Project gt boxes onto the coordinate system defined by image boundary
    :param need_meet_center_constraint whether need to meet center constraint, i.e., the center of gt box need be within image boundary
    """

    def __init__(self, need_meet_center_constraint, bigdl_type="float"):
        super(RoiProject, self).__init__(bigdl_type, need_meet_center_constraint)

class RoiHFlip(FeatureTransformer):
    """
    horizontally flip the roi
    :param normalized whether the roi is normalized, i.e. in range [0, 1]
    """

    def __init__(self, normalized=True, bigdl_type="float"):
        super(RoiHFlip, self).__init__(bigdl_type, normalized)
        
class RoiResize(FeatureTransformer):
    """
    resize the roi according to scale
    :param normalized whether the roi is normalized, i.e. in range [0, 1]
    """
    def __init__(self, normalized=True, bigdl_type="float"):
        super(RoiResize, self).__init__(bigdl_type, normalized)

class RoiNormalize(FeatureTransformer):
    """
    Normalize Roi to [0, 1]
    """

    def __init__(self, bigdl_type="float"):
        super(RoiNormalize, self).__init__(bigdl_type)

class MatToFloats(FeatureTransformer):
    """
    Transform OpenCVMat to float array, note that in this transformer, the mat is released
    :param valid_height valid height in case the mat is invalid
    :param valid_width valid width in case the mat is invalid
    :param valid_channel valid channel in case the mat is invalid
    :param out_key key to store float array
    :param share_buffer share buffer of output
    """

    def __init__(self, valid_height=300, valid_width=300, valid_channel=300,
                 out_key = "floats", share_buffer=True, bigdl_type="float"):
        super(MatToFloats, self).__init__(bigdl_type, valid_height, valid_width, valid_channel,
                                          out_key, share_buffer)

class MatToTensor(FeatureTransformer):
    """
    transform opencv mat to tensor
    :param to_rgb BGR to RGB (default is BGR)
    :param tensor_key key to store transformed tensor
    """

    def __init__(self, to_rgb=False, tensor_key="imageTensor", bigdl_type="float"):
        super(MatToTensor, self).__init__(bigdl_type, to_rgb, tensor_key)

class AspectScale(FeatureTransformer):
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

    def __init__(self, min_size, scale_multiple_of = 1, max_size = 1000,
                 resize_mode = 1, use_scale_factor=True, min_scale=-1.0,
                 bigdl_type="float"):
        super(AspectScale, self).__init__(bigdl_type, min_size, scale_multiple_of, max_size,
                                          resize_mode, use_scale_factor, min_scale)
        
class RandomAspectScale(FeatureTransformer):
    """
    resize the image by randomly choosing a scale
    :param scales array of scale options that for random choice
    :param scaleMultipleOf Resize test images so that its width and height are multiples of
    :param maxSize Max pixel size of the longest side of a scaled input image
    """
    def __init__(self, scales, scale_multiple_of = 1, max_size = 1000, bigdl_type="float"):
        super(RandomAspectScale, self).__init__(bigdl_type, scales, scale_multiple_of, max_size)

class BytesToMat(FeatureTransformer):
    """
    Transform byte array(original image file in byte) to OpenCVMat
    :param byte_key key that maps byte array
    """
    def __init__(self, byte_key = "bytes", bigdl_type="float"):
        super(BytesToMat, self).__init__(bigdl_type, byte_key)

class ImageFrameToSample(FeatureTransformer):
    """
    transform imageframe to samples
    :param input_keys keys that maps inputs (each input should be a tensor)
    :param target_keys keys that maps targets (each target should be a tensor)
    :param sample_key key to store sample
    """
    def __init__(self, input_keys=["imageTensor"], target_keys=None,
                 sample_key="sample", bigdl_type="float"):
        super(ImageFrameToSample, self).__init__(bigdl_type, input_keys, target_keys, sample_key)

class PixelBytesToMat(FeatureTransformer):
    """
    Transform byte array(pixels in byte) to OpenCVMat
    :param byte_key key that maps byte array
    """
    def __init__(self, byte_key = "bytes", bigdl_type="float"):
        super(PixelBytesToMat, self).__init__(bigdl_type, byte_key)

class FixExpand(FeatureTransformer):
    """
    Expand image with given expandHeight and expandWidth,
    put the original image to the center of expanded image
    :param expand_height height expand to
    :param expand_width width expand to
    """
    def __init__(self, expand_height, expand_width, bigdl_type="float"):
        super(FixExpand, self).__init__(bigdl_type, expand_height, expand_width)

class ChannelScaledNormalizer(FeatureTransformer):
    """
    Scaled image at channel level with offset and scale
    :param mean_r : offset for R channel
    :param mean_g : offset for G channel
    :param mean_b: offset for B channel
    :param scale: scaling factor for all channels
    """
    def __init__(self, mean_r, mean_g, mean_b, scale, bigdl_type="float"):
        super(ChannelScaledNormalizer, self).__init__(bigdl_type, mean_r, mean_g, mean_b, scale)

class RandomAlterAspect(FeatureTransformer):
    """
    Apply random crop based on area ratio and resize to cropLenth size
    :param min_area_ratio  min area ratio
    :param max_area_ratio  max area ratio
    :param min_aspect_ratio_change factor applied to ratio area
    :param interp_mode   interp mode applied in resize
    :param crop_length final size resized to
    """
    def __init__(self, min_area_ratio,
                 max_area_ratio,
                 min_aspect_ratio_change,
                 interp_mode,
                 crop_length, bigdl_type="float"):
        super(RandomAlterAspect, self).__init__(bigdl_type, min_area_ratio,
                                                max_area_ratio,
                                                min_aspect_ratio_change,
                                                interp_mode,
                                                crop_length)

class RandomCropper(FeatureTransformer):
    """
    Random cropper on uniform distribution with fixed height & width
    :param crop_w  width cropped to
    :param crop_h height cropped to
    :param mirror   whether mirror
    :param cropper_method crop method
    :param channels total channels
    """
    def __init__(self, crop_w, crop_h, mirror, cropper_method, channels, bigdl_type="float"):
        super(RandomCropper, self).__init__(bigdl_type, crop_w, crop_h, mirror, cropper_method, channels)

class RandomResize(FeatureTransformer):
    """
    Random resize between minSize and maxSize and scale height and width to each other
    :param min_size min size to resize to
    :param max_size max size to resize to
    """
    def __init__(self, min_size, max_size, bigdl_type="float"):
        super(RandomResize, self).__init__(bigdl_type, min_size, max_size)

class SeqFileFolder(JavaValue):

    @classmethod
    def files_to_image_frame(cls,
                             url,
                             sc,
                             class_num,
                             partition_num=-1,
                             bigdl_type="float"):
        """
        Extract hadoop sequence files from an HDFS path as ImageFrame
        :param url: sequence files folder path
        :param sc: spark context
        :param class_num: class number of data
        :param partition_num: partition number, default: Engine.nodeNumber() * Engine.coreNumber()
        """
        jvalue = callBigDlFunc(bigdl_type,
                               "seqFilesToImageFrame",
                               url,
                               sc,
                               class_num,
                               partition_num)
        return ImageFrame(jvalue=jvalue)


