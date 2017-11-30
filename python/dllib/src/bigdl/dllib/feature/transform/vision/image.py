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

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        """
        ImageFeature to sample
        """
        return callBigDlFunc(self.bigdl_type, "imageFeatureToSample", self.value, float_key, to_chw, with_im_info)

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
    def read(cls, path, sc=None, bigdl_type="float"):
        """
        Read images as Image Frame
        if sc is defined, Read image as DistributedImageFrame from local file system or HDFS
        if sc is null, Read image as LocalImageFrame from local file system
        :param path path to read images
        if sc is defined, path can be local or HDFS. Wildcard character are supported.
        if sc is null, path is local directory/image file/image file with wildcard character
        :param sc SparkContext
        :return ImageFrame
        """
        return ImageFrame(jvalue=callBigDlFunc(bigdl_type, "read", path, sc))

    @classmethod
    def readParquet(cls, path, sql_context, bigdl_type="float"):
        """
        Read parquet file as DistributedImageFrame
        """
        return DistributedImageFrame(jvalue=callBigDlFunc(bigdl_type, "readParquet", path, sql_context))

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

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        """
        ImageFrame toSample
        """
        return self.image_frame.to_sample(float_key, to_chw, with_im_info)


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
            image_tensor_list = image_list.map(lambda image: JTensor.from_ndarray(image))
            label_tensor_list = label_list.map(lambda label: JTensor.from_ndarray(label)) if label_list else None
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

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        """
        to sample list
        """
        return callBigDlFunc(self.bigdl_type,
                             "localImageFrameToSample", self.value, float_key, to_chw, with_im_info)



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

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        """
        to sample rdd
        """
        return callBigDlFunc(self.bigdl_type,
                             "distributedImageFrameToSampleRdd", self.value, float_key, to_chw, with_im_info)

class HFlip(FeatureTransformer):
    """
    Flip the image horizontally
    """
    def __init__(self, bigdl_type="float"):
            super(HFlip, self).__init__(bigdl_type)
