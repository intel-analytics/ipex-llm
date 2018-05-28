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

from bigdl.util.common import *


class ImageSet(JavaValue):
    """
    ImageSet wraps a set of ImageFeature
    """

    def __init__(self, jvalue, bigdl_type="float"):
        self.value = jvalue
        self.bigdl_type = bigdl_type
        if self.is_local():
            self.image_set = LocalImageSet(jvalue=self.value)
        else:
            self.image_set = DistributedImageSet(jvalue=self.value)

    def is_local(self):
        """
        whether this is a LocalImageSet
        """
        return callBigDlFunc(self.bigdl_type, "isLocalImageSet", self.value)

    def is_distributed(self):
        """
        whether this is a DistributedImageSet
        """
        return callBigDlFunc(self.bigdl_type, "isDistributedImageSet", self.value)

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read images as Image Set
        if sc is defined, Read image as DistributedImageSet from local file system or HDFS
        if sc is null, Read image as LocalImageSet from local file system
        :param path path to read images
        if sc is defined, path can be local or HDFS. Wildcard character are supported.
        if sc is null, path is local directory/image file/image file with wildcard character
        :param sc SparkContext
        :param min_partitions A suggestion value of the minimal splitting number for input data.
        :return ImageSet
        """
        return ImageSet(jvalue=callBigDlFunc(bigdl_type, "readImageSet", path, sc, min_partitions))

    def transform(self, transformer, bigdl_type="float"):
        """
        transformImageSet
        """
        self.value = callBigDlFunc(bigdl_type, "transformImageSet", transformer, self.value)
        return self

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image from ImageSet
        """
        return self.image_set.get_image(float_key, to_chw)

    def get_label(self):
        """
        get label from ImageSet
        """
        return self.image_set.get_label()

    def get_predict(self, key="predict"):
        """
        get prediction from ImageSet
        """
        return self.image_set.get_predict(key)

    def to_image_frame(self, bigdl_type="float"):
        return callBigDlFunc(bigdl_type, "imageSetToImageFrame", self.value)


class LocalImageSet(ImageSet):
    """
    LocalImageSet wraps a list of ImageFeature
    """
    def __init__(self, image_list=None, label_list=None, jvalue=None, bigdl_type="float"):
        assert jvalue or image_list, "jvalue and image_list cannot be None in the same time"
        if jvalue:
            self.value = jvalue
        else:
            # init from image ndarray list and label rdd(optional)
            image_tensor_list = list(map(lambda image: JTensor.from_ndarray(image), image_list))
            label_tensor_list = list(map(lambda label: JTensor.from_ndarray(label), label_list))\
                if label_list else None
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       image_tensor_list, label_tensor_list)

        self.bigdl_type = bigdl_type

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image list from ImageSet
        """
        tensors = callBigDlFunc(self.bigdl_type, "localImageSetToImageTensor",
                                self.value, float_key, to_chw)
        return list(map(lambda tensor: tensor.to_ndarray(), tensors))

    def get_label(self):
        """
        get label list from ImageSet
        """
        labels = callBigDlFunc(self.bigdl_type, "localImageSetToLabelTensor", self.value)
        return map(lambda tensor: tensor.to_ndarray(), labels)

    def get_predict(self, key="predict"):
        """
        get prediction list from ImageSet
        """
        predicts = callBigDlFunc(self.bigdl_type, "localImageSetToPredict", self.value, key)
        return map(lambda predict:
                   (predict[0], predict[1].to_ndarray()) if predict[1]
                   else (predict[0], None), predicts)


class DistributedImageSet(ImageSet):
    """
    DistributedImageSet wraps an RDD of ImageFeature
    """

    def __init__(self, image_rdd=None, label_rdd=None, jvalue=None, bigdl_type="float"):
        assert jvalue or image_rdd, "jvalue and image_rdd cannot be None in the same time"
        if jvalue:
            self.value = jvalue
        else:
            # init from image ndarray rdd and label rdd(optional)
            image_tensor_rdd = image_rdd.map(lambda image: JTensor.from_ndarray(image))
            label_tensor_rdd = label_rdd.map(lambda label: JTensor.from_ndarray(label))\
                if label_rdd else None
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       image_tensor_rdd, label_tensor_rdd)

        self.bigdl_type = bigdl_type

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image rdd from ImageSet
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type, "distributedImageSetToImageTensorRdd",
                                   self.value, float_key, to_chw)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_label(self):
        """
        get label rdd from ImageSet
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type, "distributedImageSetToLabelTensorRdd",
                                   self.value)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_predict(self, key="predict"):
        """
        get prediction rdd from ImageSet
        """
        predicts = callBigDlFunc(self.bigdl_type, "distributedImageSetToPredict", self.value, key)
        return predicts.map(lambda predict:
                            (predict[0], predict[1].to_ndarray()) if predict[1]
                            else (predict[0], None))
