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
from bigdl.transform.vision.image import ImageFrame
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

    @property
    def label_map(self):
        """
        :return: the labelMap of this ImageSet, None if the ImageSet does not have a labelMap
        """
        return callBigDlFunc(self.bigdl_type, "imageSetGetLabelMap", self.value)

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, resize_height=-1,
             resize_width=-1, image_codec=-1, with_label=False, one_based_label=True,
             bigdl_type="float"):
        """
        Read images as Image Set
        if sc is defined, Read image as DistributedImageSet from local file system or HDFS
        if sc is null, Read image as LocalImageSet from local file system
        :param path path to read images
        if sc is defined, path can be local or HDFS. Wildcard character are supported.
        if sc is null, path is local directory/image file/image file with wildcard character

        if withLabel is set to true, path should be a directory that have two levels. The
        first level is class folders, and the second is images. All images belong to a same
        class should be put into the same class folder. So each image in the path is labeled by the
        folder it belongs.

        :param sc SparkContext
        :param min_partitions A suggestion value of the minimal splitting number for input data.
        :param resize_height height after resize, by default is -1 which will not resize the image
        :param resize_width width after resize, by default is -1 which will not resize the image
        :param image_codec specifying the color type of a loaded image, same as in OpenCV.imread.
               By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED(-1)
        :param with_label whether to treat folders in the path as image classification labels
               and read the labels into ImageSet.
        :param one_based_label whether to use one based label
        :return ImageSet
        """
        return ImageSet(jvalue=callBigDlFunc(bigdl_type, "readImageSet", path,
                                             sc, min_partitions, resize_height,
                                             resize_width, image_codec, with_label,
                                             one_based_label))

    @classmethod
    def from_image_frame(cls, image_frame, bigdl_type="float"):
        return ImageSet(jvalue=callBigDlFunc(bigdl_type, "imageFrameToImageSet", image_frame))

    @classmethod
    def from_rdds(cls, image_rdd, label_rdd=None, bigdl_type="float"):
        """
        Create a ImageSet from rdds of ndarray.

        :param image_rdd: a rdd of ndarray, each ndarray should has dimension of 3 or 4 (3D images)
        :param label_rdd: a rdd of ndarray
        :return: a DistributedImageSet
        """
        image_rdd = image_rdd.map(lambda x: JTensor.from_ndarray(x))
        if label_rdd is not None:
            label_rdd = label_rdd.map(lambda x: JTensor.from_ndarray(x))
        return ImageSet(jvalue=callBigDlFunc(bigdl_type, "createDistributedImageSet",
                                             image_rdd, label_rdd), bigdl_type=bigdl_type)

    def transform(self, transformer):
        """
        transformImageSet
        """
        return ImageSet(callBigDlFunc(self.bigdl_type, "transformImageSet",
                                      transformer, self.value), self.bigdl_type)

    def get_image(self, key="floats", to_chw=True):
        """
        get image from ImageSet
        """
        return self.image_set.get_image(key, to_chw)

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
        return ImageFrame(callBigDlFunc(bigdl_type, "imageSetToImageFrame", self.value), bigdl_type)


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

    def get_image(self, key="floats", to_chw=True):
        """
        get image list from ImageSet
        """
        tensors = callBigDlFunc(self.bigdl_type, "localImageSetToImageTensor",
                                self.value, key, to_chw)
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
        return list(map(lambda predict:
                        (predict[0], list(map(lambda x: x.to_ndarray(), predict[1]))) if predict[1]
                        else (predict[0], None), predicts))


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

    def get_image(self, key="floats", to_chw=True):
        """
        get image rdd from ImageSet
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type, "distributedImageSetToImageTensorRdd",
                                   self.value, key, to_chw)
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
                            (predict[0],
                             list(map(lambda x: x.to_ndarray(), predict[1]))) if predict[1]
                            else (predict[0], None))
