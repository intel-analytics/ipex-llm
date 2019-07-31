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
from bigdl.dataset.dataset import DataSet
import sys

if sys.version >= '3':
    long = int
    unicode = str


class Relation(object):
    """
    It represents the relationship between two items.
    """
    def __init__(self, id1, id2, label, bigdl_type="float"):
        self.id1 = id1
        self.id2 = id2
        self.label = int(label)
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return Relation, (self.id1, self.id2, self.label)

    def __str__(self):
        return "Relation [id1: %s, id2: %s, label: %s]" % (
            self.id1, self.id2, self.label)

    def to_tuple(self):
        return self.id1, self.id2, self.label


class Relations(object):
    @staticmethod
    def read(path, sc=None, min_partitions=1, bigdl_type="float"):
        """
        Read relations from csv or txt file.
        Each record is supposed to contain the following three fields in order:
        id1(string), id2(string) and label(int).

        For csv file, it should be without header.
        For txt file, each line should contain one record with fields separated by comma.

        :param path: The path to the relations file, which can either be a local or disrtibuted file
                     system (such as HDFS) path.
        :param sc: An instance of SparkContext.
                   If specified, return RDD of Relation.
                   Default is None and in this case return list of Relation.
        :param min_partitions: Int. A suggestion value of the minimal partition number for input
                               texts. Only need to specify this when sc is not None. Default is 1.
        """
        if sc:
            jvalue = callBigDlFunc(bigdl_type, "readRelations", path, sc, min_partitions)
            res = jvalue.map(lambda x: Relation(str(x[0]), str(x[1]), int(x[2])))
        else:
            jvalue = callBigDlFunc(bigdl_type, "readRelations", path)
            res = [Relation(str(x[0]), str(x[1]), int(x[2])) for x in jvalue]
        return res

    @staticmethod
    def read_parquet(path, sc, bigdl_type="float"):
        """
        Read relations from parquet file.
        Schema should be the following:
        "id1"(string), "id2"(string) and "label"(int).

        :param path: The path to the parquet file.
        :param sc: An instance of SparkContext.
        :return: RDD of Relation.
        """
        jvalue = callBigDlFunc(bigdl_type, "readRelationsParquet", path, sc)
        return jvalue.map(lambda x: Relation(str(x[0]), str(x[1]), int(x[2])))


class Preprocessing(JavaValue):
    """
    Preprocessing defines data transform action during feature preprocessing. Python wrapper for
    the scala Preprocessing
    """
    def __init__(self, bigdl_type="float", *args):
        self.bigdl_type = bigdl_type
        self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def __call__(self, input):
        """
        Transform ImageSet or TextSet.
        """
        # move the import here to break circular import
        if "zoo.feature.image.imageset.ImageSet" not in sys.modules:
            from zoo.feature.image import ImageSet
        if "zoo.feature.text.text_set.TextSet" not in sys.modules:
            from zoo.feature.text import TextSet
        # if type(input) is ImageSet:
        if isinstance(input, ImageSet):
            jset = callBigDlFunc(self.bigdl_type, "transformImageSet", self.value, input)
            return ImageSet(jvalue=jset)
        elif isinstance(input, TextSet):
            jset = callBigDlFunc(self.bigdl_type, "transformTextSet", self.value, input)
            return TextSet(jvalue=jset)


class ChainedPreprocessing(Preprocessing):
    """
    chains two Preprocessing together. The output type of the first
    Preprocessing should be the same with the input type of the second Preprocessing.
    """
    def __init__(self, transformers, bigdl_type="float"):
        for transfomer in transformers:
            assert isinstance(transfomer, Preprocessing), \
                str(transfomer) + " should be subclass of Preprocessing "

        super(ChainedPreprocessing, self).__init__(bigdl_type, transformers)


class ScalarToTensor(Preprocessing):
    """
    a Preprocessing that converts a number to a Tensor.
    """
    def __init__(self, bigdl_type="float"):
        super(ScalarToTensor, self).__init__(bigdl_type)


class SeqToTensor(Preprocessing):
    """
    a Transformer that converts an Array[_] or Seq[_] to a Tensor.
    :param size dimensions of target Tensor.
    """
    def __init__(self, size=[], bigdl_type="float"):
        super(SeqToTensor, self).__init__(bigdl_type, size)


class ArrayToTensor(Preprocessing):
    """
    a Transformer that converts an Array[_] to a Tensor.
    :param size dimensions of target Tensor.
    """
    def __init__(self, size, bigdl_type="float"):
        super(ArrayToTensor, self).__init__(bigdl_type, size)


class MLlibVectorToTensor(Preprocessing):
    """
    a Transformer that converts MLlib Vector to a Tensor.
    .. note:: Deprecated in 0.4.0. NNEstimator will automatically extract Vectors now.
    :param size dimensions of target Tensor.
    """
    def __init__(self, size, bigdl_type="float"):
        super(MLlibVectorToTensor, self).__init__(bigdl_type, size)


class FeatureLabelPreprocessing(Preprocessing):
    """
    construct a Transformer that convert (Feature, Label) tuple to a Sample.
    The returned Transformer is robust for the case label = null, in which the
    Sample is derived from Feature only.
    :param feature_transformer transformer for feature, transform F to Tensor[T]
    :param label_transformer transformer for label, transform L to Tensor[T]
    """
    def __init__(self, feature_transformer, label_transformer, bigdl_type="float"):
        super(FeatureLabelPreprocessing, self).__init__(bigdl_type,
                                                        feature_transformer, label_transformer)


class TensorToSample(Preprocessing):
    """
     a Transformer that converts Tensor to Sample.
    """
    def __init__(self, bigdl_type="float"):
        super(TensorToSample, self).__init__(bigdl_type)


class FeatureToTupleAdapter(Preprocessing):
    def __init__(self, sample_transformer, bigdl_type="float"):
        super(FeatureToTupleAdapter, self).__init__(bigdl_type, sample_transformer)


class BigDLAdapter(Preprocessing):
    def __init__(self, bigdl_transformer, bigdl_type="float"):
        super(BigDLAdapter, self).__init__(bigdl_type, bigdl_transformer)


class ToTuple(Preprocessing):
    """
     a Transformer that converts Feature to (Feature, None).
    """
    def __init__(self, bigdl_type="float"):
        super(ToTuple, self).__init__(bigdl_type)


class FeatureSet(DataSet):
    """
    A set of data which is used in the model optimization process. The FeatureSet can be accessed in
    a random data sample sequence. In the training process, the data sequence is a looped endless
    sequence. While in the validation process, the data sequence is a limited length sequence.
    Different from BigDL's DataSet, this FeatureSet could be cached to Intel Optane DC Persistent
    Memory, if you set memory_type to PMEM when creating FeatureSet.
    """
    def __init__(self, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if jvalue:
            self.value = jvalue

    @classmethod
    def image_frame(cls, image_frame, memory_type="DRAM", bigdl_type="float"):
        """
        Create FeatureSet from ImageFrame.
        :param image_frame: ImageFrame
        :param memory_type: string, DRAM, PMEM or a Int number.
                            If it's DRAM, will cache dataset into dynamic random-access memory
                            If it's PMEM, will cache dataset into Intel Optane DC Persistent Memory
                            If it's a Int number n, will cache dataset into disk, and only hold 1/n
                              of the data into memory during the training. After going through the
                              1/n, we will release the current cache, and load another 1/n into
                              memory.
        :param bigdl_type: numeric type
        :return: A feature set
        """
        jvalue = callBigDlFunc(bigdl_type, "createFeatureSetFromImageFrame",
                               image_frame, memory_type)
        return cls(jvalue=jvalue)

    @classmethod
    def image_set(cls, imageset, memory_type="DRAM", bigdl_type="float"):
        """
        Create FeatureSet from ImageFrame.
        :param image_frame: ImageFrame
        :param memory_type: string, DRAM or PMEM
                            If it's DRAM, will cache dataset into dynamic random-access memory
                            If it's PMEM, will cache dataset into Intel Optane DC Persistent Memory
        :param bigdl_type: numeric type
        :return: A feature set
        """
        jvalue = callBigDlFunc(bigdl_type, "createFeatureSetFromImageFrame",
                               imageset.to_image_frame(), memory_type)
        return cls(jvalue=jvalue)

    @classmethod
    def rdd(cls, rdd, memory_type="DRAM", bigdl_type="float"):
        """
        Create FeatureSet from RDD.
        :param rdd: A RDD
        :param memory_type: string, DRAM, PMEM or a Int number.
                            If it's DRAM, will cache dataset into dynamic random-access memory
                            If it's PMEM, will cache dataset into Intel Optane DC Persistent Memory
                            If it's a Int number n, will cache dataset into disk, and only hold 1/n
                              of the data into memory during the training. After going through the
                              1/n, we will release the current cache, and load another 1/n into
                              memory.
        :param bigdl_type:numeric type
        :return: A feature set
        """
        jvalue = callBigDlFunc(bigdl_type, "createFeatureSetFromRDD", rdd, memory_type)
        return cls(jvalue=jvalue)

    def transform(self, transformer):
        """
        Helper function to transform the data type in the data set.
        :param transformer: the transformers to transform this feature set.
        :return: A feature set
        """
        jvalue = callBigDlFunc(self.bigdl_type, "transformFeatureSet", self.value, transformer)
        return FeatureSet(jvalue=jvalue)

    def to_dataset(self):
        """
        To BigDL compatible DataSet
        :return:
        """
        jvalue = callBigDlFunc(self.bigdl_type, "featureSetToDataSet", self.value)
        return FeatureSet(jvalue=jvalue)
