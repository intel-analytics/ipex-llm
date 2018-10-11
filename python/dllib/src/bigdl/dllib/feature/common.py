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
import sys

if sys.version >= '3':
    long = int
    unicode = str


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
