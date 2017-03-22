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

from util.common import JavaValue
from util.common import callBigDlFunc
from util.common import JTensor
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class Criterion(JavaValue):

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    @classmethod
    def of(cls, jcriterion, bigdl_type="float"):
        criterion = Criterion(bigdl_type, jcriterion)
        criterion.value = jcriterion
        criterion.bigdl_type = bigdl_type
        return criterion


class ClassNLLCriterion(Criterion):

    '''
    >>> classNLLCriterion = ClassNLLCriterion()
    creating: createClassNLLCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(ClassNLLCriterion, self).__init__(None, bigdl_type,
                                                size_average)


class MSECriterion(Criterion):

    '''
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    '''

    def __init__(self, bigdl_type="float"):
        super(MSECriterion, self).__init__(None, bigdl_type)


class AbsCriterion(Criterion):

    '''
    >>> absCriterion = AbsCriterion(True)
    creating: createAbsCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(AbsCriterion, self).__init__(None, bigdl_type,
                                           size_average)


class ClassSimplexCriterion(Criterion):

    '''
    >>> classSimplexCriterion = ClassSimplexCriterion(2)
    creating: createClassSimplexCriterion
    '''

    def __init__(self,
                 n_classes,
                 bigdl_type="float"):
        super(ClassSimplexCriterion, self).__init__(None, bigdl_type,
                                                    n_classes)


class CosineEmbeddingCriterion(Criterion):

    '''
    >>> cosineEmbeddingCriterion = CosineEmbeddingCriterion(1e-5, True)
    creating: createCosineEmbeddingCriterion
    '''

    def __init__(self,
                 margin=0.0,
                 size_average=True,
                 bigdl_type="float"):
        super(CosineEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                       margin,
                                                       size_average)


class DistKLDivCriterion(Criterion):

    '''
    >>> distKLDivCriterion = DistKLDivCriterion(True)
    creating: createDistKLDivCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(DistKLDivCriterion, self).__init__(None, bigdl_type,
                                                 size_average)


class HingeEmbeddingCriterion(Criterion):

    '''
    >>> hingeEmbeddingCriterion = HingeEmbeddingCriterion(1e-5, True)
    creating: createHingeEmbeddingCriterion
    '''

    def __init__(self,
                 margin=1,
                 size_average=True,
                 bigdl_type="float"):
        super(HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                      margin,
                                                      size_average)


class L1HingeEmbeddingCriterion(Criterion):

    '''
    >>> l1HingeEmbeddingCriterion = L1HingeEmbeddingCriterion(1e-5)
    creating: createL1HingeEmbeddingCriterion
    '''

    def __init__(self,
                 margin=1,
                 bigdl_type="float"):
        super(L1HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                        margin)


class MarginCriterion(Criterion):

    '''
    >>> marginCriterion = MarginCriterion(1e-5, True)
    creating: createMarginCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(MarginCriterion, self).__init__(None, bigdl_type,
                                              margin,
                                              size_average)


class MarginRankingCriterion(Criterion):

    '''
    >>> marginRankingCriterion = MarginRankingCriterion(1e-5, True)
    creating: createMarginRankingCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(MarginRankingCriterion, self).__init__(None, bigdl_type,
                                                     margin,
                                                     size_average)


class MultiCriterion(Criterion):

    '''
    >>> multiCriterion = MultiCriterion()
    creating: createMultiCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MultiCriterion, self).__init__(None, bigdl_type)


class MultiLabelMarginCriterion(Criterion):

    '''
    >>> multiLabelMarginCriterion = MultiLabelMarginCriterion(True)
    creating: createMultiLabelMarginCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(MultiLabelMarginCriterion, self).__init__(None, bigdl_type,
                                                        size_average)


class ParallelCriterion(Criterion):

    '''
    >>> parallelCriterion = ParallelCriterion(True)
    creating: createParallelCriterion
    '''

    def __init__(self,
                 repeat_target=False,
                 bigdl_type="float"):
        super(ParallelCriterion, self).__init__(None, bigdl_type,
                                                repeat_target)


class SmoothL1Criterion(Criterion):

    '''
    >>> smoothL1Criterion = SmoothL1Criterion(True)
    creating: createSmoothL1Criterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(SmoothL1Criterion, self).__init__(None, bigdl_type,
                                                size_average)


class SmoothL1CriterionWithWeights(Criterion):

    '''
    >>> smoothL1CriterionWithWeights = SmoothL1CriterionWithWeights(1e-5, 1)
    creating: createSmoothL1CriterionWithWeights
    '''

    def __init__(self,
                 sigma,
                 num=0,
                 bigdl_type="float"):
        super(SmoothL1CriterionWithWeights, self).__init__(None, bigdl_type,
                                                           sigma,
                                                           num)


class SoftmaxWithCriterion(Criterion):

    '''
    >>> softmaxWithCriterion = SoftmaxWithCriterion()
    creating: createSoftmaxWithCriterion
    >>> softmaxWithCriterion = SoftmaxWithCriterion(1, "FULL")
    creating: createSoftmaxWithCriterion
    '''

    def __init__(self,
                 ignore_label=None,
                 normalize_mode="VALID",
                 bigdl_type="float"):
        super(SoftmaxWithCriterion, self).__init__(None, bigdl_type,
                                                   ignore_label,
                                                   normalize_mode)


class TimeDistributedCriterion(Criterion):
    '''
    >>> td = TimeDistributedCriterion(ClassNLLCriterion())
    creating: createClassNLLCriterion
    creating: createTimeDistributedCriterion
    '''

    def __init__(self, criterion, size_average=False, bigdl_type="float"):
        super(TimeDistributedCriterion, self).__init__(
            None, bigdl_type, criterion, size_average)


class CrossEntropyCriterion(Criterion):
    """
    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> cec = CrossEntropyCriterion(weights)
    creating: createCrossEntropyCriterion
    >>> cec = CrossEntropyCriterion()
    creating: createCrossEntropyCriterion
    """

    def __init__(self,
                 weights=np.zeros([2]),
                 size_average=True,
                 bigdl_type="float"):
        super(CrossEntropyCriterion, self).__init__(None, bigdl_type,
                                                    JTensor.from_ndarray(
                                                        weights),
                                                    size_average)

def _test():
    import doctest
    from pyspark import SparkContext
    from nn import criterion
    from util.common import init_engine
    from util.common import create_spark_conf
    globs = criterion.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test criterion",
                      conf=create_spark_conf())
    globs['sc'] = sc
    init_engine()

    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
