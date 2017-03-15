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
from util.common import callBigDlFunc
from util.common import JavaValue
from util.common import callJavaFunc
from pyspark import SparkContext

import numpy as np

if sys.version >= '3':
    long = int
    unicode = str

INTMAX = 2147483647
INTMIN = -2147483648
DOUBLEMAX = 1.7976931348623157E308


class Model(JavaValue):

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    @classmethod
    def of(cls, jmodel, bigdl_type="float"):
        model = Model(bigdl_type, jmodel)
        model.value = jmodel
        model.bigdl_type = bigdl_type
        return model

    def set_name(self, name):
        callJavaFunc(SparkContext.getOrCreate(), self.value.setName, name)
        return self

    def name(self):
        return callJavaFunc(SparkContext.getOrCreate(), self.value.getName)

    def set_seed(self, seed=123):
        callBigDlFunc(self.bigdl_type, "setModelSeed", seed)
        return self

    def get_dtype(self):
        if "float" == self.bigdl_type:
            return "float32"
        else:
            return "float64"

    def reset(self):
        callJavaFunc(SparkContext.getOrCreate(), self.value.reset)
        return self

    def parameters(self):
        name_to_params = callBigDlFunc(self.bigdl_type,
                                       "modelGetParameters",
                                       self.value)

        def to_ndarray(params):
            return {
                param_name: np.array(values[0],
                                     dtype=self.get_dtype()).reshape(
                    values[1]) for param_name, values in params.iteritems()}

        return {layer_name: to_ndarray(params) for layer_name, params in
                name_to_params.iteritems()}

    def predict(self, data_rdd):
        return callBigDlFunc(self.bigdl_type,
                             "modelPredictRDD", self.value, data_rdd)

    def test(self, val_rdd, batch_size, val_methods):
        return callBigDlFunc(self.bigdl_type,
                             "modelTest",
                             self.value,
                             val_rdd, batch_size, val_methods)

    @staticmethod
    def from_path(path, bigdl_type="float"):
        jmodel = callBigDlFunc(bigdl_type, "modelFromPath", path)
        return Model.of(jmodel, bigdl_type)


class Linear(Model):

    '''
    >>> linear = Linear(100, 10, "Xavier")
    creating: createLinear
    '''

    def __init__(self, input_size, output_size, init_method="default",
                 bigdl_type="float"):
        super(Linear, self).__init__(None, bigdl_type, input_size, output_size,
                                     init_method)


class ReLU(Model):

    '''
    >>> relu = ReLU()
    creating: createReLU
    '''

    def __init__(self, bigdl_type="float"):
        super(ReLU, self).__init__(None, bigdl_type)


class Tanh(Model):

    '''
    >>> tanh = Tanh()
    creating: createTanh
    '''

    def __init__(self, bigdl_type="float"):
        super(Tanh, self).__init__(None, bigdl_type)


class Echo(Model):

    '''
    >>> echo = Echo()
    creating: createEcho
    '''

    def __init__(self, bigdl_type="float"):
        super(Echo, self).__init__(None, bigdl_type)


class LogSoftMax(Model):

    '''
    >>> logSoftMax = LogSoftMax()
    creating: createLogSoftMax
    '''

    def __init__(self, bigdl_type="float"):
        super(LogSoftMax, self).__init__(None, bigdl_type)


class Sequential(Model):

    '''
    >>> echo = Echo()
    creating: createEcho
    >>> s = Sequential()
    creating: createSequential
    >>> s = s.add(echo)
    >>> s = s.add(s)
    >>> s = s.add(echo)

    '''

    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type)

    def add(self, model):
        self.value.add(model.value)
        return self


class SpatialConvolution(Model):

    '''
    >>> spatialConvolution = SpatialConvolution(6, 12, 5, 5)
    creating: createSpatialConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 n_group=1,
                 propagate_back=True,
                 init_method="Default",
                 bigdl_type="float"):
        super(SpatialConvolution, self).__init__(None, bigdl_type,
                                                 n_input_plane,
                                                 n_output_plane,
                                                 kernel_w,
                                                 kernel_h,
                                                 stride_w,
                                                 stride_h,
                                                 pad_w,
                                                 pad_h,
                                                 n_group,
                                                 propagate_back,
                                                 init_method)


class SpatialMaxPooling(Model):

    '''
    >>> spatialMaxPooling = SpatialMaxPooling(2, 2, 2, 2)
    creating: createSpatialMaxPooling
    '''
    # to_ceil: call floor() when False; call ceil() when True

    def __init__(self, kw,
                 kh,
                 dw,
                 dh,
                 pad_w=0,
                 pad_h=0,
                 to_ceil=False,
                 bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type, kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h,
                                                to_ceil)

class Recurrent(Model):
    '''
    >>> recurrent = Recurrent()
    creating: createRecurrent
    '''

    def __init__(self, bigdl_type="float"):
        super(Recurrent, self).__init__(None, bigdl_type)

class LSTM(Model):
    '''
    >>> lstm = LSTM(4, 3)
    creating: createLSTM
    '''

    def __init__(self, input_size, hidden_size, bigdl_type="float"):
        super(LSTM, self).__init__(None, bigdl_type, input_size, hidden_size)

class GRU(Model):
    '''
    >>> gru = GRU(4, 3)
    creating: createGRU
    '''

    def __init__(self,  input_size, hidden_size, bigdl_type="float"):
        super(GRU, self).__init__(None, bigdl_type, input_size, hidden_size)

class RNNCell(Model):
    '''
    >>> reshape = RNNCell(4, 3, Tanh())
    creating: createTanh
    creating: createRNNCell
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 activation,
                 bigdl_type="float"):
        super(RNNCell, self).__init__(None, bigdl_type, input_size, hidden_size, activation)


class TimeDistributed(Model):
    '''
    >>> td = TimeDistributed(Linear(2, 3))
    creating: createLinear
    creating: createTimeDistributed
    '''

    def __init__(self, model, bigdl_type="float"):
        super(TimeDistributed, self).__init__(None, bigdl_type, model)

class TimeDistributedCriterion(Model):
    '''
    >>> from optim.optimizer import ClassNLLCriterion
    >>> td = TimeDistributedCriterion(ClassNLLCriterion())
    creating: createClassNLLCriterion
    creating: createTimeDistributedCriterion
    '''

    def __init__(self, criterion, bigdl_type="float"):
        super(TimeDistributedCriterion, self).__init__(None, bigdl_type, criterion)

class Concat(Model):

    '''
    >>> concat = Concat(2)
    creating: createConcat
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(Concat, self).__init__(None, bigdl_type,
                                     dimension)


class SpatialAveragePooling(Model):

    '''
    >>> spatialAveragePooling = SpatialAveragePooling(7,7)
    creating: createSpatialAveragePooling
    '''

    def __init__(self,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divide=True,
                 bigdl_type="float"):
        super(SpatialAveragePooling, self).__init__(None, bigdl_type,
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h,
                                                    ceil_mode,
                                                    count_include_pad,
                                                    divide)


class SpatialBatchNormalization(Model):

    '''
    >>> spatialBatchNormalization = SpatialBatchNormalization(1)
    creating: createSpatialBatchNormalization
    '''

    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 bigdl_type="float"):
        super(SpatialBatchNormalization, self).__init__(None, bigdl_type,
                                                        n_output,
                                                        eps,
                                                        momentum,
                                                        affine)


class SpatialCrossMapLRN(Model):

    '''
    >>> spatialCrossMapLRN = SpatialCrossMapLRN()
    creating: createSpatialCrossMapLRN
    '''

    def __init__(self,
                 size=5,
                 alpha=1.0,
                 beta=0.75,
                 k=1.0,
                 bigdl_type="float"):
        super(SpatialCrossMapLRN, self).__init__(None, bigdl_type,
                                                 size,
                                                 alpha,
                                                 beta,
                                                 k)


class Dropout(Model):

    '''
    >>> dropout = Dropout(0.4)
    creating: createDropout
    '''

    def __init__(self,
                 init_p=0.5,
                 inplace=False,
                 scale=True,
                 bigdl_type="float"):
        super(Dropout, self).__init__(None, bigdl_type,
                                      init_p,
                                      inplace,
                                      scale)


class View(Model):

    '''
    >>> view = View([1024,2])
    creating: createView
    '''

    def __init__(self,
                 sizes,
                 num_input_dims=0,
                 bigdl_type="float"):
        super(View, self).__init__(None, bigdl_type,
                                   sizes,
                                   num_input_dims)


class Abs(Model):

    '''
    >>> abs = Abs()
    creating: createAbs
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Abs, self).__init__(None, bigdl_type)


class Add(Model):

    '''
    >>> add = Add(1)
    creating: createAdd
    '''

    def __init__(self,
                 input_size,
                 bigdl_type="float"):
        super(Add, self).__init__(None, bigdl_type,
                                  input_size)


class AddConstant(Model):

    '''
    >>> addConstant = AddConstant(1e-5, True)
    creating: createAddConstant
    '''

    def __init__(self,
                 constant_scalar,
                 inplace=False,
                 bigdl_type="float"):
        super(AddConstant, self).__init__(None, bigdl_type,
                                          constant_scalar,
                                          inplace)


class BatchNormalization(Model):

    '''
    >>> batchNormalization = BatchNormalization(1, 1e-5, 1e-5, True)
    creating: createBatchNormalization
    '''

    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 n_output,
                                                 eps,
                                                 momentum,
                                                 affine)


class Bilinear(Model):

    '''
    >>> bilinear = Bilinear(1, 1, 1, True)
    creating: createBilinear
    '''

    def __init__(self,
                 input_size1,
                 input_size2,
                 output_size,
                 bias_res=True,
                 bigdl_type="float"):
        super(Bilinear, self).__init__(None, bigdl_type,
                                       input_size1,
                                       input_size2,
                                       output_size,
                                       bias_res)


class Bottle(Model):

    '''
    >>> bottle = Bottle(Linear(100,10), 1, 1)
    creating: createLinear
    creating: createBottle
    '''

    def __init__(self,
                 module,
                 n_input_dim=2,
                 n_output_dim1=INTMAX,
                 bigdl_type="float"):
        super(Bottle, self).__init__(None, bigdl_type,
                                     module,
                                     n_input_dim,
                                     n_output_dim1)


class CAdd(Model):

    '''
    >>> cAdd = CAdd([1,2])
    creating: createCAdd
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(CAdd, self).__init__(None, bigdl_type,
                                   size)


class CAddTable(Model):

    '''
    >>> cAddTable = CAddTable(True)
    creating: createCAddTable
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(CAddTable, self).__init__(None, bigdl_type,
                                        inplace)


class CDivTable(Model):

    '''
    >>> cDivTable = CDivTable()
    creating: createCDivTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CDivTable, self).__init__(None, bigdl_type)


class CMaxTable(Model):

    '''
    >>> cMaxTable = CMaxTable()
    creating: createCMaxTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMaxTable, self).__init__(None, bigdl_type)


class CMinTable(Model):

    '''
    >>> cMinTable = CMinTable()
    creating: createCMinTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMinTable, self).__init__(None, bigdl_type)


class CMul(Model):

    '''
    >>> cMul = CMul([1,2])
    creating: createCMul
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(CMul, self).__init__(None, bigdl_type,
                                   size)


class CMulTable(Model):

    '''
    >>> cMulTable = CMulTable()
    creating: createCMulTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMulTable, self).__init__(None, bigdl_type)


class CSubTable(Model):

    '''
    >>> cSubTable = CSubTable()
    creating: createCSubTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CSubTable, self).__init__(None, bigdl_type)


class Clamp(Model):

    '''
    >>> clamp = Clamp(1, 3)
    creating: createClamp
    '''

    def __init__(self,
                 min,
                 max,
                 bigdl_type="float"):
        super(Clamp, self).__init__(None, bigdl_type,
                                    min,
                                    max)


class Contiguous(Model):

    '''
    >>> contiguous = Contiguous()
    creating: createContiguous
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Contiguous, self).__init__(None, bigdl_type)


class Copy(Model):

    '''
    >>> copy = Copy()
    creating: createCopy
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Copy, self).__init__(None, bigdl_type)


class Cosine(Model):

    '''
    >>> cosine = Cosine(2,3)
    creating: createCosine
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 bigdl_type="float"):
        super(Cosine, self).__init__(None, bigdl_type,
                                     input_size,
                                     output_size)


class CosineDistance(Model):

    '''
    >>> cosineDistance = CosineDistance()
    creating: createCosineDistance
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CosineDistance, self).__init__(None, bigdl_type)


class DotProduct(Model):

    '''
    >>> dotProduct = DotProduct()
    creating: createDotProduct
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(DotProduct, self).__init__(None, bigdl_type)


class ELU(Model):

    '''
    >>> eLU = ELU(1e-5, True)
    creating: createELU
    '''

    def __init__(self,
                 alpha=1.0,
                 inplace=False,
                 bigdl_type="float"):
        super(ELU, self).__init__(None, bigdl_type,
                                  alpha,
                                  inplace)


class Euclidean(Model):

    '''
    >>> euclidean = Euclidean(1, 1, True)
    creating: createEuclidean
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 fast_backward=True,
                 bigdl_type="float"):
        super(Euclidean, self).__init__(None, bigdl_type,
                                        input_size,
                                        output_size,
                                        fast_backward)


class Exp(Model):

    '''
    >>> exp = Exp()
    creating: createExp
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Exp, self).__init__(None, bigdl_type)


class FlattenTable(Model):

    '''
    >>> flattenTable = FlattenTable()
    creating: createFlattenTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(FlattenTable, self).__init__(None, bigdl_type)


class GradientReversal(Model):

    '''
    >>> gradientReversal = GradientReversal(1e-5)
    creating: createGradientReversal
    '''

    def __init__(self,
                 the_lambda=1,
                 bigdl_type="float"):
        super(GradientReversal, self).__init__(None, bigdl_type,
                                               the_lambda)


class HardShrink(Model):

    '''
    >>> hardShrink = HardShrink(1e-5)
    creating: createHardShrink
    '''

    def __init__(self,
                 the_lambda=0.5,
                 bigdl_type="float"):
        super(HardShrink, self).__init__(None, bigdl_type,
                                         the_lambda)


class HardTanh(Model):

    '''
    >>> hardTanh = HardTanh(1e-5, 1e5, True)
    creating: createHardTanh
    '''

    def __init__(self,
                 min_value=-1,
                 max_value=1,
                 inplace=False,
                 bigdl_type="float"):
        super(HardTanh, self).__init__(None, bigdl_type,
                                       min_value,
                                       max_value,
                                       inplace)


class Index(Model):

    '''
    >>> index = Index(1)
    creating: createIndex
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(Index, self).__init__(None, bigdl_type,
                                    dimension)


class InferReshape(Model):

    '''
    >>> inferReshape = InferReshape([4, 0, 3, -1], False)
    creating: createInferReshape
    '''

    def __init__(self,
                 size,
                 batch_mode=False,
                 bigdl_type="float"):
        super(InferReshape, self).__init__(None, bigdl_type,
                                           size,
                                           batch_mode)


class JoinTable(Model):

    '''
    >>> joinTable = JoinTable(1, 1)
    creating: createJoinTable
    '''

    def __init__(self,
                 dimension,
                 n_input_dims,
                 bigdl_type="float"):
        super(JoinTable, self).__init__(None, bigdl_type,
                                        dimension,
                                        n_input_dims)


class L1Cost(Model):

    '''
    >>> l1Cost = L1Cost()
    creating: createL1Cost
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(L1Cost, self).__init__(None, bigdl_type)


class L1Penalty(Model):

    '''
    >>> l1Penalty = L1Penalty(1, True, True)
    creating: createL1Penalty
    '''

    def __init__(self,
                 l1weight,
                 size_average=False,
                 provide_output=True,
                 bigdl_type="float"):
        super(L1Penalty, self).__init__(None, bigdl_type,
                                        l1weight,
                                        size_average,
                                        provide_output)


class LeakyReLU(Model):

    '''
    >>> leakyReLU = LeakyReLU(1e-5, True)
    creating: createLeakyReLU
    '''

    def __init__(self,
                 negval=0.01,
                 inplace=False,
                 bigdl_type="float"):
        super(LeakyReLU, self).__init__(None, bigdl_type,
                                        negval,
                                        inplace)


class Log(Model):

    '''
    >>> log = Log()
    creating: createLog
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Log, self).__init__(None, bigdl_type)


class LogSigmoid(Model):

    '''
    >>> logSigmoid = LogSigmoid()
    creating: createLogSigmoid
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(LogSigmoid, self).__init__(None, bigdl_type)


class LookupTable(Model):

    '''
    >>> lookupTable = LookupTable(1, 1, 1e-5, 1e-5, 1e-5, True)
    creating: createLookupTable
    '''

    def __init__(self,
                 n_index,
                 n_output,
                 padding_value=0,
                 max_norm=DOUBLEMAX,
                 norm_type=2.0,
                 should_scale_grad_by_freq=False,
                 bigdl_type="float"):
        super(LookupTable, self).__init__(None, bigdl_type,
                                          n_index,
                                          n_output,
                                          padding_value,
                                          max_norm,
                                          norm_type,
                                          should_scale_grad_by_freq)


class MM(Model):

    '''
    >>> mM = MM(True, True)
    creating: createMM
    '''

    def __init__(self,
                 trans_a=False,
                 trans_b=False,
                 bigdl_type="float"):
        super(MM, self).__init__(None, bigdl_type,
                                 trans_a,
                                 trans_b)


class MV(Model):

    '''
    >>> mV = MV(True)
    creating: createMV
    '''

    def __init__(self,
                 trans=False,
                 bigdl_type="float"):
        super(MV, self).__init__(None, bigdl_type,
                                 trans)


class MapTable(Model):

    '''
    >>> mapTable = MapTable(Linear(100,10))
    creating: createLinear
    creating: createMapTable
    '''

    def __init__(self,
                 module,
                 bigdl_type="float"):
        super(MapTable, self).__init__(None, bigdl_type,
                                       module)


class MaskedSelect(Model):

    '''
    >>> maskedSelect = MaskedSelect()
    creating: createMaskedSelect
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MaskedSelect, self).__init__(None, bigdl_type)


class Max(Model):

    '''
    >>> max = Max(1)
    creating: createMax
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Max, self).__init__(None, bigdl_type,
                                  dim,
                                  num_input_dims)


class Mean(Model):

    '''
    >>> mean = Mean(1, 1)
    creating: createMean
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 bigdl_type="float"):
        super(Mean, self).__init__(None, bigdl_type,
                                   dimension,
                                   n_input_dims)


class Min(Model):

    '''
    >>> min = Min(1)
    creating: createMin
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Min, self).__init__(None, bigdl_type,
                                  dim,
                                  num_input_dims)


class MixtureTable(Model):

    '''
    >>> mixtureTable = MixtureTable()
    creating: createMixtureTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MixtureTable, self).__init__(None, bigdl_type)


class Mul(Model):

    '''
    >>> mul = Mul()
    creating: createMul
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Mul, self).__init__(None, bigdl_type)


class MulConstant(Model):

    '''
    >>> mulConstant = MulConstant(2.5)
    creating: createMulConstant
    '''

    def __init__(self,
                 scalar,
                 inplace=False,
                 bigdl_type="float"):
        super(MulConstant, self).__init__(None, bigdl_type,
                                          scalar,
                                          inplace)


class Narrow(Model):

    '''
    >>> narrow = Narrow(1, 1, 1)
    creating: createNarrow
    '''

    def __init__(self,
                 dimension,
                 offset,
                 length=1,
                 bigdl_type="float"):
        super(Narrow, self).__init__(None, bigdl_type,
                                     dimension,
                                     offset,
                                     length)


class NarrowTable(Model):

    '''
    >>> narrowTable = NarrowTable(1, 1)
    creating: createNarrowTable
    '''

    def __init__(self,
                 offset,
                 length=1,
                 bigdl_type="float"):
        super(NarrowTable, self).__init__(None, bigdl_type,
                                          offset,
                                          length)


class Normalize(Model):

    '''
    >>> normalize = Normalize(1e-5, 1e-5)
    creating: createNormalize
    '''

    def __init__(self,
                 p,
                 eps=1e-10,
                 bigdl_type="float"):
        super(Normalize, self).__init__(None, bigdl_type,
                                        p,
                                        eps)


class PReLU(Model):

    '''
    >>> pReLU = PReLU(1)
    creating: createPReLU
    '''

    def __init__(self,
                 n_output_plane=0,
                 bigdl_type="float"):
        super(PReLU, self).__init__(None, bigdl_type,
                                    n_output_plane)


class Padding(Model):

    '''
    >>> padding = Padding(1, 1, 1, 1e-5, 1)
    creating: createPadding
    '''

    def __init__(self,
                 dim,
                 pad,
                 n_input_dim,
                 value=0.0,
                 n_index=1,
                 bigdl_type="float"):
        super(Padding, self).__init__(None, bigdl_type,
                                      dim,
                                      pad,
                                      n_input_dim,
                                      value,
                                      n_index)


class PairwiseDistance(Model):

    '''
    >>> pairwiseDistance = PairwiseDistance(2)
    creating: createPairwiseDistance
    '''

    def __init__(self,
                 norm=2,
                 bigdl_type="float"):
        super(PairwiseDistance, self).__init__(None, bigdl_type,
                                               norm)


class ParallelTable(Model):

    '''
    >>> parallelTable = ParallelTable()
    creating: createParallelTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(ParallelTable, self).__init__(None, bigdl_type)


class Power(Model):

    '''
    >>> power = Power(1e-5)
    creating: createPower
    '''

    def __init__(self,
                 power,
                 scale=1.0,
                 shift=0.0,
                 bigdl_type="float"):
        super(Power, self).__init__(None, bigdl_type,
                                    power,
                                    scale,
                                    shift)


class RReLU(Model):

    '''
    >>> rReLU = RReLU(1e-5, 1e5, True)
    creating: createRReLU
    '''

    def __init__(self,
                 lower=1.0/8,
                 upper=1.0/3,
                 inplace=False,
                 bigdl_type="float"):
        super(RReLU, self).__init__(None, bigdl_type,
                                    lower,
                                    upper,
                                    inplace)


class ReLU6(Model):

    '''
    >>> reLU6 = ReLU6(True)
    creating: createReLU6
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(ReLU6, self).__init__(None, bigdl_type,
                                    inplace)


class Replicate(Model):

    '''
    >>> replicate = Replicate(2)
    creating: createReplicate
    '''

    def __init__(self,
                 n_features,
                 dim=1,
                 n_dim=INTMAX,
                 bigdl_type="float"):
        super(Replicate, self).__init__(None, bigdl_type,
                                        n_features,
                                        dim,
                                        n_dim)


class RoiPooling(Model):

    '''
    >>> roiPooling = RoiPooling(1, 1, 1e-5)
    creating: createRoiPooling
    '''

    def __init__(self,
                 pooled_w,
                 pooled_h,
                 spatial_scale,
                 bigdl_type="float"):
        super(RoiPooling, self).__init__(None, bigdl_type,
                                         pooled_w,
                                         pooled_h,
                                         spatial_scale)


class Scale(Model):

    '''
    >>> scale = Scale([1,2])
    creating: createScale
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(Scale, self).__init__(None, bigdl_type,
                                    size)


class Select(Model):

    '''
    >>> select = Select(1, 1)
    creating: createSelect
    '''

    def __init__(self,
                 dimension,
                 index,
                 bigdl_type="float"):
        super(Select, self).__init__(None, bigdl_type,
                                     dimension,
                                     index)


class SelectTable(Model):

    '''
    >>> selectTable = SelectTable(1)
    creating: createSelectTable
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(SelectTable, self).__init__(None, bigdl_type,
                                          dimension)


class Sigmoid(Model):

    '''
    >>> sigmoid = Sigmoid()
    creating: createSigmoid
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Sigmoid, self).__init__(None, bigdl_type)


class SoftMax(Model):

    '''
    >>> softMax = SoftMax()
    creating: createSoftMax
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftMax, self).__init__(None, bigdl_type)


class SoftMin(Model):

    '''
    >>> softMin = SoftMin()
    creating: createSoftMin
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftMin, self).__init__(None, bigdl_type)


class SoftPlus(Model):

    '''
    >>> softPlus = SoftPlus(1e-5)
    creating: createSoftPlus
    '''

    def __init__(self,
                 beta=1.0,
                 bigdl_type="float"):
        super(SoftPlus, self).__init__(None, bigdl_type,
                                       beta)


class SoftShrink(Model):

    '''
    >>> softShrink = SoftShrink(1e-5)
    creating: createSoftShrink
    '''

    def __init__(self,
                 the_lambda=0.5,
                 bigdl_type="float"):
        super(SoftShrink, self).__init__(None, bigdl_type,
                                         the_lambda)


class SoftSign(Model):

    '''
    >>> softSign = SoftSign()
    creating: createSoftSign
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftSign, self).__init__(None, bigdl_type)


class SpatialDilatedConvolution(Model):

    '''
    >>> spatialDilatedConvolution = SpatialDilatedConvolution(1, 1, 1, 1)
    creating: createSpatialDilatedConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 dilation_w=1,
                 dilation_h=1,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialDilatedConvolution, self).__init__(None, bigdl_type,
                                                        n_input_plane,
                                                        n_output_plane,
                                                        kw,
                                                        kh,
                                                        dw,
                                                        dh,
                                                        pad_w,
                                                        pad_h,
                                                        dilation_w,
                                                        dilation_h,
                                                        init_method)


class SpatialFullConvolution(Model):

    '''
    >>> spatialFullConvolution = SpatialFullConvolution(1, 1, 1, 1)
    creating: createSpatialFullConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 adj_w=0,
                 adj_h=0,
                 n_group=1,
                 no_bias=False,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialFullConvolution, self).__init__(None, bigdl_type,
                                                     n_input_plane,
                                                     n_output_plane,
                                                     kw,
                                                     kh,
                                                     dw,
                                                     dh,
                                                     pad_w,
                                                     pad_h,
                                                     adj_w,
                                                     adj_h,
                                                     n_group,
                                                     no_bias,
                                                     init_method)


class SpatialShareConvolution(Model):

    '''
    >>> spatialShareConvolution = SpatialShareConvolution(1, 1, 1, 1)
    creating: createSpatialShareConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 n_group=1,
                 propagate_back=True,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialShareConvolution, self).__init__(None, bigdl_type,
                                                      n_input_plane,
                                                      n_output_plane,
                                                      kernel_w,
                                                      kernel_h,
                                                      stride_w,
                                                      stride_h,
                                                      pad_w,
                                                      pad_h,
                                                      n_group,
                                                      propagate_back,
                                                      init_method)


class SpatialZeroPadding(Model):

    '''
    >>> spatialZeroPadding = SpatialZeroPadding(1, 1, 1, 1)
    creating: createSpatialZeroPadding
    '''

    def __init__(self,
                 pad_left,
                 pad_right,
                 pad_top,
                 pad_bottom,
                 bigdl_type="float"):
        super(SpatialZeroPadding, self).__init__(None, bigdl_type,
                                                 pad_left,
                                                 pad_right,
                                                 pad_top,
                                                 pad_bottom)


class SplitTable(Model):

    '''
    >>> splitTable = SplitTable(1, 1)
    creating: createSplitTable
    '''

    def __init__(self,
                 dimension,
                 n_input_dims=-1,
                 bigdl_type="float"):
        super(SplitTable, self).__init__(None, bigdl_type,
                                         dimension,
                                         n_input_dims)


class Sqrt(Model):

    '''
    >>> sqrt = Sqrt()
    creating: createSqrt
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Sqrt, self).__init__(None, bigdl_type)


class Square(Model):

    '''
    >>> square = Square()
    creating: createSquare
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Square, self).__init__(None, bigdl_type)


class Squeeze(Model):

    '''
    >>> squeeze = Squeeze(1)
    creating: createSqueeze
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Squeeze, self).__init__(None, bigdl_type,
                                      dim,
                                      num_input_dims)


class Sum(Model):

    '''
    >>> sum = Sum(1, 1, True)
    creating: createSum
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 size_average=False,
                 bigdl_type="float"):
        super(Sum, self).__init__(None, bigdl_type,
                                  dimension,
                                  n_input_dims,
                                  size_average)


class TanhShrink(Model):

    '''
    >>> tanhShrink = TanhShrink()
    creating: createTanhShrink
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(TanhShrink, self).__init__(None, bigdl_type)


class Threshold(Model):

    '''
    >>> threshold = Threshold(1e-5, 1e-5, True)
    creating: createThreshold
    '''

    def __init__(self,
                 th=1e-6,
                 v=0.0,
                 ip=False,
                 bigdl_type="float"):
        super(Threshold, self).__init__(None, bigdl_type,
                                        th,
                                        v,
                                        ip)


class Unsqueeze(Model):

    '''
    >>> unsqueeze = Unsqueeze(1, 1)
    creating: createUnsqueeze
    '''

    def __init__(self,
                 pos,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Unsqueeze, self).__init__(None, bigdl_type,
                                        pos,
                                        num_input_dims)


def _test():
    import doctest
    from pyspark import SparkContext
    from nn import layer
    globs = layer.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test layer")
    globs['sc'] = sc
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)


class Reshape(Model):
    '''
    >>> reshape = Reshape([1, 28, 28])
    creating: createReshape
    '''

    def __init__(self, size, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type, size)
if __name__ == "__main__":
    _test()
