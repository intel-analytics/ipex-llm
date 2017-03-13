<<<<<<< HEAD
#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


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

    def get_dtype(self):
        if "float" == self.bigdl_type:
            return "float32"
        else:
            return "float64"

    def parameters(self):
        (w, g, wshape, gshape) = callBigDlFunc(self.bigdl_type,
                                               "modelGetParameters",
                                               self.value)
        return \
            {"weights": np.array(w, dtype=self.get_dtype()).reshape(wshape),
             "gradients": np.array(g, dtype=self.get_dtype()).reshape(gshape)}

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
        return callBigDlFunc(bigdl_type, "modelFromPath", path)


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

    def __init__(self, kw,
                 kh,
                 dw,
                 dh,
                 pad_w=0,
                 pad_h=0, bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type, kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h)



class Reshape(Model):
    '''
    >>> reshape = Reshape([1, 28, 28])
    creating: createReshape
    '''

    def __init__(self, size, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type, size)


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
                bigdl_type="float"):
        super(View, self).__init__(None, bigdl_type,
                                   sizes)

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



if __name__ == "__main__":
    _test()
=======
#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


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

    def get_dtype(self):
        if "float" == self.bigdl_type:
            return "float32"
        else:
            return "float64"

    def parameters(self):
        (w, g, wshape, gshape) = callBigDlFunc(self.bigdl_type,
                                               "modelGetParameters",
                                               self.value)
        return \
            {"weights": np.array(w, dtype=self.get_dtype()).reshape(wshape),
             "gradients": np.array(g, dtype=self.get_dtype()).reshape(gshape)}

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
        return callBigDlFunc(bigdl_type, "modelFromPath", path)


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

    def __init__(self, kw,
                 kh,
                 dw,
                 dh,
                 pad_w=0,
                 pad_h=0, bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type, kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h)


class Reshape(Model):
    '''
    >>> reshape = Reshape([1, 28, 28])
    creating: createReshape
    '''

    def __init__(self, size, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type, size)


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
                 bigdl_type="float"):
        super(View, self).__init__(None, bigdl_type,
                                   sizes)


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


if __name__ == "__main__":
    _test()
>>>>>>> 0de2849bf0f7dd75d13d9d91c2a11bc92b31d87d
