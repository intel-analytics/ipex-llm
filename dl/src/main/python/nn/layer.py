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


from util.common import JavaValue
import sys

if sys.version >= '3':
    long = int
    unicode = str


class Linear(JavaValue):
    '''
    >>> linear = Linear(100, 10, "Xavier")
    creating: createLinear
    '''
    def __init__(self, input_size, output_size, init_method="default",
                 bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type, input_size, output_size,
                           init_method)


class ReLU(JavaValue):
    '''
    >>> relu = ReLU()
    creating: createReLU
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type)


class Tanh(JavaValue):
    '''
    >>> tanh = Tanh()
    creating: createTanh
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type)


class Echo(JavaValue):
    '''
    >>> echo = Echo()
    creating: createEcho
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type)


class LogSoftMax(JavaValue):
    '''
    >>> logSoftMax = LogSoftMax()
    creating: createLogSoftMax
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type)


class Sequential(JavaValue):
    '''
    >>> sequential = Sequential()
    creating: createSequential
    '''
    def __init__(self, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type)

    def add(self, model):
        self.value.add(model.value)
        return self


class SpatialConvolution(JavaValue):
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
        JavaValue.__init__(self, bigdl_type,
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


class SpatialMaxPooling(JavaValue):
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
        JavaValue.__init__(self, bigdl_type, kw,
                           kh,
                           dw,
                           dh,
                           pad_w,
                           pad_h)


class Reshape(JavaValue):
    '''
    >>> reshape = Reshape([1, 28, 28])
    creating: createReshape
    '''
    def __init__(self, size, bigdl_type="float"):
        JavaValue.__init__(self, bigdl_type, size)


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
