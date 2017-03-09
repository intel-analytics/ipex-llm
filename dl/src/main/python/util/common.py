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
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap

from pyspark import RDD, SparkContext
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame, SQLContext
from pyspark.mllib.common import callJavaFunc
from pyspark import SparkConf
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class JavaValue(object):
    def jvm_class_constructor(self):
        name = "create" + self.__class__.__name__
        print("creating: " + name)
        return name

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), *args)
        self.bigdl_type = bigdl_type


class TestResult():
    def __init__(self, result, total_num, method):
        self.result = result
        self.total_num = total_num
        self.method = method

    def __reduce__(self):
        return (TestResult, (self.result, self.total_num, self.method))

    def __str__(self):
        return "Test result: %s, total_num: %s, method: %s" % (
            self.result, self.total_num, self.method)


class Sample(object):
    def __init__(self, features, label, features_shape, label_shape,
                 bigdl_type="float"):
        def get_dtype():
            if "float" == bigdl_type:
                return "float32"
            else:
                return "float64"
        self.features = np.array(features, dtype=get_dtype()).reshape(features_shape)  # noqa
        self.label = np.array(label, dtype=get_dtype()).reshape(label_shape)
        self.bigdl_type = bigdl_type

    @classmethod
    def from_ndarray(cls, features, label, bigdl_type="float"):
        return cls(
            features=[float(i) for i in features.ravel()],
            label=[float(i) for i in label.ravel()],
            features_shape=list(features.shape),
            label_shape=list(label.shape) if label.shape else [label.size],
            bigdl_type=bigdl_type)

    @classmethod
    def flatten(cls, a_ndarray):
        """
        Utility method to flatten a ndarray
        :return (storage, shape)
        >>> import numpy as np
        >>> from util.common import Sample
        >>> np.random.seed(123)
        >>> data = np.random.uniform(0, 1, (2, 3))
        >>> (storage, shape) = Sample.flatten(data)
        >>> shape
        [2, 3]
        >>> (storage, shape) = Sample.flatten(np.array(2))
        >>> shape
        [1]
        """
        storage = [float(i) for i in a_ndarray.ravel()]
        shape = list(a_ndarray.shape) if a_ndarray.shape else [a_ndarray.size]
        return storage, shape

    def __reduce__(self):
        (features_storage, features_shape) = Sample.flatten(self.features)
        (label_storage, label_shape) = Sample.flatten(self.label)
        return (Sample, (
            features_storage, label_storage, features_shape, label_shape,
            self.bigdl_type))

    def __str__(self):
        return "features: %s, label: %s," % (self.features, self.label)


_picklable_classes = [
    'LinkedList',
    'SparseVector',
    'DenseVector',
    'DenseMatrix',
    'Rating',
    'LabeledPoint',
    'Sample',
    'TestResult'
]


def initEngine(bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initEngine")


def callBigDlFunc(bigdl_type, name, *args):
    """ Call API in PythonBigDL """
    sc = SparkContext.getOrCreate()
    if bigdl_type == "float":
        api = getattr(
            sc._jvm.com.intel.analytics.bigdl.python.api.PythonBigDL.ofFloat(),
            name)
    elif bigdl_type == "double":
        api = getattr(
            sc._jvm.com.intel.analytics.bigdl.python.api.PythonBigDL.ofDouble(),
            name)
    else:
        raise Exception("Not supported bigdl_type: %s" % bigdl_type)
    return callJavaFunc(sc, api, *args)


def _java2py(sc, r, encoding="bytes"):
    if isinstance(r, JavaObject):
        clsName = r.getClass().getSimpleName()
        # convert RDD into JavaRDD
        if clsName != 'JavaRDD' and clsName.endswith("RDD"):
            r = r.toJavaRDD()
            clsName = 'JavaRDD'

        if clsName == 'JavaRDD':
            jrdd = sc._jvm.SerDe.javaToPython(r)
            return RDD(jrdd, sc)

        if clsName == 'DataFrame':
            return DataFrame(r, SQLContext.getOrCreate(sc))

        if clsName in _picklable_classes:
            r = sc._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.dumps(r)
        elif isinstance(r, (JavaArray, JavaList, JavaMap)):
            try:
                r = sc._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.dumps(
                    r)
            except Py4JJavaError:
                pass  # not pickable

    if isinstance(r, (bytearray, bytes)):
        r = PickleSerializer().loads(bytes(r), encoding=encoding)
    return r


def callJavaFunc(sc, func, *args):
    """ Call Java Function """
    args = [_py2java(sc, a) for a in args]
    result = func(*args)
    return _java2py(sc, result)


def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever
    the RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return \
        rdd.ctx._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.pythonToJava(
            rdd._jrdd, True)


def _py2java(sc, obj):
    """ Convert Python object into Java """
    if isinstance(obj, RDD):
        obj = _to_java_object_rdd(obj)
    elif isinstance(obj, DataFrame):
        obj = obj._jdf
    elif isinstance(obj, SparkContext):
        obj = obj._jsc
    elif isinstance(obj, (list, tuple)):
        obj = ListConverter().convert([_py2java(sc, x) for x in obj],
                                      sc._gateway._gateway_client)
    elif isinstance(obj, dict):
        result = {}
        for (key, value) in obj.iteritems():
            result[key] = _py2java(sc, value) if isinstance(value, JavaValue) else value  # noqa
        obj = result

    elif isinstance(obj, JavaValue):
        obj = obj.value
    elif isinstance(obj, JavaObject):
        pass
    elif isinstance(obj, (int, long, float, bool, bytes, unicode)):
        pass
    else:
        data = bytearray(PickleSerializer().dumps(obj))
        obj = sc._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.loads(data)
    return obj


def _test():
    import doctest
    from pyspark import SparkContext
    from nn import layer
    globs = layer.__dict__.copy()
    sc = SparkContext(master="local[2]", appName="test common utility")
    globs['sc'] = sc
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
