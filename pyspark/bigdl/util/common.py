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

import os
import sys
import glob
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap

from pyspark import RDD, SparkContext
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame, SQLContext
from pyspark.mllib.common import callJavaFunc
from pyspark import SparkConf
import numpy as np
import threading
from bigdl.util.engine import prepare_env

INTMAX = 2147483647
INTMIN = -2147483648
DOUBLEMAX = 1.7976931348623157E308

if sys.version >= '3':
    long = int
    unicode = str

class SingletonMixin(object):
    _lock = threading.RLock()
    _instance = None

    @classmethod
    def instance(cls,
                 bigdl_type="float"):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls(bigdl_type)
        return cls._instance

class JavaCreator(SingletonMixin):
    __creator_class="com.intel.analytics.bigdl.python.api.PythonBigDL"

    @classmethod
    def get_creator_class(cls):
        with JavaCreator._lock:
            return JavaCreator.__creator_class

    @classmethod
    def set_creator_class(cls, cclass):
        with JavaCreator._lock:
            JavaCreator.__creator_class = cclass
            JavaCreator._instance = None

    def __init__(self, bigdl_type):
        sc = get_spark_context()
        jclass = getattr(sc._jvm, JavaCreator.get_creator_class())
        if bigdl_type == "float":
            self.value = getattr(jclass, "ofFloat")()
        elif bigdl_type == "double":
            self.value = getattr(jclass, "ofDouble")()
        else:
            raise Exception("Not supported bigdl_type: %s" % bigdl_type)


class JavaValue(object):
    def jvm_class_constructor(self):
        name = "create" + self.__class__.__name__
        print("creating: " + name)
        return name

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), *args)
        self.bigdl_type = bigdl_type

    def __str__(self):
        return self.value.toString()


class TestResult():
    """
    A testing result used to benchmark the model quality.
    """
    def __init__(self, result, total_num, method):
        """

        :param result: the validation result. i.e: top1 accuracy percentage.
        :param total_num: the total processed records.
        :param method: the validation method. i.e: Top1Accuracy
        """
        self.result = result
        self.total_num = total_num
        self.method = method

    def __reduce__(self):
        return (TestResult, (self.result, self.total_num, self.method))

    def __str__(self):
        return "Test result: %s, total_num: %s, method: %s" % (
            self.result, self.total_num, self.method)

def get_dtype(bigdl_type):
    # Always return float32 for now
    return "float32"

class JTensor(object):
    """
    A wrapper to easy our work when need to pass or return Tensor to/from Scala.


    >>> import numpy as np
    >>> from bigdl.util.common import JTensor
    >>> np.random.seed(123)
    >>>
    """
    def __init__(self, storage, shape, bigdl_type="float"):
        if isinstance(storage, bytes) and isinstance(shape, bytes):
            self.storage = np.frombuffer(storage, dtype=get_dtype(bigdl_type))
            self.shape = np.frombuffer(shape, dtype=np.int32)
        else:
            self.storage = np.array(storage, dtype=get_dtype(bigdl_type))
            self.shape = np.array(shape, dtype=np.int32)
        self.bigdl_type = bigdl_type

    @classmethod
    def from_ndarray(cls, a_ndarray, bigdl_type="float"):
        """
        Convert a ndarray to Tensor which would be used in Java side.

        >>> import numpy as np
        >>> from bigdl.util.common import JTensor
        >>> from bigdl.util.common import callBigDlFunc
        >>> np.random.seed(123)
        >>> data = np.random.uniform(0, 1, (2, 3)).astype("float32")
        >>> result = JTensor.from_ndarray(data)
        >>> data_back = result.to_ndarray()
        >>> (data == data_back).all()
        True
        >>> tensor1 = callBigDlFunc("float", "testTensor", JTensor.from_ndarray(data))  # noqa
        >>> array_from_tensor = tensor1.to_ndarray()
        >>> (array_from_tensor == data).all()
        True
        """
        if a_ndarray is None:
            return None
        assert isinstance(a_ndarray, np.ndarray), \
            "input should be a np.ndarray, not %s" % type(a_ndarray)
        return cls(a_ndarray,
                   a_ndarray.shape if a_ndarray.shape else (a_ndarray.size),
                   bigdl_type= bigdl_type)

    def to_ndarray(self):
        return np.array(self.storage, dtype=get_dtype(self.bigdl_type)).reshape(self.shape)  # noqa

    def __reduce__(self):
        return JTensor, (self.storage.tostring(), self.shape.tostring(), self.bigdl_type)

    def __str__(self):
        return "JTensor: storage: %s, shape: %s" % (str(self.storage), str(self.shape))

    def __repr__(self):
        return "JTensor: storage: %s, shape: %s" % (str(self.storage), str(self.shape))


class Sample(object):
    def __init__(self, features, label, bigdl_type="float"):
        """
        User should always use Sample.from_ndarray to construct Sample.
        :param features: a JTensor
        :param label: a JTensor
        :param bigdl_type: "double" or "float"
        """
        self.features = features
        self.label = label
        self.bigdl_type = bigdl_type

    @classmethod
    def from_ndarray(cls, features, label, bigdl_type="float"):
        """
        Convert a ndarray of features and label to Sample, which would be used in Java side.

        >>> import numpy as np
        >>> from bigdl.util.common import callBigDlFunc
        >>> from bigdl.util.common import Sample
        >>> from numpy.testing import assert_allclose
        >>> sample = Sample.from_ndarray(np.random.random((2,3)), np.random.random((2,3)))
        >>> sample_back = callBigDlFunc("float", "testSample", sample)
        >>> assert_allclose(sample.features.to_ndarray(), sample_back.features.to_ndarray())
        >>> assert_allclose(sample.label.to_ndarray(), sample_back.label.to_ndarray())
        """
        assert isinstance(features, np.ndarray), \
            "features should be a np.ndarray, not %s" % type(features)
        if not isinstance(label, np.ndarray): # in case label is a scalar.
            label = np.array(label)
        return cls(
            features=JTensor.from_ndarray(features),
            label=JTensor.from_ndarray(label),
            bigdl_type=bigdl_type)

    def __reduce__(self):
        return Sample, (self.features, self.label, self.bigdl_type)

    def __str__(self):
        return "Sample: features: %s, label: %s," % (self.features, self.label)

    def __repr__(self):
        return "Sample: features: %s, label: %s" % (self.features, self.label)

class RNG():
    """
    generate tensor data with seed
    """
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type

    def set_seed(self, seed):
        callBigDlFunc(self.bigdl_type, "setModelSeed", seed)

    def uniform(self, a, b, size):
        return callBigDlFunc(self.bigdl_type, "uniform", a, b, size).to_ndarray() # noqa


_picklable_classes = [
    'LinkedList',
    'SparseVector',
    'DenseVector',
    'DenseMatrix',
    'Rating',
    'LabeledPoint',
    'Sample',
    'TestResult',
    'JTensor'
]


def init_engine(bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initEngine")


def get_bigdl_conf():
    bigdl_conf_file = "spark-bigdl.conf"
    bigdl_python_wrapper = "python-api.zip"

    def load_conf(conf_str):
        return dict(line.split() for line in conf_str.split("\n") if
                    "#" not in line and line.strip())

    for p in sys.path:
        if bigdl_conf_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file: # noqa
                return load_conf(conf_file.read())
        if bigdl_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                content = zip_conf.read(bigdl_conf_file)
                if sys.version_info >= (3,):
                    content = str(content, 'latin-1')
                return load_conf(content)
    raise Exception("Cannot find spark-bigdl.conf.Pls add it to PYTHONPATH.")


def to_list(a):
    if type(a) is list:
        return a
    return [a]


def create_spark_conf():
    bigdl_conf = get_bigdl_conf()
    sparkConf = SparkConf()
    sparkConf.setAll(bigdl_conf.items())
    return sparkConf


def get_spark_context(conf = None):
    """
    Get the current active spark context and create one if no active instance
    :param conf: combining bigdl configs into spark conf
    :return: SparkContext
    """
    with SparkContext._lock:  # Compatible with Spark1.5.1
        if SparkContext._active_spark_context is None:
            SparkContext(conf=conf or create_spark_conf())
        return SparkContext._active_spark_context


def get_spark_sql_context(sc):
    if "getOrCreate" in SQLContext.__dict__:
        return SQLContext.getOrCreate()
    else:
        return SQLContext(sc)  # Compatible with Spark1.5.1

def callBigDlFunc(bigdl_type, name, *args):
    """ Call API in PythonBigDL """
    jinstance = JavaCreator.instance(bigdl_type=bigdl_type).value
    sc = get_spark_context()
    api = getattr(jinstance, name)
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
            return DataFrame(r, get_spark_sql_context(sc))

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
        for (key, value) in obj.items():
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
    from bigdl.nn import layer
    globs = layer.__dict__.copy()
    sc = SparkContext(master="local[2]", appName="test common utility")
    globs['sc'] = sc
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
