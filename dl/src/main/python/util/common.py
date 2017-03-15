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
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap

from pyspark import RDD, SparkContext
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame, SQLContext
from pyspark.mllib.common import callJavaFunc
from pyspark import SparkConf

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
    def __init__(self, result, count, method):
        self.result = result
        self.count = count
        self.method = method

    def __reduce__(self):
        return (TestResult, (self.result, self.count, self.method))

    def __str__(self):
        return "result: %s, count: %s, method: %s" % (
            self.result, self.count, self.method)


class Sample(object):
    def __init__(self, features, label, features_shape, label_shape,
                 bigdl_type="float"):
        self.features = features
        self.label = label
        self.features_shape = features_shape
        self.label_shape = label_shape
        self.bigdl_type = bigdl_type

    # features is a ndarray
    # label is a ndarray
    @classmethod
    def from_ndarray(cls, features, label, bigdl_type="float"):
        return cls(
            features=[float(i) for i in features.ravel()],
            label=[float(i) for i in label.ravel()],
            features_shape=list(features.shape),
            label_shape=list(label.shape) if label.shape else [label.size],
            bigdl_type=bigdl_type)

    @classmethod
    def of(cls, features, label, features_shape, bigdl_type="float"):
        return cls(
            features=[float(i) for i in features],
            label=[float(label)],
            features_shape=features_shape,
            label_shape=[1],
            bigdl_type=bigdl_type)

    def __reduce__(self):
        return (Sample, (
            self.features, self.label, self.features_shape, self.label_shape,
            self.bigdl_type))

    def __str__(self):
        return "features: %s, label: %s," \
               "features_shape: %s, label_shape: %s, bigdl_type: %s" % (
                   self.features, self.label, self.features_shape,
                   self.label_shape,
                   self.bigdl_type)


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


def initEngine(nodeNum, coreNum, bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initEngine", nodeNum, coreNum)


def create_spark_conf(coreNum, nodeNum):
    print("coreNum:%s,  nodeNum: %s" % (coreNum, nodeNum))
    sparkConf = SparkConf()
    sparkConf.setExecutorEnv("DL_ENGINE_TYPE", "mklblas")
    sparkConf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    sparkConf.setExecutorEnv("KMP_BLOCKTIME", "0")
    sparkConf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    sparkConf.setExecutorEnv("OMP_NUM_THREADS", "1")
    sparkConf.setExecutorEnv("DL_CORE_NUMBER", str(coreNum))
    sparkConf.setExecutorEnv("DL_NODE_NUMBER", str(nodeNum))
    sparkConf.set("spark.shuffle.blockTransferService", "nio")
    sparkConf.set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    return sparkConf


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
        obj = {key: _py2java(sc, jv) for (key, jv) in obj.iteritems() if
               isinstance(jv, JavaValue)}
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
