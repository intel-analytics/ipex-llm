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
import six
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap, MapConverter
from py4j.java_gateway import JavaGateway, GatewayClient

from pyspark import RDD, SparkContext
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame, SQLContext
from pyspark.mllib.common import callJavaFunc
from pyspark import SparkConf
from pyspark.files import SparkFiles
import numpy as np
import threading
import tempfile
import traceback
from bigdl.util.engine import get_bigdl_classpath, is_spark_below_2_2

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
                 bigdl_type, *args):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls(bigdl_type, *args)
        return cls._instance


class GatewayWrapper(SingletonMixin):

    def __init__(self, bigdl_type, port=25333):
        self.value = JavaGateway(GatewayClient(port=port), auto_convert=True)


class JavaCreator(SingletonMixin):
    __creator_class=[
        "com.intel.analytics.bigdl.python.api.PythonBigDLKeras",
        "com.intel.analytics.bigdl.python.api.PythonBigDLOnnx"
    ]

    @classmethod
    def add_creator_class(cls, jinvoker):
        with JavaCreator._lock:
            JavaCreator.__creator_class.append(jinvoker)
            JavaCreator._instance = None

    @classmethod
    def get_creator_class(cls):
        with JavaCreator._lock:
            return JavaCreator.__creator_class

    @classmethod
    def set_creator_class(cls, cclass):
        if isinstance(cclass, six.string_types):
            cclass = [cclass]
        with JavaCreator._lock:
            JavaCreator.__creator_class = cclass
            JavaCreator._instance = None

    def __init__(self, bigdl_type, gateway):
        self.value = []
        for creator_class in JavaCreator.get_creator_class():
            jclass = getattr(gateway.jvm, creator_class)
            if bigdl_type == "float":
                self.value.append(getattr(jclass, "ofFloat")())
            elif bigdl_type == "double":
                self.value.append(getattr(jclass, "ofDouble")())
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


class EvaluatedResult():
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
        return (EvaluatedResult, (self.result, self.total_num, self.method))

    def __str__(self):
        return "Evaluated result: %s, total_num: %s, method: %s" % (
            self.result, self.total_num, self.method)


def get_dtype(bigdl_type):
    # Always return float32 for now
    return "float32"


class JActivity(object):

    def __init__(self, value):
        self.value = value


class JTensor(object):
    """
    A wrapper to easy our work when need to pass or return Tensor to/from Scala.

    >>> import numpy as np
    >>> from bigdl.util.common import JTensor
    >>> np.random.seed(123)
    >>>
    """
    def __init__(self, storage, shape, bigdl_type="float", indices=None):
        """

        :param storage: values in this tensor
        :param shape: shape of this tensor
        :param bigdl_type: numeric type
        :param indices: if indices is provided, means this is a SparseTensor;
                        if not provided, means this is a DenseTensor
        """
        if isinstance(storage, bytes) and isinstance(shape, bytes):
            self.storage = np.frombuffer(storage, dtype=get_dtype(bigdl_type))
            self.shape = np.frombuffer(shape, dtype=np.int32)
        else:
            self.storage = np.array(storage, dtype=get_dtype(bigdl_type))
            self.shape = np.array(shape, dtype=np.int32)
        if indices is None:
            self.indices = None
        elif isinstance(indices, bytes):
            self.indices = np.frombuffer(indices, dtype=np.int32)
        else:
            assert isinstance(indices, np.ndarray), \
            "indices should be a np.ndarray, not %s, %s" % (type(a_ndarray), str(indices))
            self.indices = np.array(indices, dtype=np.int32)
        self.bigdl_type = bigdl_type

    @classmethod
    def from_ndarray(cls, a_ndarray, bigdl_type="float"):
        """
        Convert a ndarray to a DenseTensor which would be used in Java side.

        >>> import numpy as np
        >>> from bigdl.util.common import JTensor
        >>> from bigdl.util.common import callBigDlFunc
        >>> np.random.seed(123)
        >>> data = np.random.uniform(0, 1, (2, 3)).astype("float32")
        >>> result = JTensor.from_ndarray(data)
        >>> expected_storage = np.array([[0.69646919, 0.28613934, 0.22685145], [0.55131477, 0.71946895, 0.42310646]])
        >>> expected_shape = np.array([2, 3])
        >>> np.testing.assert_allclose(result.storage, expected_storage, rtol=1e-6, atol=1e-6)
        >>> np.testing.assert_allclose(result.shape, expected_shape)
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
                   bigdl_type)

    @classmethod
    def sparse(cls, a_ndarray, i_ndarray, shape, bigdl_type="float"):
        """
        Convert a three ndarray to SparseTensor which would be used in Java side.
        For example:
        a_ndarray = [1, 3, 2, 4]
        i_ndarray = [[0, 0, 1, 2],
                     [0, 3, 2, 1]]
        shape = [3, 4]
        Present a dense tensor
        [[ 1,  0,  0,  3],
         [ 0,  0,  2,  0],
         [ 0,  4,  0,  0]]

        :param a_ndarray non-zero elements in this SparseTensor
        :param i_ndarray zero-based indices for non-zero element
                         i_ndarray's shape should be (shape.size, a_ndarray.size)
                         And the i-th non-zero elements indices is i_ndarray[:, 1],
                         should be zero-based and ascending;
        :param shape     shape as a DenseTensor.

        >>> import numpy as np
        >>> from bigdl.util.common import JTensor
        >>> from bigdl.util.common import callBigDlFunc
        >>> np.random.seed(123)
        >>> data = np.arange(1, 7).astype("float32")
        >>> indices = np.arange(1, 7)
        >>> shape = np.array([10])
        >>> result = JTensor.sparse(data, indices, shape)
        >>> expected_storage = np.array([1., 2., 3., 4., 5., 6.])
        >>> expected_shape = np.array([10])
        >>> expected_indices = np.array([1, 2, 3, 4, 5, 6])
        >>> np.testing.assert_allclose(result.storage, expected_storage)
        >>> np.testing.assert_allclose(result.shape, expected_shape)
        >>> np.testing.assert_allclose(result.indices, expected_indices)
        >>> tensor1 = callBigDlFunc("float", "testTensor", result)  # noqa
        >>> array_from_tensor = tensor1.to_ndarray()
        >>> expected_ndarray = np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
        >>> (array_from_tensor == expected_ndarray).all()
        True
        """
        if a_ndarray is None:
            return None
        assert isinstance(a_ndarray, np.ndarray), \
            "values array should be a np.ndarray, not %s" % type(a_ndarray)
        assert isinstance(i_ndarray, np.ndarray), \
            "indices array should be a np.ndarray, not %s" % type(a_ndarray)
        assert i_ndarray.size == a_ndarray.size * shape.size, \
            "size of values and indices should match."
        return cls(a_ndarray,
                   shape,
                   bigdl_type,
                   i_ndarray)

    def to_ndarray(self):
        """
        Transfer JTensor to ndarray.
        As SparseTensor may generate an very big ndarray, so we don't support this function for SparseTensor.
        :return: a ndarray
        """
        assert self.indices is None, "sparseTensor to ndarray is not supported"
        return np.array(self.storage, dtype=get_dtype(self.bigdl_type)).reshape(self.shape)  # noqa

    def __reduce__(self):
        if self.indices is None:
            return JTensor, (self.storage.tostring(), self.shape.tostring(), self.bigdl_type)
        else:
            return JTensor, (self.storage.tostring(), self.shape.tostring(), self.bigdl_type, self.indices.tostring())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        indices = "" if self.indices is None else " ,indices %s" % str(self.indices)
        return "JTensor: storage: %s, shape: %s%s, %s" % (str(self.storage), str(self.shape), indices, self.bigdl_type)


class Sample(object):
    def __init__(self, features, labels, bigdl_type="float"):
        """
        User should always use Sample.from_ndarray to construct Sample.
        :param features: a list of JTensors
        :param labels: a list of JTensors
        :param bigdl_type: "double" or "float"
        """
        self.feature = features[0]
        self.features = features
        self.label = labels[0]
        self.bigdl_type = bigdl_type
        self.labels = labels

    @classmethod
    def from_ndarray(cls, features, labels, bigdl_type="float"):
        """
        Convert a ndarray of features and labels to Sample, which would be used in Java side.
        :param features: an ndarray or a list of ndarrays
        :param labels: an ndarray or a list of ndarrays or a scalar
        :param bigdl_type: "double" or "float"

        >>> import numpy as np
        >>> from bigdl.util.common import callBigDlFunc
        >>> from numpy.testing import assert_allclose
        >>> np.random.seed(123)
        >>> sample = Sample.from_ndarray(np.random.random((2,3)), np.random.random((2,3)))
        >>> sample_back = callBigDlFunc("float", "testSample", sample)
        >>> assert_allclose(sample.features[0].to_ndarray(), sample_back.features[0].to_ndarray())
        >>> assert_allclose(sample.label.to_ndarray(), sample_back.label.to_ndarray())
        >>> expected_feature_storage = np.array(([[0.69646919, 0.28613934, 0.22685145], [0.55131477, 0.71946895, 0.42310646]]))
        >>> expected_feature_shape = np.array([2, 3])
        >>> expected_label_storage = np.array(([[0.98076421, 0.68482971, 0.48093191], [0.39211753, 0.343178, 0.72904968]]))
        >>> expected_label_shape = np.array([2, 3])
        >>> assert_allclose(sample.features[0].storage, expected_feature_storage, rtol=1e-6, atol=1e-6)
        >>> assert_allclose(sample.features[0].shape, expected_feature_shape)
        >>> assert_allclose(sample.labels[0].storage, expected_label_storage, rtol=1e-6, atol=1e-6)
        >>> assert_allclose(sample.labels[0].shape, expected_label_shape)
        """
        if isinstance(features, np.ndarray):
            features = [features]
        else:
            assert all(isinstance(feature, np.ndarray) for feature in features), \
                "features should be a list of np.ndarray, not %s" % type(features)
        if np.isscalar(labels):  # in case labels is a scalar.
            labels = [np.array(labels)]
        elif isinstance(labels, np.ndarray):
            labels = [labels]
        else:
            assert all(isinstance(label, np.ndarray) for label in labels), \
                "labels should be a list of np.ndarray, not %s" % type(labels)
        return cls(
            features=[JTensor.from_ndarray(feature) for feature in features],
            labels=[JTensor.from_ndarray(label) for label in labels],
            bigdl_type=bigdl_type)

    @classmethod
    def from_jtensor(cls, features, labels, bigdl_type="float"):
        """
        Convert a sequence of JTensor to Sample, which would be used in Java side.
        :param features: an JTensor or a list of JTensor
        :param labels: an JTensor or a list of JTensor or a scalar
        :param bigdl_type: "double" or "float"

        >>> import numpy as np
        >>> data = np.random.uniform(0, 1, (6)).astype("float32")
        >>> indices = np.arange(1, 7)
        >>> shape = np.array([10])
        >>> feature0 = JTensor.sparse(data, indices, shape)
        >>> feature1 = JTensor.from_ndarray(np.random.uniform(0, 1, (2, 3)).astype("float32"))
        >>> sample = Sample.from_jtensor([feature0, feature1], 1)
        """
        if isinstance(features, JTensor):
            features = [features]
        else:
            assert all(isinstance(feature, JTensor) for feature in features), \
                "features should be a list of JTensor, not %s" % type(features)
        if np.isscalar(labels):  # in case labels is a scalar.
            labels = [JTensor.from_ndarray(np.array(labels))]
        elif isinstance(labels, JTensor):
            labels = [labels]
        else:
            assert all(isinstance(label, JTensor) for label in labels), \
                "labels should be a list of np.ndarray, not %s" % type(labels)
        return cls(
            features=features,
            labels=labels,
            bigdl_type=bigdl_type)

    def __reduce__(self):
        return Sample, (self.features, self.labels, self.bigdl_type)

    def __str__(self):
        return "Sample: features: %s, labels: %s," % (self.features, self.labels)

    def __repr__(self):
        return "Sample: features: %s, labels: %s" % (self.features, self.labels)

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
    'EvaluatedResult',
    'JTensor',
    'JActivity'
]


def init_engine(bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initEngine")
    # Spark context is supposed to have been created when init_engine is called
    get_spark_context()._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.initialize()

def get_bigdl_engine_type(bigdl_type="float"):
    return callBigDlFunc(bigdl_type, "getEngineType")

def set_optimizer_version(optimizerVersion, bigdl_type="float"):
    return callBigDlFunc(bigdl_type, "setOptimizerVersion", optimizerVersion)

def get_optimizer_version(bigdl_type="float"):
    return callBigDlFunc(bigdl_type, "getOptimizerVersion")

def init_executor_gateway(sc, bigdl_type="float"):
    callBigDlFunc(bigdl_type, "initExecutorGateway", sc, sc._gateway._gateway_client.port)


def get_node_and_core_number(bigdl_type="float"):
    result = callBigDlFunc(bigdl_type, "getNodeAndCoreNumber")
    return result[0], result[1]


def redire_spark_logs(bigdl_type="float", log_path=os.getcwd()+"/bigdl.log"):
    """
    Redirect spark logs to the specified path.
    :param bigdl_type: "double" or "float"
    :param log_path: the file path to be redirected to; the default file is under the current workspace named `bigdl.log`.
    """
    callBigDlFunc(bigdl_type, "redirectSparkLogs", log_path)


def show_bigdl_info_logs(bigdl_type="float"):
    """
    Set BigDL log level to INFO.
    :param bigdl_type: "double" or "float"
    """
    callBigDlFunc(bigdl_type, "showBigDlInfoLogs")


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
                if bigdl_conf_file  in zip_conf.namelist():
                    content = zip_conf.read(bigdl_conf_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content)
    return {}


def to_list(a):
    if type(a) is list:
        return a
    return [a]


def to_sample_rdd(x, y, numSlices=None):
    """
    Conver x and y into RDD[Sample]
    :param x: ndarray and the first dimension should be batch
    :param y: ndarray and the first dimension should be batch
    :param numSlices:
    :return:
    """
    sc = get_spark_context()
    from bigdl.util.common import Sample
    x_rdd = sc.parallelize(x, numSlices)
    y_rdd = sc.parallelize(y, numSlices)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


def extend_spark_driver_cp(sparkConf, path):
    original_driver_classpath = ":" + sparkConf.get("spark.driver.extraClassPath") \
        if sparkConf.contains("spark.driver.extraClassPath") else ""
    sparkConf.set("spark.driver.extraClassPath", path + original_driver_classpath)


def create_spark_conf():
    bigdl_conf = get_bigdl_conf()
    sparkConf = SparkConf()
    sparkConf.setAll(bigdl_conf.items())
    if os.environ.get("BIGDL_JARS", None) and not is_spark_below_2_2():
        for jar in os.environ["BIGDL_JARS"].split(":"):
            extend_spark_driver_cp(sparkConf, jar)

    # add content in PYSPARK_FILES in spark.submit.pyFiles
    # This is a workaround for current Spark on k8s
    python_lib = os.environ.get('PYSPARK_FILES', None)
    if python_lib:
        existing_py_files = sparkConf.get("spark.submit.pyFiles")
        if existing_py_files:
            sparkConf.set(key="spark.submit.pyFiles", value="%s,%s" % (python_lib, existing_py_files))
        else:
            sparkConf.set(key="spark.submit.pyFiles", value=python_lib)

    return sparkConf


def get_spark_context(conf=None):
    """
    Get the current active spark context and create one if no active instance
    :param conf: combining bigdl configs into spark conf
    :return: SparkContext
    """
    if hasattr(SparkContext, "getOrCreate"):
        with SparkContext._lock:
            if SparkContext._active_spark_context is None:
                spark_conf = create_spark_conf() if conf is None else conf
                return SparkContext.getOrCreate(spark_conf)
            else:
                return SparkContext.getOrCreate()

    else:
        # Might have threading issue but we cann't add _lock here
        # as it's not RLock in spark1.5;
        if SparkContext._active_spark_context is None:
            spark_conf = create_spark_conf() if conf is None else conf
            return SparkContext(conf=spark_conf)
        else:
            return SparkContext._active_spark_context


def get_spark_sql_context(sc):
    if "getOrCreate" in SQLContext.__dict__:
        return SQLContext.getOrCreate(sc)
    else:
        return SQLContext(sc)  # Compatible with Spark1.5.1


def _get_port():
    root_dir = SparkFiles.getRootDirectory()
    path = os.path.join(root_dir, "gateway_port")
    try:
        with open(path) as f:
            port = int(f.readline())
    except IOError as e:
        traceback.print_exc()
        raise RuntimeError("Could not open the file %s, which contains the listening port of"
                           " local Java Gateway, please make sure the init_executor_gateway()"
                           " function is called before any call of java function on the"
                           " executor side." % e.filename)
    return port


def _get_gateway():
    if SparkFiles._is_running_on_worker:
        gateway_port = _get_port()
        gateway = GatewayWrapper.instance(None, gateway_port).value
    else:
        sc = get_spark_context()
        gateway = sc._gateway
    return gateway


def callBigDlFunc(bigdl_type, name, *args):
    """ Call API in PythonBigDL """
    gateway = _get_gateway()
    args = [_py2java(gateway, a) for a in args]
    error = Exception("Cannot find function: %s" % name)
    for jinvoker in JavaCreator.instance(bigdl_type, gateway).value:
        # hasattr(jinvoker, name) always return true here,
        # so you need to invoke the method to check if it exist or not
        try:
            api = getattr(jinvoker, name)
            result = callJavaFunc(api, *args)
        except Exception as e:
            error = e
            if "does not exist" not in str(e):
                raise e
        else:
            return result
    raise error


def _java2py(gateway, r, encoding="bytes"):
    if isinstance(r, JavaObject):
        clsName = r.getClass().getSimpleName()
        # convert RDD into JavaRDD
        if clsName != 'JavaRDD' and clsName.endswith("RDD"):
            r = r.toJavaRDD()
            clsName = 'JavaRDD'

        if clsName == 'JavaRDD':
            jrdd = gateway.jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.javaToPython(r)
            return RDD(jrdd, get_spark_context())

        if clsName == 'DataFrame':
            return DataFrame(r, get_spark_sql_context(get_spark_context()))

        if clsName == 'Dataset':
            return DataFrame(r, get_spark_sql_context(get_spark_context()))

        if clsName == "ImageFrame[]":
            return r

        if clsName in _picklable_classes:
            r = gateway.jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.dumps(r)
        elif isinstance(r, (JavaArray, JavaList, JavaMap)):
            try:
                r = gateway.jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.dumps(
                    r)
            except Py4JJavaError:
                pass  # not pickable

    if isinstance(r, (bytearray, bytes)):
        r = PickleSerializer().loads(bytes(r), encoding=encoding)
    return r


def callJavaFunc(func, *args):
    """ Call Java Function """
    gateway = _get_gateway()
    result = func(*args)
    return _java2py(gateway, result)


def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling


    It will convert each Python object into Java object by Pyrolite, whenever
    the RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return \
        rdd.ctx._jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.pythonToJava(
            rdd._jrdd, True)


def _py2java(gateway, obj):
    """ Convert Python object into Java """
    if isinstance(obj, RDD):
        obj = _to_java_object_rdd(obj)
    elif isinstance(obj, DataFrame):
        obj = obj._jdf
    elif isinstance(obj, SparkContext):
        obj = obj._jsc
    elif isinstance(obj, (list, tuple)):
        obj = ListConverter().convert([_py2java(gateway, x) for x in obj],
                                      gateway._gateway_client)
    elif isinstance(obj, dict):
        result = {}
        for (key, value) in obj.items():
            result[key] = _py2java(gateway, value)
        obj = MapConverter().convert(result, gateway._gateway_client)
    elif isinstance(obj, JavaValue):
        obj = obj.value
    elif isinstance(obj, JavaObject):
        pass
    elif isinstance(obj, (int, long, float, bool, bytes, unicode)):
        pass
    else:
        data = bytearray(PickleSerializer().dumps(obj))
        obj = gateway.jvm.org.apache.spark.bigdl.api.python.BigDLSerDe.loads(data)
    return obj


def create_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="bigdl")
    tmp_file.close()
    return tmp_file.name


def text_from_path(path):
    sc = get_spark_context()
    return sc.textFile(path).collect()[0]


def get_local_file(a_path):
    if not is_distributed(a_path):
        return a_path
    path, data = get_spark_context().binaryFiles(a_path).collect()[0]
    local_file_path = create_tmp_path()
    with open(local_file_path, 'w') as local_file:
        local_file.write(data)
    return local_file_path


def is_distributed(path):
    return "://" in path


def get_activation_by_name(activation_name, activation_id=None):
    """ Convert to a bigdl activation layer
        given the name of the activation as a string  """
    import bigdl.nn.layer as BLayer
    activation = None
    activation_name = activation_name.lower()
    if activation_name == "tanh":
        activation = BLayer.Tanh()
    elif activation_name == "sigmoid":
        activation = BLayer.Sigmoid()
    elif activation_name == "hard_sigmoid":
        activation = BLayer.HardSigmoid()
    elif activation_name == "relu":
        activation = BLayer.ReLU()
    elif activation_name == "softmax":
        activation = BLayer.SoftMax()
    elif activation_name == "softplus":
        activation = BLayer.SoftPlus(beta=1.0)
    elif activation_name == "softsign":
        activation = BLayer.SoftSign()
    elif activation_name == "linear":
        activation = BLayer.Identity()
    else:
        raise Exception("Unsupported activation type: %s" % activation_name)
    if not activation_id:
        activation.set_name(activation_id)
    return activation


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
