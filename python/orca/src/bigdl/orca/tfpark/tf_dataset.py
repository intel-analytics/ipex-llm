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

import numpy as np
import tensorflow as tf
import sys
import functools
import logging

from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from bigdl.dllib.feature.transform.vision.image import FeatureTransformer
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.file_utils import Sample, JTensor
from bigdl.dllib.nncontext import getOrCreateSparkContext
from bigdl.dllib.feature.common import FeatureSet, SampleToMiniBatch, Preprocessing
from bigdl.dllib.feature.image import ImagePreprocessing, ImageFeatureToSample
from bigdl.dllib.utils import nest
from bigdl.dllib.utils.utils import convert_row_to_numpy

if sys.version >= '3':
    long = int
    unicode = str


def _to_tensor_structure(tensors):
    if isinstance(tensors, tuple):
        tensor_structure = TensorMeta(dtype=tensors[0], shape=tensors[1], name="input0")
    elif isinstance(tensors, list):
        tensor_structure = [TensorMeta(dtype=value[0], shape=value[1],
                                       name="list_input_" + str(idx))
                            for (idx, value) in enumerate(tensors)]
    elif isinstance(tensors, dict):
        tensor_structure = {}
        for key, value in tensors.items():
            tensor_structure[key] = TensorMeta(dtype=value[0], shape=value[1], name=key)
    else:
        raise ValueError("In TFDataset.from_rdd, features and labels should be a tuple, "
                         "a list of tuples or a dict of tuples")
    return tensor_structure


def _tensors_to_rdd(tensors, sc, splits):
    import tensorflow as tf

    if isinstance(tensors, np.ndarray):
        tensors = (tensors,)

    if isinstance(tensors, list):
        for i in range(len(tensors)):
            if tensors[i].dtype == np.dtype("float64"):
                tensors[i] = np.float32(tensors[i])

        data_list = _splits(tensors)
        rdd = sc.parallelize(data_list, splits)
        tensor_structure = [TensorMeta(tf.as_dtype(t.dtype),
                                       shape=t.shape[1:],
                                       name="input_%s" % i)
                            for i, t in enumerate(tensors)]
    else:
        flattened = nest.flatten(tensors)
        for i in range(len(flattened)):
            if flattened[i].dtype == np.dtype("float64"):
                flattened[i] = np.float32(flattened[i])
        data_list = _splits(flattened)
        rdd = sc.parallelize(data_list, splits)
        rdd = rdd.map(lambda x: nest.pack_sequence_as(tensors, x))
        tensor_structure = nest.pack_sequence_as(tensors,
                                                 [TensorMeta(tf.as_dtype(t.dtype),
                                                             shape=t.shape[1:],
                                                             name="input_%s" % i)
                                                  for i, t in enumerate(flattened)])
    return rdd, tensor_structure


def _splits(tensors):
    data_list = []
    data_size = tensors[0].shape[0]
    for i in range(data_size):
        sample = []
        for j in range(len(tensors)):
            sample.append(tensors[j][i])
        data_list.append(sample)
    return data_list


class MergeFeatureLabelImagePreprocessing(ImagePreprocessing):
    def __init__(self, bigdl_type="float"):
        super(MergeFeatureLabelImagePreprocessing, self).__init__(bigdl_type)


class MergeFeatureLabelFeatureTransformer(FeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(MergeFeatureLabelFeatureTransformer, self).__init__(bigdl_type)


class TensorMeta(object):
    def __init__(self, dtype, name=None, shape=None):
        self.dtype = dtype
        self.name = name
        self.shape = shape

    def __repr__(self):
        return "TensorMeta(dtype: " + self.dtype.name + ", name: " + self.name + \
               ", shape: " + str(self.shape) + ")"


class TFDataset(object):
    def __init__(self, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False):
        """

        TFDataset represents a distributed collection of elements (backed by a RDD)
        to be feed into Tensorflow graph.

        :param tensor_structure: a nested structure of TensorMeta objects specifying the
        name, shape and data type of each element in this TFDataset
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        """

        if batch_size > 0 and batch_per_thread > 0:
            raise ValueError("bath_size and batch_per_thread should not be set simultaneously")

        self.has_batch = True
        node_num, core_num = get_node_and_core_number()
        self.total_core_num = node_num * core_num
        self.node_num = node_num
        self.core_num = core_num
        if batch_size > 0:
            if batch_size % self.total_core_num != 0:
                raise ValueError("batch_size should be a multiple " +
                                 "of total core number, but got batch_size: " +
                                 "%s where total core number is %s" % (batch_size,
                                                                       self.total_core_num))
        if batch_size <= 0 and batch_per_thread <= 0:
            batch_per_thread = 1
            batch_size = self.total_core_num
            self.has_batch = False

        self.batch_size = batch_size
        self.batch_per_thread = batch_per_thread
        self.hard_code_batch_size = hard_code_batch_size
        self.tensor_structure = tensor_structure

        if not self.hard_code_batch_size:
            self.output_shapes = nest.pack_sequence_as(
                self.tensor_structure, [[None] + list(t.shape)
                                        if t is not None else None
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_per_thread] + t.shape
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])
            else:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_size // self.total_core_num] + list(t.shape)
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])

        self.input_names = nest.pack_sequence_as(
            self.tensor_structure, [t.name
                                    if t is not None else None
                                    for t in nest.flatten(self.tensor_structure)])

        self._tensors = None

    def _create_placeholders(self):

        import tensorflow as tf

        if not self.hard_code_batch_size:
            tensors = nest.pack_sequence_as(
                self.tensor_structure, [tf.placeholder(name=t.name,
                                                       dtype=t.dtype,
                                                       shape=[None] + list(t.shape))
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_per_thread] + list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])
            else:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_size // self.total_core_num] + list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])

        for tensor in nest.flatten(tensors):
            tf.get_default_graph().clear_collection(tensor.name)
            tf.add_to_collection(tensor.name, self)

        self._original_tensors = tensors
        self._tensors = tensors

        if not self.has_batch:
            self._tensors = nest.pack_sequence_as(self.tensor_structure,
                                                  [t[0] for t in nest.flatten(tensors)])

        return tensors

    @property
    def tensors(self):
        """
        a nested structure of TensorFlow tensor object in TensorFlow graph.
        The elements of this dataset will be fed into these tensors on each iteration.
        :return: the nested structure of TensorFlow tensor object
        """

        if self._tensors is None:
            self._create_placeholders()

        return self._tensors

    @property
    def feature_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use feature_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[0]

    @property
    def label_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use label_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[1]

    @staticmethod
    def _to_tensor_structure(features, labels):
        feature_structure = _to_tensor_structure(features)
        if labels is not None:
            label_structure = _to_tensor_structure(labels)
            tensor_structure = (feature_structure, label_structure)

        else:
            tensor_structure = (feature_structure,)
        return tensor_structure

    def get_prediction_data(self):
        """
        :return: an object that can be used for TFNet.predict
        e.g. an RDD of Sample or a ImageSet
        """
        assert self.batch_per_thread > 0, "batch_per_thread must be set when used in prediction"
        return self._get_prediction_data()

    def get_evaluation_data(self):
        """
        :return: an object that can be used for TFNet.evaluate,
        e.g. an RDD of Sample or a ImageSet
        """
        assert self.batch_per_thread > 0, "batch_per_thread must be set when used in evaluation"
        return self._get_evaluation_data()

    def get_training_data(self):
        """
        :return: an object that can be used to create a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        assert self.batch_size > 0, "batch_size must be set when used in training"
        return self._get_training_data()

    def get_validation_data(self):
        """
        :return: an object that can be used to set validation in a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        assert self.batch_size > 0, "batch_size must be set when used in training"
        return self._get_validation_data()

    def _get_prediction_data(self):
        raise NotImplementedError

    def _get_evaluation_data(self):
        raise NotImplementedError

    def _get_training_data(self):
        raise NotImplementedError

    def _get_validation_data(self):
        raise NotImplementedError

    def get_num_partitions(self):
        """
        :return: the num of partitions of the underlying RDD
        """
        raise NotImplementedError

    @staticmethod
    def from_rdd(*args, **kwargs):
        """
        Create a TFDataset from a rdd.

        For training and evaluation, both `features` and `labels` arguments should be specified.
        The element of the rdd should be a tuple of two, (features, labels), each has the
        same structure of numpy.ndarrays of the argument `features`, `labels`.

        E.g. if `features` is [(tf.float32, [10]), (tf.float32, [20])],
        and `labels` is {"label1":(tf.float32, [10]), "label2": (tf.float32, [20])}
        then a valid element of the rdd could be

        (
        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))],
         {"label1": np.zeros(dtype=float, shape=(10,)),
          "label2":np.zeros(dtype=float, shape=(10,))))}
        )

        If `labels` is not specified,
        then the above element should be changed to
        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))]

        For inference, `labels` can be not specified.
        The element of the rdd should be some ndarrays of the same structure of the `features`
        argument.

        A note on the legacy api: if you are using `names`, `shapes`, `types` arguments,
        each element of the rdd should be a list of numpy.ndarray.

        :param rdd: a rdd containing the numpy.ndarrays to be used
        for training/evaluation/inference
        :param features: the structure of input features, should one the following:
               - a tuple (dtype, shape), e.g. (tf.float32, [28, 28, 1])
               - a list of such tuple [(dtype1, shape1), (dtype2, shape2)],
                     e.g. [(tf.float32, [10]), (tf.float32, [20])],
               - a dict of such tuple, mapping string names to tuple {"name": (dtype, shape},
                     e.g. {"input1":(tf.float32, [10]), "input2": (tf.float32, [20])}

        :param labels: the structure of input labels, format is the same as features
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param val_rdd: validation data with the same structure of rdd
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        return TFNdarrayDataset.from_rdd(*args, **kwargs)

    @staticmethod
    def from_ndarrays(*args, **kwargs):
        """
        Create a TFDataset from a nested structure of numpy ndarrays. Each element
        in the resulting TFDataset has the same structure of the argument tensors and
        is created by indexing on the first dimension of each ndarray in the tensors
        argument.

        This method is equivalent to sc.parallize the tensors and call TFDataset.from_rdd

        :param tensors: the numpy ndarrays
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param val_tensors: the numpy ndarrays used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        return TFNdarrayDataset.from_ndarrays(*args, **kwargs)

    @staticmethod
    def from_image_set(image_set, image, label=None,
                       batch_size=-1, batch_per_thread=-1,
                       hard_code_batch_size=False,
                       validation_image_set=None,
                       memory_type='DRAM',
                       sequential_order=False,
                       shuffle=True):
        """
        Create a TFDataset from a ImagetSet. Each ImageFeature in the ImageSet should
        already has the "sample" field, i.e. the result of ImageSetToSample transformer

        :param image_set: the ImageSet used to create this TFDataset
        :param image: a tuple of two, the first element is the type of image, the second element
        is the shape of this element, i.e. (tf.float32, [224, 224, 3]))
        :param label: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1]))
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_image_set: the ImageSet used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(image, label)
        return TFImageDataset(image_set, tensor_structure, batch_size,
                              batch_per_thread, hard_code_batch_size,
                              validation_image_set,
                              memory_type=memory_type,
                              sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_text_set(text_set, text, label=None,
                      batch_size=-1, batch_per_thread=-1,
                      hard_code_batch_size=False, validation_image_set=None,
                      sequential_order=False, shuffle=True):
        """
        Create a TFDataset from a TextSet. The TextSet must be transformed to Sample, i.e.
        the result of TextFeatureToSample transformer.
        :param text_set: the TextSet used to create this TFDataset
        :param text: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [10, 100, 4])).
        text can also be nested structure of this tuple of two.
        :param label: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_image_set: The TextSet used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(text, label)
        return TFTextDataset(text_set, tensor_structure, batch_size,
                             batch_per_thread, hard_code_batch_size,
                             validation_image_set,
                             sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_tfrecord_file(sc, file_path, batch_size=-1, batch_per_thread=-1,
                           hard_code_batch_size=False, validation_file_path=None,
                           sequential_order=False, shuffle=True):
        """
        Create a TFDataset from tfrecord files.
        :param sc: The SparkContext
        :param file_path: comma seperated tfrecord file(s) path
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_file_path: The tfrecord files used for validation
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        input_format_class = "org.tensorflow.hadoop.io.TFRecordFileInputFormat"
        key_class = "org.apache.hadoop.io.BytesWritable"
        value_class = "org.apache.hadoop.io.NullWritable"
        bytes_rdd = sc.newAPIHadoopFile(file_path, input_format_class,
                                        keyClass=key_class,
                                        valueClass=value_class)
        bytes_rdd = bytes_rdd.map(lambda record: bytearray(record[0]))
        validation_bytes_rdd = None
        if validation_file_path is not None:
            validation_bytes_rdd = sc.newAPIHadoopFile(validation_file_path,
                                                       input_format_class,
                                                       keyClass=key_class,
                                                       valueClass=value_class)
            validation_bytes_rdd = validation_bytes_rdd.map(lambda record: bytearray(record[0]))

        return TFBytesDataset(bytes_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_bytes_rdd,
                              sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_feature_set(dataset, features, labels=None, batch_size=-1, batch_per_thread=-1,
                         hard_code_batch_size=False, validation_dataset=None):
        """
        Create a TFDataset from a FeatureSet. Currently, the element in this Feature set must be a
        Sample, i.e. the result of ImageFeatureToSample transformer
        :param dataset: the feature set used to create this TFDataset
        :param features: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [224, 224, 3])).
        text can also be nested structure of this tuple of two.
        :param labels: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_dataset: The FeatureSet used for validation during training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(features, labels)

        return TFFeatureDataset(dataset, tensor_structure, batch_size,
                                batch_per_thread, hard_code_batch_size,
                                validation_dataset)

    @staticmethod
    def from_string_rdd(string_rdd, batch_size=-1, batch_per_thread=-1,
                        hard_code_batch_size=False, validation_string_rdd=None):
        """
        Create a TFDataset from a RDD of strings. Each element is the RDD should be a single string.
        The returning TFDataset's feature_tensors has only one Tensor. the type of the Tensor
        is tf.string, and the shape is (None,). The returning don't have label_tensors. If the
        dataset is used for training, the label should be encoded in the string.
        :param string_rdd: the RDD of strings
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_string_rdd: the RDD of strings to be used in validation
        :return: a TFDataset
        """
        string_rdd = string_rdd.map(lambda x: bytearray(x, "utf-8"))
        if validation_string_rdd is not None:
            validation_string_rdd = validation_string_rdd.map(lambda x: bytearray(x, "utf-8"))
        return TFBytesDataset(string_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_string_rdd)

    @staticmethod
    def from_bytes_rdd(bytes_rdd, batch_size=-1, batch_per_thread=-1,
                       hard_code_batch_size=False, validation_bytes_rdd=None):
        """
        Create a TFDataset from a RDD of bytes. Each element is the RDD should be a bytes object.
        The returning TFDataset's feature_tensors has only one Tensor. the type of the Tensor
        is tf.string, and the shape is (None,). The returning don't have label_tensors. If the
        dataset is used for training, the label should be encoded in the bytes.
        :param bytes_rdd: the RDD of bytes
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_bytes_rdd: the RDD of bytes to be used in validation
        :return: a TFDataset
        """
        return TFBytesDataset(bytes_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_bytes_rdd)

    @staticmethod
    def from_tf_data_dataset(dataset, batch_size=-1,
                             batch_per_thread=-1, hard_code_batch_size=False,
                             validation_dataset=None,
                             sequential_order=False,
                             shuffle=True,
                             remove_checking=False, batch_outside=False,
                             inter_threads=None, intra_threads=None, auto_shard_files=False):
        """
        Create a TFDataset from a tf.data.Dataset.

        The recommended way to create the dataset is to reading files in a shared file
        system (e.g. HDFS) that is accessible from every executor of this Spark Application.

        If the dataset is created by reading files in the local file system, then the
        files must exist in every executor in the exact same path. The path should be
        absolute path and relative path is not supported.

        A few kinds of dataset is not supported for now:
        1. dataset created from tf.data.Dataset.from_generators
        2. dataset with Dataset.batch operation.
        3. dataset with Dataset.repeat operation
        4. dataset contains tf.py_func, tf.py_function or tf.numpy_function

        :param dataset: the tf.data.Dataset
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_dataset: the dataset used for validation
        :return: a TFDataset
        """
        return TFDataDataset(dataset, batch_size, batch_per_thread,
                             hard_code_batch_size, validation_dataset,
                             sequential_order, shuffle, remove_checking, batch_outside,
                             inter_threads, intra_threads, auto_shard_files=auto_shard_files)

    @staticmethod
    def from_dataframe(df, feature_cols, labels_cols=None, batch_size=-1,
                       batch_per_thread=-1, hard_code_batch_size=False,
                       validation_df=None, memory_type="DRAM",
                       sequential_order=False, shuffle=True):
        """
        Create a TFDataset from a pyspark.sql.DataFrame.

        :param df: the DataFrame for the dataset
        :param feature_cols: a list of string, indicating which columns are used as features.
                            Currently supported types are FloatType, DoubleType, IntegerType,
                            LongType, ArrayType (value should be numbers), DenseVector
                            and SparseVector. For ArrayType, DenseVector and SparseVector,
                            the sizes are assume to the same.
        :param labels_cols: a list of string, indicating which columns are used as labels.
                            Currently supported types are FloatType, DoubleType, IntegerType,
                            LongType, ArrayType (value should be numbers), DenseVector
                            and SparseVector. For ArrayType, DenseVector and SparseVector,
                            the sizes are assume to the same.
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_df: the DataFrame used for validation
        :return: a TFDataset
        """
        return DataFrameDataset(df, feature_cols, labels_cols, batch_size,
                                batch_per_thread, hard_code_batch_size, validation_df,
                                memory_type, sequential_order, shuffle)


def _tf_get_types(dataset):
    import tensorflow as tf
    return tf.compat.v1.data.get_output_types(dataset)


def _tf_get_shapes(dataset):
    import tensorflow as tf
    return tf.compat.v1.data.get_output_shapes(dataset)


def _tf_make_iterator(dataset):
    import tensorflow as tf
    return tf.compat.v1.data.make_initializable_iterator(dataset)


class TFDataDataset(TFDataset):
    def get_num_partitions(self):
        # only called in inference case
        return self.total_core_num

    @staticmethod
    def _assert_not_batched(dataset):
        from tensorflow.python.data.ops import dataset_ops

        if isinstance(dataset, dataset_ops.DatasetV1Adapter):
            TFDataDataset._assert_not_batched(dataset._dataset)
        elif isinstance(dataset, dataset_ops.BatchDataset):
            raise ValueError("Dataset should not be batched,"
                             "please use a dataset without the batch operation")
        else:
            for dt in dataset._inputs():
                TFDataDataset._assert_not_batched(dt)

    @staticmethod
    def check_rules(dataset, rules, is_training):
        from tensorflow.python.data.ops import dataset_ops

        if isinstance(dataset, dataset_ops.DatasetV1Adapter):
            TFDataDataset.check_rules(dataset._dataset, rules, is_training)
        else:
            for rule, message in rules:
                assert not rule(dataset, is_training), message
            else:
                for dt in dataset._inputs():
                    TFDataDataset.check_rules(dt, rules, is_training)

    def __init__(self, tf_data_dataset, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_dataset=None,
                 sequential_order=False, shuffle=True,
                 remove_checking=False, batch_outside=False,
                 inter_threads=None, intra_threads=None, auto_shard_files=False):

        self.auto_shard_files = auto_shard_files

        from tensorflow.python.data.ops import dataset_ops
        import tensorflow as tf
        # rule 1: we assume that the dataset user passed is not batched
        if not batch_outside:
            rules = [(
                lambda dataset, is_training: isinstance(dataset, dataset_ops.BatchDataset),
                "Dataset should not be batched, please use a dataset without the batch operation")]
        else:
            rules = []

        rules += [
            (
                lambda dataset, is_training: isinstance(dataset, dataset_ops.RepeatDataset),
                "Dataset should not be repeated, please use a dataset without the repeat operation")
        ]

        if not remove_checking:
            TFDataDataset.check_rules(tf_data_dataset, rules, True)
            if validation_dataset is not None:
                TFDataDataset.check_rules(validation_dataset, rules, False)

        py_func_ops = {"PyFunc", "PyFuncStateless", "EagerPyFunc"}
        for node in tf.get_default_graph().as_graph_def().node:
            op_type = node.op
            if op_type in py_func_ops:
                raise ValueError("tf.py_func, tf.py_function, tf.numpy_function and" +
                                 " Dataset.from_generators are not supported in TFPark")

        if shuffle:
            from tensorflow.python.keras.engine import training_utils
            training_utils.verify_dataset_shuffled(tf_data_dataset)

        flatten_shapes = nest.flatten(_tf_get_shapes(tf_data_dataset))
        if batch_outside:
            flatten_shapes = [shape[1:] for shape in flatten_shapes]

        flatten_types = nest.flatten(_tf_get_types(tf_data_dataset))

        flatten_tensor_structure = [TensorMeta(dtype=flatten_types[i],
                                               shape=list(flatten_shapes[i]),
                                               name="zoo_input_{}".format(i))
                                    for i in range(len(flatten_shapes))]
        structure = _tf_get_types(tf_data_dataset)
        if isinstance(structure, tf.DType):
            structure = (structure,)
        tensor_structure = nest.pack_sequence_as(structure,
                                                 flatten_tensor_structure)

        super(TFDataDataset, self).__init__(tensor_structure, batch_size,
                                            batch_per_thread, hard_code_batch_size)
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        if intra_threads is None:
            self.intra_threads = self.core_num

        if inter_threads is None:
            self.inter_threads = 1

        if self.batch_size > 0 and self.has_batch:
            # training case
            self._per_partition_batch_size = self.batch_size // self.node_num
            self._shard_num = self.node_num
            self.drop_remainder = True
        else:
            # inference case
            self._per_partition_batch_size = self.batch_per_thread
            self._shard_num = self.total_core_num
            if hard_code_batch_size:
                self.drop_remainder = True
                logging.warning("hard_code_batch_size is set to true, so we"
                                " must drop remainder elements in the dataset"
                                " to avoid outputting small batches, the dropped"
                                " elements will not get processed. You can "
                                "pad your dataset so that the total number "
                                "of elements is divisible by the total batch size"
                                " to avoid this.")
            else:
                self.drop_remainder = False

        if self.hard_code_batch_size:
            self.drop_remainder = True
        if not batch_outside:
            tf_data_dataset = tf_data_dataset.batch(self._per_partition_batch_size,
                                                    drop_remainder=self.drop_remainder)
        if validation_dataset is not None and not batch_outside:
            drop_remainder = self.hard_code_batch_size
            validation_dataset = validation_dataset.batch(self._per_partition_batch_size,
                                                          drop_remainder=drop_remainder)

        shard_index = tf.placeholder(dtype=tf.int64, shape=())
        from tensorflow.python.distribute.input_ops import auto_shard_dataset
        if self.auto_shard_files:
            tf_data_dataset = auto_shard_dataset(tf_data_dataset, self._shard_num, shard_index)
        else:
            tf_data_dataset = tf_data_dataset.shard(self._shard_num, shard_index)
        if validation_dataset is not None:
            if self.auto_shard_files:
                validation_dataset = auto_shard_dataset(validation_dataset, self._shard_num,
                                                        shard_index)
            else:
                validation_dataset = validation_dataset.shard(self._shard_num, shard_index)

        self.shard_index = shard_index
        self.train_dataset = tf_data_dataset
        self.train_iterator = _tf_make_iterator(self.train_dataset)
        self.train_next_ops = nest.flatten(self.train_iterator.get_next())
        self.output_types = [t.as_datatype_enum
                             for t in nest.flatten(_tf_get_types(self.train_dataset))]

        self.validation_dataset = validation_dataset
        self.validation_iterator = None
        self.validation_next_ops = None

        self._train_init_op_name = self.train_iterator.initializer.name
        self._train_output_names = [op.name for op in self.train_next_ops]
        if validation_dataset is not None:
            self.validation_iterator = _tf_make_iterator(
                self.validation_dataset)
            self.validation_next_ops = nest.flatten(self.validation_iterator.get_next())
            self._val_init_op_name = self.validation_iterator.initializer.name
            self._val_output_names = [op.name for op in self.validation_next_ops]

        self.table_init_name = tf.tables_initializer().name

        self.sequential_order = sequential_order
        self.shuffle = shuffle
        self.graph = self.train_next_ops[0].graph
        self.graph_def = bytearray(self.graph.as_graph_def().SerializeToString())

    def _get_prediction_data(self):
        raise Exception("TFDataDataset cannot be used for prediction")

    def _get_evaluation_data(self):

        jvalue = callZooFunc("float", "createMiniBatchRDDFromTFDatasetEval",
                             self.graph_def, self._train_init_op_name, self.table_init_name,
                             self._train_output_names,
                             self.output_types, self.shard_index.name)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def _get_training_data(self):
        jvalue = callZooFunc("float", "createTFDataFeatureSet",
                             self.graph_def, self._train_init_op_name, self.table_init_name,
                             self._train_output_names, self.output_types, self.shard_index.name,
                             self.inter_threads, self.intra_threads)
        return FeatureSet(jvalue=jvalue)

    def _get_validation_data(self):
        if self.validation_dataset is not None:
            jvalue = callZooFunc("float", "createTFDataFeatureSet",
                                 self.graph_def, self._val_init_op_name, self.table_init_name,
                                 self._val_output_names,
                                 self.output_types, self.shard_index.name, self.inter_threads,
                                 self.intra_threads)
            return FeatureSet(jvalue=jvalue)
        return None


class TFFeatureDataset(TFDataset):
    def __init__(self, dataset, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False, validation_dataset=None):
        super(TFFeatureDataset, self).__init__(tensor_structure, batch_size,
                                               batch_per_thread, hard_code_batch_size)
        self.dataset = dataset
        self.validation_dataset = validation_dataset

    def _get_prediction_data(self):
        raise Exception("TFFeatureDataset is only supported in training")

    def _get_evaluation_data(self):
        raise Exception("TFFeatureDataset is only supported in training")

    def _get_training_data(self):
        fs = self.dataset.transform(MergeFeatureLabelFeatureTransformer())
        fs = fs.transform(SampleToMiniBatch(self.batch_size))
        return fs

    def _get_validation_data(self):
        if self.validation_dataset is not None:
            fs = self.validation_dataset.transform(
                MergeFeatureLabelFeatureTransformer())
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        raise Exception("TFFeatureDataset is only supported in training")


class TFBytesDataset(TFDataset):
    def get_num_partitions(self):
        return self.train_rdd.getNumPartitions()

    def __init__(self, string_rdd, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_string_rdd=None, sequential_order=False, shuffle=True):
        import tensorflow as tf
        tensor_structure = (TensorMeta(dtype=tf.string, shape=(), name="input"),)

        super(TFBytesDataset, self).__init__(tensor_structure, batch_size,
                                             batch_per_thread, hard_code_batch_size)

        self.train_rdd = string_rdd
        self.validation_rdd = validation_string_rdd
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def _get_prediction_data(self):
        jvalue = callZooFunc("float", "createMiniBatchRDDFromStringRDD",
                             self.train_rdd,
                             self.batch_per_thread)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def _get_evaluation_data(self):
        jvalue = callZooFunc("float", "createMiniBatchRDDFromStringRDD",
                             self.train_rdd,
                             self.batch_per_thread)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def _get_training_data(self):
        jvalue = callZooFunc("float", "createMiniBatchFeatureSetFromStringRDD",
                             self.train_rdd,
                             self.batch_size, self.sequential_order, self.shuffle)
        fs = FeatureSet(jvalue)
        return fs

    def _get_validation_data(self):
        if self.validation_rdd is not None:
            jvalue = callZooFunc("float", "createMiniBatchFeatureSetFromStringRDD",
                                 self.validation_rdd,
                                 self.batch_size, self.sequential_order, self.shuffle)
            fs = FeatureSet(jvalue)
            return fs
        return None


class TFTextDataset(TFDataset):
    def __init__(self, text_set, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_text_set=None, sequential_order=False, shuffle=True):
        super(TFTextDataset, self).__init__(tensor_structure, batch_size,
                                            batch_per_thread, hard_code_batch_size)
        self.text_set = text_set
        self.validation_text_set = validation_text_set
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def _get_prediction_data(self):
        rdd = self.text_set.get_samples().map(
            lambda sample: Sample.from_jtensor(features=sample.features,
                                               labels=JTensor.from_ndarray(np.array([0.0]))))
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread)
        return rdd_wrapper.value().toJavaRDD()

    def _get_evaluation_data(self):
        rdd = self.text_set.get_samples()
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread)
        return rdd_wrapper.value().toJavaRDD()

    def _get_training_data(self):
        sample_rdd = self.text_set.get_samples().map(
            lambda sample: Sample.from_jtensor(features=sample.features + sample.labels,
                                               labels=JTensor.from_ndarray(np.array([0.0]))))
        fs = FeatureSet.sample_rdd(sample_rdd,
                                   sequential_order=self.sequential_order,
                                   shuffle=self.shuffle)
        fs = fs.transform(SampleToMiniBatch(self.batch_size))
        return fs

    def _get_validation_data(self):
        if self.validation_text_set is not None:
            sample_rdd = self.validation_text_set.get_samples().map(
                lambda sample: Sample.from_jtensor(features=sample.features + sample.labels,
                                                   labels=JTensor.from_ndarray(np.array([0.0]))))
            fs = FeatureSet.sample_rdd(sample_rdd,
                                       sequential_order=self.sequential_order,
                                       shuffle=self.shuffle)
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.text_set.get_samples().getNumPartitions()


class TFImageDataset(TFDataset):
    def __init__(self, image_set, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_image_set=None,
                 memory_type='DRAM',
                 sequential_order=False, shuffle=True):
        super(TFImageDataset, self).__init__(tensor_structure, batch_size,
                                             batch_per_thread, hard_code_batch_size)
        self.image_set = image_set
        self.validation_image_set = validation_image_set
        self.sequential_order = sequential_order
        self.shuffle = shuffle
        self.memory_type = memory_type

    def _get_prediction_data(self):
        return self.image_set

    def _get_evaluation_data(self):
        return self.image_set.to_image_frame() \
            .transform(MergeFeatureLabelImagePreprocessing())

    def _get_training_data(self):
        fs = FeatureSet.image_set(self.image_set,
                                  self.memory_type,
                                  sequential_order=self.sequential_order,
                                  shuffle=self.shuffle)
        fs = fs.transform(MergeFeatureLabelImagePreprocessing())
        fs = fs.transform(ImageFeatureToSample())
        fs = fs.transform(SampleToMiniBatch(self.batch_size))

        return fs

    def _get_validation_data(self):
        if self.validation_image_set is not None:
            fs = FeatureSet.image_set(self.validation_image_set,
                                      sequential_order=self.sequential_order,
                                      shuffle=self.shuffle)
            fs = fs.transform(MergeFeatureLabelImagePreprocessing())
            fs = fs.transform(ImageFeatureToSample())
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.image_set.get_image().getNumPartitions()


class TFParkSampleToMiniBatch(Preprocessing):
    """
     a Transformer that converts Feature to (Feature, None).
    """

    def __init__(self,
                 batch_size,
                 drop_remainder,
                 bigdl_type="float"):
        super(TFParkSampleToMiniBatch, self).__init__(bigdl_type, batch_size, drop_remainder)


class TFNdarrayDataset(TFDataset):
    def __init__(self, rdd, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 val_rdd=None, memory_type="DRAM",
                 sequential_order=True, shuffle=False):

        super(TFNdarrayDataset, self).__init__(tensor_structure, batch_size,
                                               batch_per_thread, hard_code_batch_size)

        self.val_rdd = val_rdd
        self.rdd = rdd
        self.sequential_order = sequential_order
        self.shuffle = shuffle
        if self.hard_code_batch_size:
            logging.warning("hard_code_batch_size is set to true, so we"
                            " must drop remainder elements in the dataset"
                            " to avoid outputting small batches, the dropped"
                            " elements will not get processed. You can "
                            "pad your dataset so that the total number "
                            "of elements is divisible by the total batch size"
                            " to avoid this.")
        self.memory_type = memory_type

    def _get_prediction_data(self):
        rdd = self.rdd.map(lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread,
                                  self.hard_code_batch_size)
        return rdd_wrapper.value().toJavaRDD()

    def _get_evaluation_data(self):
        rdd = self.rdd.map(lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread,
                                  self.hard_code_batch_size)
        return rdd_wrapper.value().toJavaRDD()

    def _get_training_data(self):
        sample_rdd = self.rdd.map(
            lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
        fs = FeatureSet.sample_rdd(sample_rdd,
                                   self.memory_type,
                                   sequential_order=self.sequential_order,
                                   shuffle=self.shuffle)
        # for training there won't be any remainder, the input to SampleToMiniBatch
        # will loop indefinitely
        fs = fs.transform(TFParkSampleToMiniBatch(self.batch_size, drop_remainder=False))
        return fs

    def _get_validation_data(self):
        if self.val_rdd is not None:
            sample_rdd = self.val_rdd.map(
                lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
            fs = FeatureSet.sample_rdd(sample_rdd,
                                       sequential_order=self.sequential_order,
                                       shuffle=self.shuffle)
            fs = fs.transform(TFParkSampleToMiniBatch(self.batch_size, self.hard_code_batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.rdd.getNumPartitions()

    @staticmethod
    def from_rdd(rdd, names=None, shapes=None, types=None,
                 batch_size=-1, batch_per_thread=-1,
                 hard_code_batch_size=False, val_rdd=None,
                 features=None, labels=None,
                 memory_type="DRAM",
                 sequential_order=False,
                 shuffle=True):

        import tensorflow as tf

        if features is not None:
            feature_structure = _to_tensor_structure(features)
            if labels is not None:
                label_structure = _to_tensor_structure(labels)
                tensor_structure = (feature_structure, label_structure)

            else:
                tensor_structure = (feature_structure,)

            return TFNdarrayDataset(rdd, tensor_structure,
                                    batch_size, batch_per_thread,
                                    hard_code_batch_size, val_rdd,
                                    memory_type=memory_type,
                                    sequential_order=sequential_order,
                                    shuffle=shuffle)

        if names is not None or shapes is not None or types is not None:
            if not names:
                names = ["features", "labels"]
            if not shapes:
                shapes = [None] * len(names)

            if not types:
                types = [tf.float32] * len(names)
            tensor_structure = []
            for i in range(len(names)):
                tensor_structure.append(TensorMeta(types[i], name=names[i], shape=shapes[i]))
        else:
            tensor_structure = [TensorMeta(dtype=tf.float32), TensorMeta(dtype=tf.float32)]

        return TFNdarrayDataset(rdd, tensor_structure,
                                batch_size, batch_per_thread,
                                hard_code_batch_size, val_rdd,
                                memory_type=memory_type,
                                sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_ndarrays(tensors, batch_size=-1, batch_per_thread=-1,
                      hard_code_batch_size=False, val_tensors=None,
                      memory_type='DRAM', sequential_order=False, shuffle=True):
        sc = getOrCreateSparkContext()
        node_num, core_num = get_node_and_core_number()
        total_core_num = node_num * core_num

        rdd, tensor_structure = _tensors_to_rdd(tensors, sc, total_core_num)

        val_rdd = None
        if val_tensors is not None:
            val_rdd, _ = _tensors_to_rdd(val_tensors, sc, total_core_num)

        return TFNdarrayDataset(rdd, tensor_structure, batch_size,
                                batch_per_thread, hard_code_batch_size,
                                val_rdd, memory_type=memory_type,
                                sequential_order=sequential_order, shuffle=shuffle)


class DataFrameDataset(TFNdarrayDataset):
    @staticmethod
    def df_datatype_to_tf(dtype):
        import tensorflow as tf
        import pyspark.sql.types as df_types
        if isinstance(dtype, df_types.FloatType):
            return (tf.float32, ())
        if isinstance(dtype, df_types.IntegerType):
            return (tf.int32, ())
        if isinstance(dtype, df_types.LongType):
            return (tf.int64, ())
        if isinstance(dtype, df_types.DoubleType):
            return (tf.float64, ())
        if isinstance(dtype, df_types.ArrayType):
            return (tf.float32, (None,))
        if isinstance(dtype, VectorUDT):
            return (tf.float32, (None,))
        return None

    def __init__(self, df, feature_cols, labels_cols=None, batch_size=-1,
                 batch_per_thread=-1, hard_code_batch_size=False,
                 validation_df=None, memory_type="DRAM",
                 sequential_order=False, shuffle=True):
        assert isinstance(feature_cols, list), "feature_cols should be a list"
        if labels_cols is not None:
            assert isinstance(labels_cols, list), "label_cols should be a list"
        import pyspark
        assert isinstance(df, pyspark.sql.DataFrame)

        if labels_cols is None:
            labels_cols = []

        schema = df.schema
        feature_meta = []
        for feature_col in feature_cols:
            field = schema[feature_col]
            name = field.name
            data_type = field.dataType
            if DataFrameDataset.df_datatype_to_tf(data_type) is None:
                raise ValueError(
                    "data type {} of col {} is not supported for now".format(data_type, name))
            tf_type, tf_shape = DataFrameDataset.df_datatype_to_tf(data_type)
            feature_meta.append(TensorMeta(tf_type, name=name, shape=tf_shape))

        if labels_cols:
            label_meta = []
            for label_col in labels_cols:
                field = schema[label_col]
                name = field.name
                data_type = field.dataType
                if DataFrameDataset.df_datatype_to_tf(data_type) is None:
                    raise ValueError(
                        "data type {} of col {} is not supported for now".format(data_type, name))
                tf_type, tf_shape = DataFrameDataset.df_datatype_to_tf(data_type)
                label_meta.append(TensorMeta(tf_type, name=name, shape=tf_shape))

            tensor_structure = (feature_meta, label_meta)
        else:
            tensor_structure = (feature_meta,)

        rdd = df.rdd.map(lambda row: convert_row_to_numpy(row,
                                                          schema,
                                                          feature_cols,
                                                          labels_cols))
        if validation_df is not None:
            val_rdd = validation_df.rdd.map(lambda row: convert_row_to_numpy(row,
                                                                             schema,
                                                                             feature_cols,
                                                                             labels_cols))
        else:
            val_rdd = None

        super(DataFrameDataset, self).__init__(rdd, tensor_structure, batch_size,
                                               batch_per_thread, hard_code_batch_size,
                                               val_rdd, memory_type, sequential_order, shuffle)


def _check_compatible(names, structure, data_type="model_input"):
    if isinstance(structure, dict):
        err_msg = f"all {data_type} names should exist in data, " \
                  f"got {data_type} {names}, data {structure}"
        assert all([name in structure for name in names]), err_msg
    elif isinstance(structure, list) or isinstance(structure, tuple):
        err_msg = f"{data_type} number does not match data number, " \
                  f"got {data_type} {names}, data {structure}"
        assert len(nest.flatten(structure)) == len(names), err_msg
    else:
        assert len(names) == 1, f"data does not match {data_type}, " \
                                f"data {structure}, {data_type} {names}"


def check_data_compatible(dataset, model, mode):
    input_names = model.input_names
    output_names = model.output_names
    err_msg = f"each element in dataset should be a tuple for {mode}, " \
              f"but got {dataset.tensor_structure}"
    if mode == "train" or mode == "evaluate":
        assert isinstance(dataset.tensor_structure, tuple), err_msg

        feature = dataset.tensor_structure[0]
        _check_compatible(input_names, feature, data_type="model_input")

        label = dataset.tensor_structure[1]
        _check_compatible(output_names, label, data_type="model_target")
    else:
        _check_compatible(input_names, dataset.tensor_structure, data_type="model_input")


def _standarize_feature_label_dataset(dataset, model):
    input_names = model.input_names
    output_names = model.output_names

    def _process_labels(ys):
        if isinstance(ys, dict):
            return {k: np.expand_dims(y, axis=-1) if y.ndim == 0 else y for k, y in ys.items()}
        elif isinstance(ys, list):
            return [np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys]
        elif isinstance(ys, tuple):
            return tuple([np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys])
        else:
            return np.expand_dims(ys, axis=-1) if ys.ndim == 0 else ys

    def _training_reorder(x, input_names, output_names):
        assert isinstance(x, tuple)

        return (_reorder(x[0], input_names), _reorder(x[1], output_names))

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list) or isinstance(x, tuple):
            return x
        else:
            return [x]

    rdd = dataset.rdd.map(lambda x: (x[0], _process_labels(x[1])))\
        .map(lambda sample: _training_reorder(sample, input_names, output_names))
    if dataset.val_rdd is not None:
        val_rdd = dataset.val_rdd.map(lambda x: (x[0], _process_labels(x[1])))\
            .map(lambda sample: _training_reorder(sample, input_names, output_names))
    else:
        val_rdd = None
    tensor_structure = _training_reorder(dataset.tensor_structure, input_names, output_names)
    new_dataset = TFNdarrayDataset(rdd, tensor_structure, dataset.batch_size,
                                   -1, dataset.hard_code_batch_size, val_rdd,
                                   dataset.memory_type, dataset.sequential_order, dataset.shuffle)
    new_dataset.batch_per_thread = dataset.batch_per_thread
    return new_dataset


def _standarize_feature_dataset(dataset, model):
    input_names = model.input_names

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list):
            return x
        elif isinstance(x, tuple):
            return list(x)
        return [x]

    rdd = dataset.rdd.map(lambda sample: _reorder(sample, input_names))
    feature_schema = _reorder(dataset.tensor_structure[0], input_names)

    dataset = TFNdarrayDataset(rdd, feature_schema, -1,
                               dataset.batch_per_thread, dataset.hard_code_batch_size,
                               memory_type=dataset.memory_type,
                               sequential_order=dataset.sequential_order,
                               shuffle=dataset.shuffle
                               )
    return dataset


def _standardize_keras_target_data(x, ys):
    def check_y_dims(y):
        return y is not None and len(y.shape) == 0

    if isinstance(ys, dict):
        ys = {k: tf.expand_dims(y, axis=0) if check_y_dims(y) else y for k, y in ys.items()}
    elif isinstance(ys, list):
        ys = [tf.expand_dims(y, axis=0) if check_y_dims(y) else y for y in ys]
    elif isinstance(ys, tuple):
        ys = tuple(tf.expand_dims(y, axis=0) if check_y_dims(y) else y for y in ys)
    else:
        ys = tf.expand_dims(ys, axis=0) if check_y_dims(ys) else ys

    return x, ys
