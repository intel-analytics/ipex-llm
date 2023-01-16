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

from pyspark import SparkContext
from pyspark.sql import SparkSession

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.file import open_text, write_text
from bigdl.orca.data.image.utils import chunks, dict_to_row, row_to_dict, encode_schema, \
    decode_schema, SchemaField, FeatureType, DType, ndarray_dtype_to_dtype, \
    decode_feature_type_ndarray, pa_fs
from bigdl.orca.data.image.voc_dataset import VOCDatasets
from bigdl.dllib.utils.common import get_node_and_core_number
import os
import numpy as np
import random
import io
import math
from bigdl.dllib.utils.log4Error import invalidInputError
from numpy import ndarray, uint32

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, Any
if TYPE_CHECKING:
    import tensorflow as tf
    from torch.utils.data import DataLoader
    from bigdl.orca.data.shard import SparkXShards
    from pyspark.rdd import RDD
    from bigdl.orca.data.tf.data import TensorSliceDataset, Dataset


class ParquetDataset:
    @staticmethod
    def write(path: str,
              generator: List[Dict[str, Union[str, int, float, ndarray]]],
              schema: Dict[str, SchemaField],
              block_size: int=1000,
              write_mode: str="overwrite",
              **kwargs) -> None:
        """
        Take each record in the generator and write it to a parquet file.

        **generator**
        Each record in the generator is a dict, the key is a string and will be the
        column name of saved parquet record and the value is the data.

        **schema**
        schema defines the name, dtype, shape of a column, as well as the feature
        type of a column. The feature type, defines how to encode and decode the column value.

        There are three kinds of feature type:
        1. Scalar, such as a int or float number, or a string, which can be directly mapped
           to a parquet type
        2. NDarray, which takes a np.ndarray and save it serialized bytes. The corresponding
           parquet type is BYTE_ARRAY .
        3. Image, which takes a string representing a image file in local file system and save
           the raw file content bytes.
           The corresponding parquet type is BYTE_ARRAY.

        :param path: the output path, e.g. file:///output/path, hdfs:///output/path
        :param generator: generate a dict, whose key is a string and value is one of
                          (a scalar value, ndarray, image file path)
        :param schema: a dict, whose key is a string, value is one of
                      (schema_field.Scalar, schema_field.NDarray, schema_field.Image)
        :param kwargs: other args
        """

        sc = init_nncontext()
        spark = SparkSession(sc)
        node_num, core_num = get_node_and_core_number()
        for i, chunk in enumerate(chunks(generator, block_size)):
            chunk_path = os.path.join(path, f"chunk={i}")
            rows_rdd = sc.parallelize(chunk, core_num * node_num) \
                .map(lambda x: dict_to_row(schema, x))
            spark.createDataFrame(rows_rdd).write.mode(
                write_mode).parquet(chunk_path)
        metadata_path = os.path.join(path, "_orca_metadata")

        write_text(metadata_path, encode_schema(schema))

    @staticmethod
    def _read_as_dict_rdd(path: str) -> Tuple["RDD[Any]", Dict[str, SchemaField]]:
        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)

        df = spark.read.parquet(path)
        schema_path = os.path.join(path, "_orca_metadata")

        j_str = open_text(schema_path)[0]

        schema = decode_schema(j_str)

        rdd = df.rdd.map(lambda r: row_to_dict(schema, r))
        return rdd, schema

    @staticmethod
    def _read_as_xshards(path: str) -> "SparkXShards":
        rdd, schema = ParquetDataset._read_as_dict_rdd(path)

        def merge_records(schema, iter):
            l = list(iter)
            result = {}
            for k in schema.keys():
                result[k] = []
            for i, rec in enumerate(l):

                for k in schema.keys():
                    result[k].append(rec[k])
            for k, v in schema.items():
                if not v.feature_type == FeatureType.IMAGE:
                    result[k] = np.stack(result[k])

            return [result]

        result_rdd = rdd.mapPartitions(
            lambda iter: merge_records(schema, iter))
        xshards = SparkXShards(result_rdd)
        return xshards

    @staticmethod
    def read_as_tf(path: str) -> "TensorSliceDataset":
        """
        return a orca.data.tf.data.Dataset
        :param path:
        :return:
        """
        from bigdl.orca.data.tf.data import Dataset
        xshards = ParquetDataset._read_as_xshards(path)
        return Dataset.from_tensor_slices(xshards)

    @staticmethod
    def read_as_torch(path):
        """
        return a orca.data.torch.data.DataLoader
        :param path:
        :return:
        """
        invalidInputError(False, "not implemented")


class ParquetIterable:
    """
    Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
    The Dataset will contain all elements of total dataset whose index % num_shards = rank.
    """

    def __init__(self, row_group, schema, num_shards=None,
                 rank=None, transforms=None):
        self.row_group = row_group

        # To get the indices we expect
        self.row_group.sort()

        self.num_shards = num_shards
        self.rank = rank
        self.transforms = transforms
        self.datapiece = self._load_data(schema)
        self.cur = 0
        self.cur_tail = len(self.datapiece)

    def _load_data(self, schema):
        import pyarrow.parquet as pq
        if self.num_shards is None or self.rank is None:
            filter_row_group_indexed = [
                index for index in list(range(len(self.row_group)))]
        else:
            invalidInputError(self.num_shards <= len(self.row_group),
                              "num_shards should be not larger than partitions. but got"
                              " num_shards {} with "
                              "partitions {}.".format(self.num_shards, len(self.row_group)))
            invalidInputError(self.rank < self.num_shards,
                              "shard index should be included in [0,num_shard),"
                              "but got rank {} with "
                              "num_shard {}.".format(self.rank, self.num_shards))
            filter_row_group_indexed = [index for index in list(range(len(self.row_group)))
                                        if index % self.num_shards == self.rank]

        data_record = []
        for select_chunk_path in [self.row_group[i] for i in filter_row_group_indexed]:
            pq_table = pq.read_table(select_chunk_path)
            df = decode_feature_type_ndarray(pq_table.to_pandas(), schema)
            data_record.extend(df.to_dict("records"))
        return data_record

    def __iter__(self):
        return self

    def __next__(self):
        # move iter here so we can do transforms
        if self.cur < self.cur_tail:
            elem = self.datapiece[self.cur]
            self.cur += 1
            if self.transforms:
                return self.transforms(elem)
            else:
                return elem
        else:
            raise StopIteration

    def __call__(self):
        self.cur = 0
        return self


def _read32(bytestream: io.BufferedReader) -> uint32:
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_mnist_images(image_filepath: str) -> ndarray:
    with open(image_filepath, "rb") as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            invalidInputError(False,
                              'Invalid magic number %d in MNIST image file: %s' %
                              (magic, image_filepath))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(int(rows * cols * num_images))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _extract_mnist_labels(labels_filepath: str) -> ndarray:
    with open(labels_filepath, "rb") as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            invalidInputError(False,
                              'Invalid magic number %d in MNIST label file: %s' %
                              (magic, labels_filepath))
        num_items = _read32(bytestream)
        buf = bytestream.read(int(num_items))
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def write_from_directory(directory: str,
                         label_map: Dict[str, int],
                         output_path: str,
                         shuffle: bool=True,
                         **kwargs) -> None:
    labels = os.listdir(directory)
    valid_labels = [label for label in labels if label in label_map]
    generator = []
    for label in valid_labels:
        label_path = os.path.join(directory, label)
        images = os.listdir(label_path)
        for image in images:
            image_path = os.path.join(label_path, image)
            generator.append({"image": image_path,
                              "label": label_map[label],
                              "image_id": image_path,
                              "label_str": label})
    if shuffle:
        random.shuffle(generator)

    schema = {"image": SchemaField(feature_type=FeatureType.IMAGE,
                                   dtype=DType.FLOAT32,
                                   shape=()),
              "label": SchemaField(feature_type=FeatureType.SCALAR,
                                   dtype=DType.INT32,
                                   shape=()),
              "image_id": SchemaField(feature_type=FeatureType.SCALAR,
                                      dtype=DType.STRING,
                                      shape=()),
              "label_str": SchemaField(feature_type=FeatureType.SCALAR,
                                       dtype=DType.STRING,
                                       shape=())}

    ParquetDataset.write(output_path, generator, schema, **kwargs)  # type: ignore


def _write_ndarrays(images: ndarray, labels: ndarray, output_path: str, **kwargs) -> None:
    images_shape = [int(x) for x in images.shape[1:]]
    labels_shape = [int(x) for x in labels.shape[1:]]
    schema = {
        "image": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(images.dtype),
                             shape=images_shape),
        "label": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(labels.dtype),
                             shape=labels_shape)
    }

    def make_generator():
        for i in range(images.shape[0]):
            yield {"image": images[i], "label": labels[i]}

    ParquetDataset.write(output_path, make_generator(), schema, **kwargs)


def write_mnist(image_file: str, label_file: str, output_path: str, **kwargs) -> None:
    images = _extract_mnist_images(image_filepath=image_file)
    labels = _extract_mnist_labels(labels_filepath=label_file)
    _write_ndarrays(images, labels, output_path, **kwargs)


def write_voc(voc_root_path: str,
              splits_names: List[Tuple[int, str]],
              output_path: str,
              **kwargs) -> None:
    custom_classes = kwargs.get("classes", None)
    voc_datasets = VOCDatasets(
        voc_root_path, splits_names, classes=custom_classes)

    def make_generator():
        for img_path, label in voc_datasets:
            yield {"image": img_path, "label": label, "image_id": img_path}

    image, label = voc_datasets[0]
    label_shape = (-1, label.shape[-1])
    schema = {
        "image": SchemaField(feature_type=FeatureType.IMAGE,
                             dtype=DType.FLOAT32,
                             shape=()),
        "label": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(label.dtype),
                             shape=label_shape),
        "image_id": SchemaField(feature_type=FeatureType.SCALAR,
                                dtype=DType.STRING,
                                shape=())
    }
    kwargs = {key: value for key, value in kwargs.items() if key not in [
        "classes"]}
    ParquetDataset.write(output_path, make_generator(), schema, **kwargs)


def _check_arguments(_format: str,
                     kwargs: Dict[str, Any],
                     args: Union[List[str], Tuple[str]]) -> None:
    for keyword in args:
        invalidInputError(keyword in kwargs,
                          keyword + " is not specified for format " + _format + ".")


def write_parquet(format: str, output_path: str, *args, **kwargs) -> None:
    supported_format = {"mnist", "image_folder", "voc"}
    if format not in supported_format:
        invalidInputError(False,
                          format + " is not supported, should be one of 'mnist',"
                                   "'image_folder' and 'voc'.")

    format_to_function = {"mnist": (write_mnist, ["image_file", "label_file"]),
                          "image_folder": (write_from_directory, ["directory", "label_map"]),
                          "voc": (write_voc, ["voc_root_path", "splits_names"])}
    func, required_args = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    func(output_path=output_path, *args, **kwargs)


def read_as_tfdataset(path: str,
                      output_types: Dict[str, 'tf.Dtype'],
                      config: Dict[str, int],
                      output_shapes: Optional[Dict[str, 'tf.TensorShape']]=None,
                      *args, **kwargs) -> "tf.data.Dataset":
    """
    return a orca.data.tf.data.Dataset
    :param path:
    :return:
    """
    path, _ = pa_fs(path)
    import tensorflow as tf

    schema_path = os.path.join(path, "_orca_metadata")
    j_str = open_text(schema_path)[0]
    schema = decode_schema(j_str)

    row_group = []

    for root, dirs, files in os.walk(path):
        for name in dirs:
            if name.startswith("chunk="):
                chunk_path = os.path.join(path, name)
                row_group.append(chunk_path)

    dataset = ParquetIterable(
        row_group=row_group, schema=schema, num_shards=config.get("num_shards"),
        rank=config.get("rank"))

    return tf.data.Dataset.from_generator(dataset, output_types=output_types,
                                          output_shapes=output_shapes)


def read_as_dataloader(path: str,
                       config: Dict[str, int],
                       transforms: Optional[Callable] = None,
                       batch_size: int=1,
                       *args, **kwargs) -> "DataLoader":
    path, _ = pa_fs(path)
    import tensorflow as tf
    import torch

    schema_path = os.path.join(path, "_orca_metadata")
    j_str = open_text(schema_path)[0]
    schema = decode_schema(j_str)

    row_group = []

    for root, dirs, files in os.walk(path):
        for name in dirs:
            if name.startswith("chunk="):
                chunk_path = os.path.join(path, name)
                row_group.append(chunk_path)

    class ParquetIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, row_group, schema, num_shards=None,
                     rank=None, transforms=None):
            super().__init__()
            self.iterator = ParquetIterable(row_group, schema, num_shards, rank, transforms)
            self.cur = self.iterator.cur
            self.cur_tail = self.iterator.cur_tail

        def __iter__(self):
            return self.iterator.__iter__()

        def __next__(self):
            self.iterator.__next__()

    def worker_init_fn(w_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        iter_start = dataset.cur
        iter_end = dataset.cur_tail
        per_worker = int(
            math.ceil(iter_end - iter_start / float(worker_info.num_workers)))
        w_id = worker_info.id
        dataset.cur = iter_start + w_id * per_worker
        dataset.cur_tail = min(dataset.cur + per_worker, iter_end)

    dataset = ParquetIterableDataset(
        row_group=row_group, schema=schema, num_shards=config.get("num_shards"),
        rank=config.get("rank"), transforms=transforms)

    return torch.utils.data.DataLoader(dataset, num_workers=config.get("num_workers", 0),
                                       batch_size=batch_size, worker_init_fn=worker_init_fn)


def read_parquet(format: str,
                 path: str,
                 transforms: Optional[Callable]=None,
                 config: Optional[Dict[str, int]]=None,
                 batch_size: int=1,
                 *args, **kwargs) -> Union["tf.data.Dataset", "DataLoader"]:
    supported_format = {"tf_dataset", "dataloader"}
    if format not in supported_format:
        invalidInputError(False,
                          format + " is not supported, should be 'tf_dataset' or 'dataloader'.")

    format_to_function = {"tf_dataset": (read_as_tfdataset,
                                         ["output_types"]),
                          "dataloader": (read_as_dataloader,
                                         [])}  # type: Dict[str, Tuple]
    func, required_args = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    return func(path=path, config=config or {},
                transforms=transforms, batch_size=batch_size, *args, **kwargs)
