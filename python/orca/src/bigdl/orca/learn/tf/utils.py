#
# Copyright 2018 Analytics Zoo Authors.
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
from pyspark.sql.dataframe import DataFrame
import tensorflow as tf

from zoo.tfpark.tf_dataset import TFDataset
from zoo.orca.data import SparkXShards
from zoo.orca.data.tf.data import Dataset, TFDataDataset2


def xshards_to_tf_dataset(data_shard,
                          batch_size=-1, batch_per_thread=-1,
                          validation_data_shard=None,
                          hard_code_batch_size=False,
                          sequential_order=False,
                          shuffle=True):
    # todo data_shard.head ?
    import numpy as np

    def check_data_type_and_to_list(data):
        result = {}
        assert isinstance(data, dict), "each shard should be an dict"
        assert "x" in data, "key x should in each shard"
        x = data["x"]
        if isinstance(x, np.ndarray):
            new_x = [x]
        elif isinstance(x, tuple) and all([isinstance(xi, np.ndarray) for xi in x]):
            new_x = x
        else:
            raise ValueError("value of x should be a ndarray or a tuple of ndarrays")
        result["x"] = new_x
        if "y" in data:
            y = data["y"]
            if isinstance(y, np.ndarray):
                new_y = [y]
            elif isinstance(y, tuple) and all([isinstance(yi, np.ndarray) for yi in y]):
                new_y = y
            else:
                raise ValueError("value of x should be a ndarray or a tuple of ndarrays")
            result["y"] = new_y
        return result

    def get_spec(data):
        data = check_data_type_and_to_list(data)
        feature_spec = [(feat.dtype, feat.shape[1:])
                        for feat in data["x"]]
        if "y" in data:
            label_spec = [(label.dtype, label.shape[1:])
                          for label in data["y"]]
        else:
            label_spec = None
        return (feature_spec, label_spec)

    (feature_spec, label_spec) = data_shard.rdd.map(get_spec).first()

    feature_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in feature_spec]
    label_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in label_spec] \
        if label_spec is not None else None

    assert batch_size != -1 or batch_per_thread != -1, \
        "one of batch_size and batch_per_thread should be specified"

    # todo this might be very slow
    def flatten(data):
        data = check_data_type_and_to_list(data)
        features = data["x"]

        has_label = "y" in data
        labels = data["y"] if has_label else None
        length = features[0].shape[0]

        for i in range(length):
            fs = [feat[i] for feat in features]
            if has_label:
                ls = [l[i] for l in labels]
                yield (fs, ls)
            else:
                yield (fs,)

    val_rdd = None if validation_data_shard is None \
        else validation_data_shard.rdd.flatMap(flatten)

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(flatten),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 val_rdd=val_rdd,
                                 hard_code_batch_size=hard_code_batch_size,
                                 sequential_order=sequential_order,
                                 shuffle=shuffle)

    return dataset


def to_dataset(data, batch_size, batch_per_thread, validation_data,
               feature_cols, labels_cols, hard_code_batch_size,
               sequential_order, shuffle):
    if validation_data:
        if isinstance(data, SparkXShards):
            assert isinstance(validation_data, SparkXShards), \
                "train data and validation data should be both SparkXShards"
        if isinstance(data, Dataset):
            assert isinstance(validation_data, Dataset), \
                "train data and validation data should be both orca.data.tf.Dataset"
        if isinstance(data, DataFrame):
            assert isinstance(validation_data, DataFrame), \
                "train data and validation data should be both Spark DataFrame"

    if isinstance(data, SparkXShards):
        dataset = xshards_to_tf_dataset(data,
                                        batch_size,
                                        batch_per_thread,
                                        validation_data,
                                        hard_code_batch_size=hard_code_batch_size,
                                        sequential_order=sequential_order,
                                        shuffle=shuffle)
    elif isinstance(data, Dataset):
        dataset = TFDataDataset2(data, batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 validation_dataset=validation_data)
    elif isinstance(data, DataFrame):
        dataset = TFDataset.from_dataframe(data, feature_cols, labels_cols,
                                           batch_size,
                                           batch_per_thread,
                                           hard_code_batch_size,
                                           validation_data,
                                           sequential_order,
                                           shuffle
                                           )
    else:
        raise ValueError("data must be SparkXShards or orca.data.tf.Dataset or Spark DataFrame")

    return dataset


def convert_predict_to_dataframe(df, prediction_rdd):
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, FloatType
    from pyspark.ml.linalg import VectorUDT, Vectors

    def combine(pair):
        # scalar
        if len(pair[1].shape) == 0:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [float(pair[1].item(0))]))
            return row, FloatType()
        else:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [Vectors.dense(pair[1])]))
            return row, VectorUDT()

    combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
    type = combined_rdd.map(lambda data: data[1]).first()
    result_rdd = combined_rdd.map(lambda data: data[0])
    schema = StructType(df.schema.fields + [StructField('prediction', type)])
    result_df = result_rdd.toDF(schema)
    return result_df
