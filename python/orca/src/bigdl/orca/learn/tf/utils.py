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
import tempfile
import os
from os.path import join, basename, dirname
import re
import shutil
import tensorflow as tf
import numpy as np

from zoo.orca.data import SparkXShards
from zoo.tfpark.tf_dataset import TFDataset
from zoo.orca.data.utils import get_spec, flatten_xy
from zoo.common.utils import put_local_file_to_remote, get_remote_file_to_local, get_file_list,\
    is_local_path


def xshards_to_tf_dataset(data_shard,
                          batch_size=-1, batch_per_thread=-1,
                          validation_data_shard=None,
                          hard_code_batch_size=False,
                          sequential_order=False,
                          shuffle=True):
    # todo data_shard.head ?
    feature_spec, label_spec = data_shard._for_each(get_spec(allow_tuple=True, allow_list=False))\
        .first()

    feature_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in feature_spec]
    label_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in label_spec] \
        if label_spec is not None else None

    assert batch_size != -1 or batch_per_thread != -1, \
        "one of batch_size and batch_per_thread should be specified"

    val_rdd = None if validation_data_shard is None \
        else validation_data_shard.rdd.flatMap(flatten_xy(allow_tuple=True, allow_list=False))

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(flatten_xy(allow_tuple=True,
                                                                   allow_list=False)),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 val_rdd=val_rdd,
                                 hard_code_batch_size=hard_code_batch_size,
                                 sequential_order=sequential_order,
                                 shuffle=shuffle)

    return dataset


def convert_predict_to_dataframe(df, prediction_rdd):
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, FloatType, ArrayType
    from pyspark.ml.linalg import VectorUDT, Vectors

    def combine(pair):
        # list of np array
        if isinstance(pair[1], list):
            row = Row(*([pair[0][col] for col in pair[0].__fields__] +
                        [[Vectors.dense(elem) for elem in pair[1]]]))
            return row, ArrayType(VectorUDT())
        # scalar
        elif len(pair[1].shape) == 0:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [float(pair[1].item(0))]))
            return row, FloatType()
        # np array
        else:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [Vectors.dense(pair[1])]))
            return row, VectorUDT()

    combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
    type = combined_rdd.map(lambda data: data[1]).first()
    result_rdd = combined_rdd.map(lambda data: data[0])
    schema = StructType(df.schema.fields + [StructField('prediction', type)])
    result_df = result_rdd.toDF(schema)
    return result_df


def convert_predict_to_xshard(prediction_rdd):
    def transform_predict(iter):
        predictions = list(iter)
        # list of np array
        if isinstance(predictions[0], list):
            predictions = np.array(predictions).T.tolist()
            result = [np.array(predict) for predict in predictions]
            return [{'prediction': result}]
        # np array
        else:
            return [{'prediction': np.array(predictions)}]

    return SparkXShards(prediction_rdd.mapPartitions(transform_predict))


def find_latest_checkpoint(model_dir):
    import os
    import re
    import datetime
    ckpt_path = None
    latest_version = None
    for (root, dirs, files) in os.walk(model_dir, topdown=True):
        temp_versions = []
        timestamps = []
        for dir in dirs:
            if re.match('(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$', dir) is not None:
                try:
                    # check if dir name is date time
                    datetime.datetime.strptime(dir, '%Y-%m-%d_%H-%M-%S')
                    timestamps.append(dir)
                except:
                    continue
        if timestamps:
            start_dir = os.path.join(root, max(timestamps))
            return find_latest_checkpoint(start_dir)
        for file_name in files:
            if re.match("^optimMethod-TFParkTraining\.[0-9]+$", file_name) is not None:
                version = int(file_name.split(".")[1])
                temp_versions.append(version)
        if temp_versions:
            ckpt_path = root
            latest_version = max(temp_versions)
            break
    return ckpt_path, latest_version
