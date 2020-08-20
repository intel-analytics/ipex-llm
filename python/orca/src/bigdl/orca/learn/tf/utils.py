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
from pyspark.sql.dataframe import DataFrame

from zoo.orca.data import SparkXShards
from zoo.orca.data.tf.data import Dataset, TFDataDataset2
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
        if isinstance(data, tf.data.Dataset):
            assert isinstance(validation_data, tf.data.Dataset), \
                "train data and validation data should be both tf.data.Dataset"

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
    elif isinstance(data, tf.data.Dataset):
        dataset = TFDataset.from_tf_data_dataset(data,
                                                 batch_size,
                                                 batch_per_thread,
                                                 hard_code_batch_size,
                                                 validation_data,
                                                 sequential_order,
                                                 shuffle)
    else:
        raise ValueError("data must be SparkXShards or orca.data.tf.Dataset or "
                         "Spark DataFrame or tf.data.Dataset")

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


def save_tf_checkpoint(sess, checkpoint_path, saver=None):
    """
    Save tf checkpoint without using native tensorflow remote access method.
    :param sess: tf session to be saved.
    :param checkpoint_path: checkpoint path. Could be local, hdfs, s3 filesystems.
    :param saver: tf saver to save checkpoint
    """
    if is_local_path(checkpoint_path):
        if saver is None:
            saver = tf.train.Saver()
        saver.save(sess, checkpoint_path)
    else:
        ckpt_name = basename(checkpoint_path)
        remote_dir = dirname(checkpoint_path)
        # save to local checkpoint
        temp = tempfile.mkdtemp()
        if saver is None:
            saver = tf.train.Saver()
        saver.save(sess, join(temp, ckpt_name))
        # change checkpoint file
        with open(join(temp, "checkpoint")) as f:
            new_lines = []
            lines = f.readlines()
            # replace model_checkpoint_path and all_model_checkpoint_paths to checkpoint name
            #  instead of the absolute checkpoint path
            for line in lines:
                if re.compile("^model_checkpoint_path: \"(.*)\"$").match(line):
                    new_lines.append("model_checkpoint_path: \"{}\"\n".format(ckpt_name))
                elif re.compile("^all_model_checkpoint_paths: \"(.*)\"$").match(line):
                    new_lines.append("all_model_checkpoint_paths: \"{}\"\n".format(ckpt_name))
                else:
                    new_lines.append(line)
        with open(join(temp, "checkpoint"), 'w') as f:
            f.writelines(new_lines)
        # move to remote
        [put_local_file_to_remote(join(temp, file), join(remote_dir, file), over_write=True)
         for file in os.listdir(temp)]
        shutil.rmtree(temp)


def get_checkpoint_state(checkpoint_dir):
    """
    Get tf checkpoint state from checkpoint directory without using native tensorflow accessing
    remote method.
    :param checkpoint_dir: tensorflow checkpoint directory. Could be local, hdfs, s3 filesystems.
    :return: tf checkpoint protobuf
    """
    if is_local_path(checkpoint_dir):
        return tf.train.get_checkpoint_state(checkpoint_dir)
    else:
        # check if checkpoint file exists
        file_list = get_file_list(checkpoint_dir)
        has_checkpoint = False
        for file in file_list:
            if basename(file) == 'checkpoint':
                has_checkpoint = True
                break
        if not has_checkpoint:
            return None
        # get checkpoint file
        temp = tempfile.mkdtemp()
        get_remote_file_to_local(join(checkpoint_dir, "checkpoint"), join(temp, "checkpoint"))
        ckpt_name = None
        with open(join(temp, "checkpoint")) as f:
            lines = f.readlines()
            # get checkpoint name from 'checkpoint' file
            for line in lines:
                m = re.compile("^model_checkpoint_path: \"(.*)\"$").match(line)
                if m:
                    ckpt_name = m.group(1)
                    break
        if ckpt_name is None:
            shutil.rmtree(temp)
            return None
        # filter checkpoint files
        checkpoint_files = [file for file in file_list if basename(file).startswith(ckpt_name)]
        if not checkpoint_files:
            shutil.rmtree(temp)
            return None
        # get checkpoint files to local
        [get_remote_file_to_local(file, join(temp, basename(file))) for file in checkpoint_files]
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(temp)
        if not ckpt:
            shutil.rmtree(temp)
            return None
        ckpt.model_checkpoint_path = join(checkpoint_dir, ckpt_name)
        ckpt.all_model_checkpoint_paths[:] = [join(checkpoint_dir, ckpt_name)]
        shutil.rmtree(temp)
        return ckpt


def load_tf_checkpoint(sess, checkpoint_path, saver=None):
    """
    Load tensorflow checkpoint from checkpoint path without using native tensorflow accessing
    remote method.
    :param sess: tensorflow session to be loaded to.
    :param checkpoint_path: tensorflow checkpoint path. Could be local, hdfs, s3 filesystems.
    :param saver: tensorflow saver to load checkpoint
    """
    if is_local_path(checkpoint_path):
        if saver is None:
            saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
    else:
        ckpt_name = basename(checkpoint_path)
        checkpoint_dir = dirname(checkpoint_path)
        # get remote file lists
        file_list = get_file_list(checkpoint_dir)
        # filter checkpoint files
        checkpoint_files = [file for file in file_list if basename(file).startswith(ckpt_name)]
        # get checkpoint files to local
        temp = tempfile.mkdtemp()
        [get_remote_file_to_local(file, join(temp, basename(file))) for file in checkpoint_files]
        if saver is None:
            saver = tf.train.Saver()
        try:
            saver.restore(sess, join(temp, ckpt_name))
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(temp)
