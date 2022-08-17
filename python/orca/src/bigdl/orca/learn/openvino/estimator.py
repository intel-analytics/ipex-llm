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
import math
import os.path

from pyspark.sql import DataFrame

from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.spark_estimator import Estimator as SparkEstimator
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils import nest
from bigdl.dllib.nncontext import init_nncontext

from openvino.inference_engine import IECore
import numpy as np
from bigdl.dllib.utils.log4Error import *


class Estimator(object):
    @staticmethod
    def from_openvino(*, model_path):
        """
        Load an openVINO Estimator.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        """
        return OpenvinoEstimator(model_path=model_path)


class OpenvinoEstimator(SparkEstimator):
    def __init__(self,
                 *,
                 model_path):
        self.load(model_path)

    def fit(self, data, epochs, batch_size=32, feature_cols=None, label_cols=None,
            validation_data=None, checkpoint_trigger=None):
        """
        Fit is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def predict(self, data, feature_cols=None, batch_size=4, inputs=None):
        """
        Predict input data

        :param batch_size: Int. Set batch Size, default is 4.
        :param data: data to be predicted. XShards, Spark DataFrame, numpy array and list of numpy
               arrays are supported. If data is XShards, each partition is a dictionary of  {'x':
               feature}, where feature(label) is a numpy array or a list of numpy arrays.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame. Default: None.
        :return: predicted result.
                 If the input data is XShards, the predict result is a XShards, each partition
                 of the XShards is a dictionary of {'prediction': result}, where the result is a
                 numpy array or a list of numpy arrays.
                 If the input data is numpy arrays or list of numpy arrays, the predict result is
                 a numpy array or a list of numpy arrays.
        """
        sc = init_nncontext()
        model_bytes_broadcast = sc.broadcast(self.model_bytes)
        weight_bytes_broadcast = sc.broadcast(self.weight_bytes)
        if inputs is not None:
            invalidInputError(set(inputs) == set(self.inputs),
                              "The inputs names need to match the model inputs, the model inputs: "
                              + ",".join(self.inputs))
        else:
            inputs = self.inputs
        outputs = list(self.output_dict.keys())
        invalidInputError(len(outputs) != 0, "The number of model outputs should not be 0.")
        isDF = True
        schma = None

        def partition_inference(partition):
            model_bytes = model_bytes_broadcast.value
            weight_bytes = weight_bytes_broadcast.value
            ie = IECore()
            config = {'CPU_THREADS_NUM': str(self.core_num)}
            ie.set_config(config, 'CPU')
            net = ie.read_network(model=model_bytes,
                                  weights=weight_bytes, init_from_buffer=True)
            net.batch_size = batch_size
            local_model = ie.load_network(network=net, device_name="CPU",
                                          num_requests=1)
            infer_request = local_model.requests[0]

            def add_elem(d):
                d_len = len(d)
                if d_len < batch_size:
                    rep_time = [1] * (d_len - 1)
                    rep_time.append(batch_size - d_len + 1)
                    return np.repeat(d, rep_time, axis=0), d_len
                else:
                    return d, d_len

            # for idx, batch_data in enumerate(partition):
            for batch_data in partition:
                input_dict = dict()
                elem_num = 0
                if isinstance(batch_data, list):
                    for i, input in enumerate(inputs):
                        input_dict[input], elem_num = add_elem(batch_data[i])
                else:
                    input_dict[inputs[0]], elem_num = add_elem(batch_data)
                infer_request.infer(input_dict)
                if len(outputs) == 1:
                    result = infer_request.output_blobs[outputs[0]].buffer[:elem_num]
                else:
                    result = list(map(lambda output:
                                      infer_request.output_blobs[output].buffer[:elem_num],
                                      outputs))
                print("inference ", elem_num)
                yield result

            batch_dict = {col: [] for col in feature_cols}
            cnt = 0
            schema_dict = {col: schema[col].dataType for col in feature_cols}
            import pyspark.sql.types as df_types
            from pyspark.ml.linalg import DenseVector
            for row in partition:
                for col in feature_cols:
                    batch_dict[col].append(row[col])

            for col, value in batch_dict.items():
                feature_type = schema_dict[col]
                if isinstance(feature_type, df_types.FloatType):
                    result.append(np.array(row[name]).astype(np.float32))
                elif isinstance(feature_type, df_types.IntegerType):
                    result.append(np.array(row[name]).astype(np.int32))
                elif isinstance(feature_type, df_types.ArrayType):
                    if isinstance(feature_type.elementType, df_types.StringType):
                        result.append(np.array(row[name]).astype(np.str))
                    else:
                        result.append(np.array(row[name]).astype(np.float32))
                elif isinstance(value[0], DenseVector):
                    result.append(row[name].values.astype(np.float32))

        def predict_transform(dict_data, batch_size):
            invalidInputError(isinstance(dict_data, dict), "each shard should be an dict")
            invalidInputError("x" in dict_data, "key x should in each shard")
            feature_data = dict_data["x"]
            if isinstance(feature_data, np.ndarray):
                invalidInputError(feature_data.shape[0] <= batch_size,
                                  "The batch size of input data (the second dim) should be less"
                                  " than the model batch size, otherwise some inputs will"
                                  " be ignored.")
            elif isinstance(feature_data, list):
                for elem in feature_data:
                    invalidInputError(isinstance(elem, np.ndarray),
                                      "Each element in the x list should be a ndarray,"
                                      " but get " + elem.__class__.__name__)
                    invalidInputError(elem.shape[0] <= batch_size,
                                      "The batch size of each input data (the second dim) should"
                                      " be less than the model batch size, otherwise some inputs"
                                      " will be ignored.")
            else:
                invalidInputError(False,
                                  "x in each shard should be a ndarray or a list of ndarray.")
            return feature_data

        if isinstance(data, DataFrame):
            from bigdl.orca.learn.utils import dataframe_to_xshards
            from bigdl.orca import OrcaContext
            OrcaContext._shard_size = batch_size
            schema = data.schema
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict")
            OrcaContext._shard_size = None
            transformed_data = xshards.transform_shard(predict_transform, batch_size)
            result_rdd = transformed_data.rdd.mapPartitions(lambda iter: partition_inference(iter))

            def divide(data):
                if len(data) > 1:
                    # add batch dim
                    result = [list(np.expand_dims(output, axis=0).tolist()
                                   for output in single_result) for single_result in zip(*data)]
                    return result
                else:
                    data

            def add_predict_rdd_to_dataframe(df, prediction_rdd):
                from pyspark.sql import Row
                from pyspark.sql.types import StructType, StructField, FloatType, ArrayType

                # Deal with types
                result_struct = []
                for key, shape in self.output_dict.items():
                    struct_type = FloatType()
                    for _ in range(len(shape)):
                        struct_type = ArrayType(struct_type)
                    result_struct.append(StructField(key, struct_type))

                def combine(pair):
                    # multi-output
                    if len(result_struct) > 1:
                        row = Row(*([pair[0][col] for col in pair[0].__fields__] + pair[1]))
                    # single output
                    else:
                        # TODO: scalar may cause error?
                        row = Row(*([pair[0][col] for col in pair[0].__fields__] + [pair[1]]))
                    return row

                combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
                schema = StructType(df.schema.fields + result_struct)
                result_df = combined_rdd.toDF(schema)
                return result_df

            return add_predict_rdd_to_dataframe(data, result_rdd.flatMap(lambda data: divide(data)))
        elif isinstance(data, SparkXShards):
            transformed_data = data.transform_shard(predict_transform, batch_size)
            result_rdd = transformed_data.rdd.mapPartitions(lambda iter: partition_inference(iter))

            def update_result_shard(data):
                shard, y = data
                shard["prediction"] = y
                return shard
            return SparkXShards(result_rdd)
        elif isinstance(data, (np.ndarray, list)):
            if isinstance(data, np.ndarray):
                split_num = math.ceil(len(data)/batch_size)
                arrays = np.array_split(data, split_num)
                num_slices = min(split_num, self.node_num)
                data_rdd = sc.parallelize(arrays, numSlices=num_slices)
            elif isinstance(data, list):
                flattened = nest.flatten(data)
                data_length = len(flattened[0])
                data_to_be_rdd = []
                split_num = math.ceil(flattened[0].shape[0]/batch_size)
                num_slices = min(split_num, self.node_num)
                for i in range(split_num):
                    data_to_be_rdd.append([])
                for x in flattened:
                    invalidInputError(isinstance(x, np.ndarray),
                                      "the data in the data list should be ndarrays,"
                                      " but get " + x.__class__.__name__)
                    invalidInputError(len(x) == data_length,
                                      "the ndarrays in data must all have the same"
                                      " size in first dimension, got first ndarray"
                                      " of size {} and another {}".format(data_length, len(x)))
                    x_parts = np.array_split(x, split_num)
                    for idx, x_part in enumerate(x_parts):
                        data_to_be_rdd[idx].append(x_part)

                data_to_be_rdd = [nest.pack_sequence_as(data, shard) for shard in data_to_be_rdd]
                data_rdd = sc.parallelize(data_to_be_rdd, numSlices=num_slices)

            print("Partition number: ", data_rdd.getNumPartitions())
            result_rdd = data_rdd.mapPartitions(lambda iter: partition_inference(iter))
            result_arr_list = result_rdd.collect()
            result_arr = None
            if isinstance(result_arr_list[0], list):
                result_arr = [np.concatenate([r[i] for r in result_arr_list], axis=0)
                              for i in range(len(result_arr_list[0]))]
            elif isinstance(result_arr_list[0], np.ndarray):
                result_arr = np.concatenate(result_arr_list, axis=0)
            return result_arr
        else:
            invalidInputError(False,
                              "Only XShards, Spark DataFrame, a numpy array and a list of numpy"
                              " arrays are supported as input data, but"
                              " get " + data.__class__.__name__)

    def evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None):
        """
        Evaluate is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_model(self):
        """
        Get_model is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def save(self, model_path):
        """
        Save is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def load(self, model_path):
        """
        Load an openVINO model.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :return:
        """
        self.node_num, self.core_num = get_node_and_core_number()
        invalidInputError(isinstance(model_path, str), "The model_path should be string.")
        invalidInputError(os.path.exists(model_path), "The model_path should be exist.")
        with open(model_path, 'rb') as file:
            self.model_bytes = file.read()

        with open(model_path[:model_path.rindex(".")] + ".bin", 'rb') as file:
            self.weight_bytes = file.read()

        ie = IECore()
        config = {'CPU_THREADS_NUM': str(self.core_num)}
        ie.set_config(config, 'CPU')
        net = ie.read_network(model=self.model_bytes,
                              weights=self.weight_bytes, init_from_buffer=True)
        self.inputs = list(net.input_info.keys())
        self.output_dict = {k: v.shape for k, v in net.outputs.items()}

    def set_tensorboard(self, log_dir, app_name):
        """
        Set_tensorboard is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def clear_gradient_clipping(self):
        """
        Clear_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def set_constant_gradient_clipping(self, min, max):
        """
        Set_constant_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Set_l2_norm_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_train_summary(self, tag=None):
        """
        Get_train_summary is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_validation_summary(self, tag=None):
        """
        Get_validation_summary is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def load_orca_checkpoint(self, path, version):
        """
        Load_orca_checkpoint is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")
