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
from zoo.pipeline.inference import InferenceModel
from zoo.orca.data import SparkXShards
from zoo import get_node_and_core_number
from zoo.util import nest
from zoo.common.nncontext import init_nncontext

import numpy as np


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        pass

    def save(self, model_path):
        pass

    def load(self, model_path, **kwargs):
        pass

    @staticmethod
    def from_openvino(*, model_path, batch_size=0):
        """
        Load an openVINO Estimator.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :param batch_size: Int. Set batch Size, default is 0 (use default batch size).
        """
        return OpenvinoEstimatorWrapper(model_path=model_path, batch_size=batch_size)


class OpenvinoEstimatorWrapper(Estimator):
    def __init__(self,
                 *,
                 model_path,
                 batch_size=0):
        self.node_num, self.core_num = get_node_and_core_number()
        self.path = model_path
        if batch_size != 0:
            self.batch_size = batch_size
        else:
            import xml.etree.ElementTree as ET
            tree = ET.parse(model_path)
            root = tree.getroot()
            shape_item = root.find('./layers/layer/output/port/dim[1]')
            if shape_item is None:
                raise ValueError("Invalid openVINO IR xml file, please check your model_path")
            self.batch_size = int(shape_item.text)
        self.model = InferenceModel(supported_concurrent_num=self.core_num)
        self.model.load_openvino(model_path=model_path,
                                 weight_path=model_path[:model_path.rindex(".")] + ".bin",
                                 batch_size=batch_size)

    def fit(self, data, epochs, **kwargs):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        def predict_transform(dict_data, batch_size):
            assert isinstance(dict_data, dict), "each shard should be an dict"
            assert "x" in dict_data, "key x should in each shard"
            feature_data = dict_data["x"]
            if isinstance(feature_data, np.ndarray):
                assert feature_data.shape[1] <= batch_size, \
                    "The batch size of input data (the second dim) should be less than the model " \
                    "batch size, otherwise some inputs will be ignored."
            elif isinstance(feature_data, list):
                for elem in feature_data:
                    assert isinstance(elem, np.ndarray), "Each element in the x list should be " \
                                                         "a ndarray, but get " + \
                                                         elem.__class__.__name__
                    assert elem.shape[1] <= batch_size, "The batch size of each input data (the " \
                                                        "second dim) should be less than the " \
                                                        "model batch size, otherwise some inputs " \
                                                        "will be ignored."
            else:
                raise ValueError("x in each shard should be a ndarray or a list of ndarray.")
            return dict_data["x"]

        sc = init_nncontext()

        if isinstance(data, SparkXShards):
            assert sc is not None, "You should pass sc(spark context) if data is a XShards."
            from zoo.orca.learn.utils import convert_predict_to_xshard
            data = data.transform_shard(predict_transform, self.batch_size)
            result_rdd = self.model.distributed_predict(data.rdd, sc)
            return convert_predict_to_xshard(result_rdd)
        elif isinstance(data, (np.ndarray, list)):
            total_core_num = self.core_num * self.node_num
            if isinstance(data, np.ndarray):
                assert data.shape[1] <= self.batch_size, "The batch size of input data (the " \
                                                         "second dim) should be less than the " \
                                                         "model batch size, otherwise some " \
                                                         "inputs will be ignored."
                split_num = min(total_core_num, data.shape[0])
                arrays = np.array_split(data, split_num)
                data_rdd = sc.parallelize(arrays, numSlices=split_num)
            elif isinstance(data, list):
                flattened = nest.flatten(data)
                data_length = len(flattened[0])
                data_to_be_rdd = []
                split_num = min(total_core_num, flattened[0].shape[0])
                for i in range(split_num):
                    data_to_be_rdd.append([])
                for x in flattened:
                    assert isinstance(x, np.ndarray), "the data in the data list should be " \
                                                      "ndarrays, but get " + \
                                                      x.__class__.__name__
                    assert len(x) == data_length, \
                        "the ndarrays in data must all have the same size in first dimension" \
                        ", got first ndarray of size {} and another {}".format(data_length, len(x))
                    assert x.shape[1] <= self.batch_size, "The batch size of each input data (" \
                                                          "the second dim) should be less than " \
                                                          "the model batch size, otherwise some " \
                                                          "inputs will be ignored."
                    x_parts = np.array_split(x, split_num)
                    for idx, x_part in enumerate(x_parts):
                        data_to_be_rdd[idx].append(x_part)

                data_to_be_rdd = [nest.pack_sequence_as(data, shard) for shard in data_to_be_rdd]
                data_rdd = sc.parallelize(data_to_be_rdd, numSlices=split_num)

            result_rdd = self.model.distributed_predict(data_rdd, sc)
            result_arr_list = result_rdd.collect()
            result_arr = np.concatenate(result_arr_list, axis=0)
            return result_arr
        else:
            raise ValueError("Only XShards, a numpy array and a list of numpy arrays are supported "
                             "as input data, but get " + data.__class__.__name__)

    def evaluate(self, data, **kwargs):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def save(self, model_path):
        raise NotImplementedError

    def load(self, model_path, batch_size=0):
        """
        Load an openVINO model.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :param batch_size: Int. Set batch Size, default is 0 (use default batch size).
        :return:
        """
        self.node_num, self.core_num = get_node_and_core_number()
        self.path = model_path
        if batch_size != 0:
            self.batch_size = batch_size
        else:
            import xml.etree.ElementTree as ET
            tree = ET.parse(model_path)
            root = tree.getroot()
            shape_item = root.find('./layers/layer/output/port/dim[1]')
            if shape_item is None:
                raise ValueError("Invalid openVINO IR xml file, please check your model_path")
            self.batch_size = int(shape_item.text)
        self.model = InferenceModel(supported_concurrent_num=self.core_num)
        self.model.load_openvino(model_path=model_path,
                                 weight_path=model_path[:model_path.rindex(".")] + ".bin",
                                 batch_size=batch_size)
