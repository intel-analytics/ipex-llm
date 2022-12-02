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

import warnings

from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.keras.engine.topology import KerasNet
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.log4Error import invalidInputError

from typing import (
    Any,
    Union,
    Optional,
    List
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray
    from pyspark import RDD
    from pyspark.context import SparkContext
    from bigdl.dllib.utils.common import JTensor


class InferenceModel(JavaValue):
    """
    Model for thread-safe inference.
    To do inference, you need to first initiate an InferenceModel instance, then call
    load|load_caffe|load_openvino to load a pre-trained model, and finally call predict.

    # Arguments
    supported_concurrent_num: Int. How many concurrent threads to invoke. Default is 1.
    """

    def __init__(self,
                 supported_concurrent_num: int = 1,
                 bigdl_type: str = "float") -> None:
        super(InferenceModel, self).__init__(None, bigdl_type, supported_concurrent_num)

    def load_bigdl(self, model_path: str, weight_path: Optional[str] = None) -> None:
        """
        Load a pre-trained BigDL model.

        :param model_path: String. The file path to the model.
        :param weight_path: String. The file path to the weights if any. Default is None.
        """
        callZooFunc(self.bigdl_type, "inferenceModelLoadBigDL",
                    self.value, model_path, weight_path)

    # deprecated in "0.8.0"
    def load(self, model_path: str, weight_path: Optional[str] = None) -> None:
        """
        Load a pre-trained BigDL model.

        :param model_path: String. The file path to the model.
        :param weight_path: String. The file path to the weights if any. Default is None.
        """
        warnings.warn("deprecated in 0.8.0")
        callZooFunc(self.bigdl_type, "inferenceModelLoad",
                    self.value, model_path, weight_path)

    def load_caffe(self, model_path: str, weight_path: str) -> None:
        """
        Load a pre-trained Caffe model.

        :param model_path: String. The file path to the prototxt file.
        :param weight_path: String. The file path to the Caffe model.
        """
        callZooFunc(self.bigdl_type, "inferenceModelLoadCaffe",
                    self.value, model_path, weight_path)

    def load_openvino(self, model_path: str, weight_path: str,
                      batch_size: int = 0) -> None:
        """
        Load an OpenVINI IR.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :param weight_path: String. The file path to the OpenVINO IR bin file.
        :param batch_size: Int. Set batch Size, default is 0 (use default batch size).
        """
        callZooFunc(self.bigdl_type, "inferenceModelLoadOpenVINO",
                    self.value, model_path, weight_path, batch_size)

    # TODO: Add UT
    def load_openvino_ng(self, model_path: str, weight_path: str,
                         batch_size: int = 0) -> None:
        """
        Load an OpenVINI IR.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :param weight_path: String. The file path to the OpenVINO IR bin file.
        :param batch_size: Int. Set batch Size, default is 0 (use default batch size).
        """
        callZooFunc(self.bigdl_type, "inferenceModelLoadOpenVINONg",
                    self.value, model_path, weight_path, batch_size)

    # TODO: Add UT
    def load_tensorflow(self, model_path: str,
                        model_type: str = "frozenModel",
                        intra_op_parallelism_threads: int = 1,
                        inter_op_parallelism_threads: int = 1,
                        use_per_session_threads: bool = True) -> None:
        """
        Load a TensorFlow model using tensorflow.

        :param model_path: String. The file path to the TensorFlow model.
        :param model_type: String. The type of the tensorflow model file. Default is "frozenModel"
        :param intra_op_parallelism_threads: Int. The number of intraOpParallelismThreads.
                                             Default is 1.
        :param inter_op_parallelism_threads: Int. The number of interOpParallelismThreads.
                                             Default is 1.
        :param use_per_session_threads: Boolean. Whether to use perSessionThreads. Default is True.
        """
        callZooFunc(self.bigdl_type, "inferenceModelLoadTensorFlow",
                    self.value, model_path, model_type, intra_op_parallelism_threads,
                    inter_op_parallelism_threads, use_per_session_threads)

    # TODO: Add UT
    def load_tensorflow_graph(self, model_path: str,
                              model_type: str,
                              inputs: List[str],
                              outputs: List[str],
                              intra_op_parallelism_threads: int = 1,
                              inter_op_parallelism_threads: int = 1,
                              use_per_session_threads: bool = True) -> None:
        """
        Load a TensorFlow model using tensorflow.

        :param model_path: String. The file path to the TensorFlow model.
        :param model_type: String. The type of the tensorflow model file: "frozenModel" or
         "savedModel".
        :param inputs: Array[String]. the inputs of the model.
        inputs outputs: Array[String]. the outputs of the model.
        :param intra_op_parallelism_threads: Int. The number of intraOpParallelismThreads.
                                                Default is 1.
        :param inter_op_parallelism_threads: Int. The number of interOpParallelismThreads.
                                                Default is 1.
        :param use_per_session_threads: Boolean. Whether to use perSessionThreads. Default is True.
           """
        callZooFunc(self.bigdl_type, "inferenceModelLoadTensorFlow",
                    self.value, model_path, model_type,
                    inputs, outputs, intra_op_parallelism_threads,
                    inter_op_parallelism_threads, use_per_session_threads)

    # TODO: Add UT
    def load_torch(self, model_path: str) -> None:
        """
        Load a pytorch model.

        :param model_path: the path of saved pytorch model
           """
        invalidInputError(isinstance(model_path, str), "model_path should be string")
        import io
        import torch
        from bigdl.orca.torch import zoo_pickle_module
        model = torch.load(model_path, pickle_module=zoo_pickle_module)
        bys = io.BytesIO()
        torch.save(model, bys, pickle_module=zoo_pickle_module)
        callZooFunc(self.bigdl_type, "inferenceModelLoadPytorch",
                    self.value, bys.getvalue())

    # TODO: Add UT for different inputs
    def predict(self, inputs: Union["ndarray", List["ndarray"], "JTensor", List["JTensor"]]) \
            -> Union["ndarray", List["ndarray"]]:
        """
        Do prediction on inputs.

        :param inputs: A numpy array or a list of numpy arrays or JTensor or a list of JTensors.
        """
        jinputs, input_is_table = Layer.check_input(inputs)
        output = callZooFunc(self.bigdl_type,
                             "inferenceModelPredict",
                             self.value,
                             jinputs,
                             input_is_table)
        return KerasNet.convert_output(output)

    # TODO: Add UT
    def distributed_predict(self, inputs: "RDD[Any]", sc: "SparkContext") -> "RDD[Any]":
        data_type = inputs.map(lambda x: x.__class__.__name__).first()
        input_is_table = False
        if data_type == "list":
            input_is_table = True
        jinputs = inputs.map(lambda x: Layer.check_input(x)[0])

        output = callZooFunc(self.bigdl_type,
                             "inferenceModelDistriPredict",
                             self.value,
                             sc,
                             jinputs,
                             input_is_table)
        return output.map(lambda x: KerasNet.convert_output(x))
