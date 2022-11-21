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
from pathlib import Path
from tempfile import TemporaryDirectory
import tensorflow as tf
from bigdl.nano.utils.inference.tf.model import AcceleratedKerasModel
from bigdl.nano.utils.log4Error import invalidInputError

from ..core.onnxruntime_model import ONNXRuntimeModel

try:
    import tf2onnx
except ImportError:
    invalidInputError(False, "Failed to import 'tf2onnx', you should install it first.")


class KerasONNXRuntimeModel(ONNXRuntimeModel, AcceleratedKerasModel):
    '''
        This is the accelerated model for tensorflow and onnxruntime.
    '''
    def __init__(self, model, input_sample, onnxruntime_session_options=None,
                 **export_kwargs):
        """
        Create a ONNX Runtime model from tensorflow.

        :param model: 1. Keras model to be converted to ONNXRuntime for inference
                      2. Path to ONNXRuntime saved model
        :param input_sample: a (tuple or list of) tf.TensorSpec or numpy array defining
            the shape/dtype of the input
        :param onnxruntime_session_options: will be passed to tf2onnx.convert.from_keras function
        """
        with TemporaryDirectory() as tmpdir:
            if isinstance(model, tf.keras.Model):
                onnx_path = os.path.join(tmpdir, "tmp.onnx")
                if not isinstance(input_sample, (tuple, list)):
                    input_sample = (input_sample, )
                tf2onnx.convert.from_keras(model, input_signature=input_sample,
                                           output_path=onnx_path, **export_kwargs)
            else:
                onnx_path = model
            AcceleratedKerasModel.__init__(self, None)
            ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)

    def on_forward_start(self, inputs):
        if self.ortsess is None:
            invalidInputError(False,
                              "Please create an instance by KerasONNXRuntimeModel()")
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs)
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"onnx_path": 'onnx_saved_model.onnx'})
        return status

    @staticmethod
    def _load(path):
        """
        Load an ONNX model for inference from directory.

        :param path: Path to model to be loaded.
        :return: KerasONNXRuntimeModel model for ONNX Runtime inference.
        """
        status = KerasONNXRuntimeModel._load_status(path)
        if status.get('onnx_path', None):
            onnx_path = Path(status['onnx_path'])
            invalidInputError(onnx_path.suffix == '.onnx',
                              "Path of onnx model must be with '.onnx' suffix.")
        else:
            invalidInputError(False,
                              "nano_model_meta.yml must specify 'onnx_path' for loading.")
        onnx_path = Path(path) / status['onnx_path']
        return KerasONNXRuntimeModel(str(onnx_path), None)

    def _save_model(self, path):
        onnx_path = Path(path) / self.status['onnx_path']
        super()._save_model(onnx_path)
