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
        onnx_path = model
        if isinstance(model, tf.keras.Model):
            onnx_path = 'tmp.onnx'
            if not isinstance(input_sample, (tuple, list)):
                input_sample = (input_sample, )
            tf2onnx.convert.from_keras(model, input_signature=input_sample,
                                       output_path=onnx_path, **export_kwargs)
        AcceleratedKerasModel.__init__(self, None)
        ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)
        if isinstance(model, tf.keras.Model) and os.path.exists(onnx_path):
            os.remove(onnx_path)

    def on_forward_start(self, inputs):
        if self.ortsess is None:
            invalidInputError(False,
                              "Please create an instance by KerasONNXRuntimeModel()")
        inputs = self.tensors_to_numpy(inputs)
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs)
        return outputs
