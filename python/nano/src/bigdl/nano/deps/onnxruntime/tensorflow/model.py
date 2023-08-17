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
from bigdl.nano.utils.common import SafePickle
from pathlib import Path
from typing import Sequence, Any, Union, Dict
from tempfile import TemporaryDirectory
import tensorflow as tf
from bigdl.nano.utils.tf import convert_all
from bigdl.nano.tf.model import KerasOptimizedModel
from bigdl.nano.utils.common import invalidInputError

from ..core.onnxruntime_model import ONNXRuntimeModel
import onnxruntime  # should be put behind core's import

try:
    import tf2onnx
except ImportError:
    invalidInputError(False, "Failed to import 'tf2onnx', you should install it first.")


class KerasONNXRuntimeModel(ONNXRuntimeModel, KerasOptimizedModel):
    '''
        This is the accelerated model for tensorflow and onnxruntime.
    '''
    def __init__(self, model, input_spec=None, onnxruntime_session_options=None,
                 **export_kwargs):
        """
        Create a ONNX Runtime model from tensorflow.

        :param model: 1. Keras model to be converted to ONNXRuntime for inference
                      2. Path to ONNXRuntime saved model
        :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining
                           the shape/dtype of the input
        :param onnxruntime_session_options: will be passed to tf2onnx.convert.from_keras function
        """
        KerasOptimizedModel.__init__(self)
        with TemporaryDirectory() as tmpdir:
            if isinstance(model, tf.keras.Model):
                invalidInputError(hasattr(model, "input_shape") or input_spec is not None,
                                  "Subclassed model must specify `input_spec` parameter.")
                self._mode = "arg"
                if input_spec is None:
                    input_spec = tf.TensorSpec(model.input_shape, model.dtype)
                elif isinstance(input_spec, dict):
                    input_spec = [tf.TensorSpec.from_spec(spec, name=name)
                                  for name, spec in input_spec.items()]
                    self._mode = "kwarg"
                if isinstance(input_spec, tf.TensorSpec):
                    input_spec = (input_spec, )

                if self._mode == "arg":
                    self._inputs_dtypes = [spec.dtype.as_numpy_dtype for spec in input_spec]
                else:
                    self._inputs_dtypes = {spec.name: spec.dtype.as_numpy_dtype
                                           for spec in input_spec}

                onnx_path = os.path.join(tmpdir, "tmp.onnx")
                tf2onnx.convert.from_keras(model, input_signature=input_spec,
                                           output_path=onnx_path, **export_kwargs)

            else:
                onnx_path = model
            ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)

    def preprocess(self, args: Sequence[Any], kwargs: Dict[str, Any]):
        invalidInputError(self.ortsess is not None,
                          "Please create an instance by InferenceOptimizer.trace()")
        # todo: add argument check
        inputs = args if self._mode == "arg" else kwargs
        inputs = convert_all(inputs, "numpy", self._inputs_dtypes)
        return inputs

    def forward(self, inputs: Union[Sequence[Any], Dict[str, Any]]):
        if isinstance(inputs, Sequence):
            return self.forward_step(*inputs)
        else:
            return self.ortsess.run(None, inputs)

    def postprocess(self, outputs: Sequence[Any]):
        outputs = convert_all(outputs, types="tf", dtypes=tf.float32)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def status(self):
        status = super().status
        status.update({"onnx_path": 'onnx_saved_model.onnx',
                       "attr_path": "onnx_saved_model_attr.pkl",
                       "compile_path": "onnx_saved_model_compile.pkl",
                       "intra_op_num_threads": self.session_options.intra_op_num_threads,
                       "inter_op_num_threads": self.session_options.inter_op_num_threads})
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
        onnxruntime_session_options = onnxruntime.SessionOptions()
        onnxruntime_session_options.intra_op_num_threads = status['intra_op_num_threads']
        onnxruntime_session_options.inter_op_num_threads = status['inter_op_num_threads']
        model = KerasONNXRuntimeModel(model=str(onnx_path),
                                      input_sample=None,
                                      onnxruntime_session_options=onnxruntime_session_options)
        with open(Path(path) / status['attr_path'], "rb") as f:
            attrs = SafePickle.load(f)
        for attr_name, attr_value in attrs.items():
            setattr(model, attr_name, attr_value)
        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = SafePickle.load(f)
                model.compile(**kwargs)
        return model

    def _save(self, path, compression="fp32"):
        path = Path(path)
        path.mkdir(exist_ok=True)
        self._dump_status(path)

        super()._save_model(path / self.status['onnx_path'])

        attrs = {"_mode": self._mode,
                 "_inputs_dtypes": self._inputs_dtypes}
        with open(path / self.status['attr_path'], "wb") as f:
            SafePickle.dump(attrs, f)

        if self._is_compiled:
            kwargs = {"run_eagerly": self._run_eagerly,
                      "steps_per_execution": int(self._steps_per_execution)}
            if self.compiled_loss is not None:
                kwargs["loss"] = self.compiled_loss._user_losses
                kwargs["loss_weights"] = self.compiled_loss._user_loss_weights
            if self.compiled_metrics is not None:
                user_metric = self.compiled_metrics._user_metrics
                if isinstance(user_metric, (list, tuple)):
                    kwargs["metrics"] = [m._name for m in user_metric]
                else:
                    kwargs["metrics"] = user_metric._name
                weighted_metrics = self.compiled_metrics._user_weighted_metrics
                if weighted_metrics is not None:
                    if isinstance(weighted_metrics, (list, str)):
                        kwargs["weighted_metrics"] = [m._name for m in weighted_metrics]
                    else:
                        kwargs["weighted_metrics"] = weighted_metrics._name
            with open(path / self.status['compile_path'], "wb") as f:
                SafePickle.dump(kwargs, f)
