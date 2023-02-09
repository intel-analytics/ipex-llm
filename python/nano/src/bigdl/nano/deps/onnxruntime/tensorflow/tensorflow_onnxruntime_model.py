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
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import tensorflow as tf
from bigdl.nano.utils.common import get_default_args
from bigdl.nano.tf.utils import KERAS_VERSION_LESS_2_10, fake_tensor_from_spec
from bigdl.nano.tf.model import AcceleratedKerasModel
from bigdl.nano.utils.common import invalidInputError

from ..core.onnxruntime_model import ONNXRuntimeModel
import onnxruntime  # should be put behind core's import

try:
    import tf2onnx
except ImportError:
    invalidInputError(False, "Failed to import 'tf2onnx', you should install it first.")


class KerasONNXRuntimeModel(ONNXRuntimeModel, AcceleratedKerasModel):
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
        AcceleratedKerasModel.__init__(self, None)
        with TemporaryDirectory() as tmpdir:
            if isinstance(model, tf.keras.Model):
                onnx_path = os.path.join(tmpdir, "tmp.onnx")
                if input_spec is None:
                    input_spec = tf.TensorSpec(model.input_shape, model.dtype)
                if hasattr(model, "output_shape"):
                    self._output_shape = model.output_shape
                elif isinstance(input_spec, (tuple, list)):
                    self._output_shape = model.compute_output_shape((i.shape for i in input_spec))
                else:
                    self._output_shape = model.compute_output_shape(input_spec.shape)
                while isinstance(self._output_shape, list) and len(self._output_shape) == 1:
                    self._output_shape = self._output_shape[0]
                if not isinstance(input_spec, (tuple, list)):
                    input_spec = (input_spec, )
                tf2onnx.convert.from_keras(model, input_signature=input_spec,
                                           output_path=onnx_path, **export_kwargs)
                # save the nesting level of inference output
                fake_inputs = [fake_tensor_from_spec(spec) for spec in input_spec]
                fake_outputs = model(*fake_inputs)
                self._nesting_level = 0
                while isinstance(fake_outputs, (tuple, list)) and len(fake_outputs) == 1:
                    self._nesting_level += 1
                    fake_outputs = fake_outputs[0]

                self._inputs_dtypes = [inp.dtype for inp in input_spec]
                self._default_kwargs = get_default_args(model.call)
                if KERAS_VERSION_LESS_2_10:
                    self._call_fn_args_backup = model._call_fn_args
                else:
                    from keras.utils import tf_inspect
                    self._call_fn_args_backup = tf_inspect.getargspec(model.call).args[1:]
            else:
                onnx_path = model
            ONNXRuntimeModel.__init__(self, onnx_path, session_options=onnxruntime_session_options)

    def __call__(self, *args, **kwargs):
        inputs = list(args)
        # add arguments' default values into `kwargs`
        for name, value in self._default_kwargs.items():
            if name not in kwargs:
                kwargs[name] = value
        for param in self._call_fn_args_backup[len(inputs):len(self._forward_args)]:
            inputs.append(kwargs[param])
        outputs = self.call(*inputs)
        for _i in range(self._nesting_level):
            outputs = [outputs]
        return outputs

    def on_forward_start(self, inputs):
        if self.ortsess is None:
            invalidInputError(False,
                              "Please create an instance by KerasONNXRuntimeModel()")
        inputs = [self.tensors_to_numpy(inp, input_dtype)
                  for inp, input_dtype in zip(inputs, self._inputs_dtypes)]
        return inputs

    def on_forward_end(self, outputs):
        outputs = self.numpy_to_tensors(outputs)
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
            attrs = pickle.load(f)
        for attr_name, attr_value in attrs.items():
            setattr(model, attr_name, attr_value)
        if os.path.exists(Path(path) / status['compile_path']):
            with open(Path(path) / status['compile_path'], "rb") as f:
                kwargs = pickle.load(f)
                model.compile(**kwargs)
        return model

    def _save_model(self, path):
        onnx_path = Path(path) / self.status['onnx_path']
        super()._save_model(onnx_path)
        attrs = {"_default_kwargs": self._default_kwargs,
                 "_call_fn_args_backup": self._call_fn_args_backup,
                 "_inputs_dtypes": self._inputs_dtypes,
                 "_nesting_level": self._nesting_level,
                 "_output_shape": self._output_shape}
        with open(Path(path) / self.status['attr_path'], "wb") as f:
            pickle.dump(attrs, f)
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
            with open(Path(path) / self.status['compile_path'], "wb") as f:
                pickle.dump(kwargs, f)
