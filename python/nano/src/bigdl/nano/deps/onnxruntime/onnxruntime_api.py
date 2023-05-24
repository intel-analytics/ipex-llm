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
from bigdl.nano.utils.common import invalidInputError


def PytorchONNXRuntimeModel(model, input_sample=None,
                            onnxruntime_session_options=None,
                            simplification=True,
                            dynamic_axes=True,
                            output_tensors=True,
                            **export_kwargs):
    """
        Create a ONNX Runtime model from pytorch.

        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference.
                      2. Path to ONNXRuntime saved model.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached,
                             defaults to None.
        :param onnxruntime_session_options: A session option for onnxruntime accelerator.
        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when
                               accelerator='onnxruntime', otherwise will be ignored. If this option
                               is set to True, new dependency 'onnxsim' need to be installed.
        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model
                             will have the first dim of each Tensor input as a dynamic batch_size.
                             If dynamic_axes=False, the exported model will have the shapes of all
                             input and output tensors set to exactly match those given in
                             input_sample. To specify axes of tensors as dynamic (i.e. known only
                             at run-time), set dynamic_axes to a dict with schema:

                             | KEY (str): an input or output name. Each name must also be provided
                             | in input_names or output_names.
                             |
                             | VALUE (dict or list): If a dict, keys are axis indices and values
                             | are axis names. If a list, each element is an axis index.

                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.
        :param output_tensors: boolean, default to True and output of the model will be Tensors.
                               If output_tensors=False, output of the ONNX model will be ndarray.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return: A PytorchONNXRuntimeModel instance
        """
    from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
    return PytorchONNXRuntimeModel(model, input_sample,
                                   onnxruntime_session_options=onnxruntime_session_options,
                                   simplification=simplification,
                                   dynamic_axes=dynamic_axes,
                                   output_tensors=output_tensors,
                                   **export_kwargs)


def load_onnxruntime_model(path, framework='pytorch'):
    if framework == 'pytorch':
        from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
        return PytorchONNXRuntimeModel._load(path)
    elif framework == 'tensorflow':
        from .tensorflow.model import KerasONNXRuntimeModel
        return KerasONNXRuntimeModel._load(path)
    else:
        invalidInputError(False,
                          "The value {} for framework is not supported."
                          " Please choose from 'pytorch'/'tensorflow'.")


def KerasONNXRuntimeModel(model, input_spec,
                          onnxruntime_session_options=None,
                          **export_kwargs):
    """
    Create a ONNX Runtime model from tensorflow.

    :param model: 1. Keras model to be converted to ONNXRuntime for inference
                  2. Path to ONNXRuntime saved model
    :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining
                       the shape/dtype of the input
    :param onnxruntime_session_options: will be passed to tf2onnx.convert.from_keras function
    """
    from .tensorflow.model import KerasONNXRuntimeModel
    return KerasONNXRuntimeModel(model, input_spec,
                                 onnxruntime_session_options=onnxruntime_session_options,
                                 **export_kwargs)
