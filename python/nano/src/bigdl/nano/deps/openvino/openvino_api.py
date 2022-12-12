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
from functools import partial


def PytorchOpenVINOModel(model, input_sample=None, thread_num=None,
                         dynamic_axes=True, logging=True,
                         config=None, **export_kwargs):
    """
    Create a OpenVINO model from pytorch.

    :param model: Pytorch model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                         model is a LightningModule with any dataloader attached, defaults to None.
    :param thread_num: a int represents how many threads(cores) is needed for
                       inference. default: None.
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
    :param logging: whether to log detailed information of model conversion. default: True.
    :param config: The config to be inputted in core.compile_model.
    :param **export_kwargs: will be passed to torch.onnx.export function.
    :return: PytorchOpenVINOModel model for OpenVINO inference.
    """
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel(model=model,
                                input_sample=input_sample,
                                thread_num=thread_num,
                                dynamic_axes=dynamic_axes,
                                logging=logging,
                                config=config,
                                **export_kwargs)


def load_openvino_model(path):
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel._load(path)


def KerasOpenVINOModel(model, thread_num=None, config=None, logging=True):
    """
    Create a OpenVINO model from Keras.

    :param model: Keras model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param thread_num: a int represents how many threads(cores) is needed for
                       inference. default: None.
    :param config: The config to be inputted in core.compile_model.
    :param logging: whether to log detailed information of model conversion. default: True.
    :return: KerasOpenVINOModel model for OpenVINO inference.
    """
    from .tf.model import KerasOpenVINOModel
    return KerasOpenVINOModel(model, thread_num=thread_num, config=config, logging=logging)


def OpenVINOModel(model, device='CPU'):
    from .core.model import OpenVINOModel
    return OpenVINOModel(model, device)
