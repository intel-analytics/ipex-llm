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


def PytorchOpenVINOModel(model, input_sample=None, precision='fp32',
                         thread_num=None, device='CPU',
                         dynamic_axes=True, logging=True,
                         config=None, output_tensors=True, **kwargs):
    """
    Create a OpenVINO model from pytorch.

    :param model: Pytorch model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                         model is a LightningModule with any dataloader attached, defaults to None.
    :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                      defaults to 'fp32'.
    :param thread_num: a int represents how many threads(cores) is needed for
                       inference. default: None.
    :param device: (optional) A string represents the device of the inference. Default to 'CPU'.
                   'CPU', 'GPU' and 'VPUX' are supported for now.
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
    :param output_tensors: boolean, default to True and output of the model will be Tensors.
                           If output_tensors=False, output of the OpenVINO model will be ndarray.
    :param **kwargs: will be passed to torch.onnx.export function or model optimizer function.
    :return: PytorchOpenVINOModel model for OpenVINO inference.
    """
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel(model=model,
                                input_sample=input_sample,
                                precision=precision,
                                thread_num=thread_num,
                                device=device,
                                dynamic_axes=dynamic_axes,
                                logging=logging,
                                config=config,
                                output_tensors=output_tensors,
                                **kwargs)


def load_openvino_model(path, framework='pytorch', device=None, cache_dir=None, shapes=None):
    """
    Load an OpenVINO model for inference from directory.

    :param path: Path to model to be loaded.
    :param framework: Only support pytorch and tensorflow now
    :param device: A string represents the device of the inference.
    :param cache_dir: A directory for OpenVINO to cache the model. Default to None.
    :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',
               '[1,3,224,224]'. This parameter affect model Parameter shape, can be
               dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.
               Only valid for openvino model, otherwise will be ignored.
    :return: PytorchOpenVINOModel model for OpenVINO inference.
    """
    if cache_dir is not None:
        from pathlib import Path
        Path(cache_dir).mkdir(exist_ok=True)

    if framework == 'pytorch':
        from .pytorch.model import PytorchOpenVINOModel
        return PytorchOpenVINOModel._load(path, device=device, cache_dir=cache_dir, shapes=shapes)
    elif framework == 'tensorflow':
        from .tf.model import KerasOpenVINOModel
        return KerasOpenVINOModel._load(path, device=device, cache_dir=cache_dir, shapes=shapes)
    else:
        invalidInputError(False,
                          "The value {} for framework is not supported."
                          " Please choose from 'pytorch'/'tensorflow'.")


def KerasOpenVINOModel(model, input_spec=None, precision='fp32',
                       thread_num=None, device='CPU', config=None,
                       logging=True, **kwargs):
    """
    Create a OpenVINO model from Keras.

    :param model: Keras model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining
                       the shape/dtype of the input
    :param precision: Global precision of model, supported type: 'fp32', 'fp16',
                      defaults to 'fp32'.
    :param thread_num: a int represents how many threads(cores) is needed for
                       inference. default: None.
    :param device: (optional) A string represents the device of the inference. Default to 'CPU'.
                   'CPU', 'GPU' and 'VPUX' are supported for now.
    :param config: The config to be inputted in core.compile_model.
    :param logging: whether to log detailed information of model conversion. default: True.
    :param **kwargs: will be passed to model optimizer function.
    :return: KerasOpenVINOModel model for OpenVINO inference.
    """
    from .tf.model import KerasOpenVINOModel
    return KerasOpenVINOModel(model=model,
                              input_spec=input_spec,
                              precision=precision,
                              thread_num=thread_num,
                              device=device,
                              config=config,
                              logging=logging,
                              **kwargs)


def OpenVINOModel(model, device='CPU'):
    from .core.model import OpenVINOModel
    return OpenVINOModel(model, device)
