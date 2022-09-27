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
                         logging=True, **export_kwargs):
    """
    Create a OpenVINO model from pytorch.

    :param model: Pytorch model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                         model is a LightningModule with any dataloader attached, defaults to None.
    :param thread_num: a int represents how many threads(cores) is needed for
                       inference. default: None.
    :param logging: whether to log detailed information of model conversion. default: True.
    :param **export_kwargs: will be passed to torch.onnx.export function.
    :return: PytorchOpenVINOModel model for OpenVINO inference.
    """
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel(model, input_sample, thread_num, logging, **export_kwargs)


def load_openvino_model(path):
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel._load(path)


def KerasOpenVINOModel(model, input_sample=None):
    """
    Create a OpenVINO model from Keras.

    :param model: Keras model to be converted to OpenVINO for inference or
                  path to Openvino saved model.
    :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                         model is a LightningModule with any dataloader attached, defaults to None
    :return: KerasOpenVINOModel model for OpenVINO inference.
    """
    from .tf.model import KerasOpenVINOModel
    return KerasOpenVINOModel(model)


def OpenVINOModel(model, device='CPU'):
    from .core.model import OpenVINOModel
    return OpenVINOModel(model, device)
