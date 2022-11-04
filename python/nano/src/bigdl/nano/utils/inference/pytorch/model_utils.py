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
from typing import Any, Sequence
from bigdl.nano.pytorch.lightning import LightningModule
import inspect
from torch.utils.data import DataLoader
import torch
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_11, TORCH_VERSION_LESS_1_12
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule


def get_forward_args(model):
    forward_args = inspect.getfullargspec(model.forward).args[1:]
    if isinstance(model, LightningModule):
        if not isinstance(model, AcceleratedLightningModule):
            # forward param list for compiled model
            forward_args = get_forward_args(model.model)
    return forward_args


def get_input_example(model, input_sample, forward_args):
    if isinstance(input_sample, DataLoader):
        input_sample = next(iter(input_sample))
        if isinstance(input_sample, Sequence):
            input_sample = tuple(list(input_sample)[:len(forward_args)])
    elif input_sample is None:
        if getattr(model, "example_input_array", None) is not None:
            input_sample = model.example_input_array
        elif getattr(model, "trainer", None):
            # the default implementation of model.test_dataloader/train_dalaloader/val_dataloader
            # /predict_dataloader will throw an exception in pytorch lightning 1.6
            for dataloader_fn in [model.test_dataloader, model.train_dataloader,
                                  model.val_dataloader]:
                try:
                    dataloader = dataloader_fn()
                    input_sample = next(iter(input_sample))
                    if isinstance(input_sample, Sequence):
                        input_sample = tuple(list(input_sample)[:len(forward_args)])
                    break
                except Exception as _e:
                    pass

            if input_sample is None:
                try:
                    predict_dataloader = model.predict_dataloader()
                    input_sample = tuple(next(iter(predict_dataloader)))
                except Exception as _e:
                    pass
        else:
            invalidInputError(False,
                              "You must specify an input_sample or call `Trainer.fit` "
                              "on the model first to use `eval_openvino`")

    model.example_input_array = input_sample
    return input_sample


def export_to_onnx(model, input_sample=None, onnx_path="model.onnx", dynamic_axes=True, **kwargs):
    '''
    Internal function to export pytorch model as onnx.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save onnx model file.
    :param dynamic_axes: dict or boolean, default: True. By default the exported model will
           have the first dim of each input as a dynamic batch_size. If dynamic_axes=False, the
           exported model will have the shapes of all input and output tensors set to exactly match
           those given in input_sample. To specify axes of tensors as dynamic (i.e. known only at
           run-time), set dynamic_axes to a dict with schema:
           KEY (str): an input or output name. Each name must also be provided in input_names or
           output_names.
           VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
           list, each element is an axis index.
    :param **kwargs: will be passed to torch.onnx.export function.
    '''
    forward_args = get_forward_args(model)
    input_sample = get_input_example(model, input_sample, forward_args)
    invalidInputError(input_sample is not None,
                      'You should implement at least one of model.test_dataloader, '
                      'model.train_dataloader, model.val_dataloader and '
                      'model.predict_dataloader, '
                      'or set one of input_sample and model.example_input_array')
    if isinstance(dynamic_axes, dict):
        pass
    elif dynamic_axes is True:
        dynamic_axes = {}
        for arg in forward_args:
            dynamic_axes[arg] = {0: 'batch_size'}  # set all dim0 to be dynamic
    else:
        dynamic_axes = {}

    # set opset_version according to torch version
    if TORCH_VERSION_LESS_1_11:
        opset_version = 12
    else:
        opset_version = 15

    default_onnx_export_args = {'export_params': True,
                                'opset_version': opset_version,
                                'do_constant_folding': True,
                                'input_names': forward_args,
                                'dynamic_axes': dynamic_axes,
                                }
    default_onnx_export_args.update(kwargs)
    torch.onnx.export(model, input_sample, onnx_path, **default_onnx_export_args)
