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
from typing import Any
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
import inspect
from torch.utils.data import DataLoader
import torch
from bigdl.nano.utils.log4Error import invalidInputError


def get_forward_args(model):
    forward_args = inspect.getfullargspec(model.forward).args[1:]
    if isinstance(model, LightningModuleFromTorch):
        # forward param list for compiled model
        forward_args = get_forward_args(model.model)
    return forward_args


def get_input_example(model, input_sample):
    if isinstance(input_sample, DataLoader):
        # TODO: This assumpe the last output is y
        input_sample = tuple(next(iter(input_sample))[:-1])
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
                    # TODO: This assumpe the last output is y
                    input_sample = tuple(next(iter(dataloader)))[:-1]
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
    :param dynamic_axes: If we set the first dim of each input as a dynamic batch_size
    :param **kwargs: will be passed to torch.onnx.export function.
    '''
    input_sample = get_input_example(model, input_sample)
    invalidInputError(input_sample is not None,
                      'You should implement at least one of model.test_dataloader, '
                      'model.train_dataloader, model.val_dataloader and '
                      'model.predict_dataloader, '
                      'or set one of input_sample and model.example_input_array')
    forward_args = get_forward_args(model)
    if dynamic_axes:
        dynamic_axes = {}
        for arg in forward_args:
            dynamic_axes[arg] = {0: 'batch_size'}  # set all dim0 to be dynamic
    else:
        dynamic_axes = {}
    default_onnx_export_args = {'export_params': True,
                                'opset_version': 11,  # version = 11 by default
                                'do_constant_folding': True,
                                'input_names': forward_args,
                                'dynamic_axes': dynamic_axes,
                                }
    default_onnx_export_args.update(kwargs)
    torch.onnx.export(model, input_sample, onnx_path, **default_onnx_export_args)
