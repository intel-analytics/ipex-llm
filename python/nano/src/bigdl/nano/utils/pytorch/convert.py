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


import torch
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_11
from bigdl.nano.utils.pytorch import get_forward_args, get_conditional_args,\
    get_input_example, complement_input_sample


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
    tensor_args = get_conditional_args(model)
    input_sample = get_input_example(model, input_sample, forward_args)
    input_sample = complement_input_sample(model, input_sample)

    invalidInputError(input_sample is not None,
                      'You should implement at least one of model.test_dataloader, '
                      'model.train_dataloader, model.val_dataloader and '
                      'model.predict_dataloader, '
                      'or set one of input_sample and model.example_input_array')
    if isinstance(dynamic_axes, dict):
        pass
    elif dynamic_axes is True:
        dynamic_axes = {}
        for arg in tensor_args:
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
                                'input_names': tensor_args,
                                'dynamic_axes': dynamic_axes,
                                }
    default_onnx_export_args.update(kwargs)
    torch.onnx.export(model, input_sample, onnx_path, **default_onnx_export_args)
