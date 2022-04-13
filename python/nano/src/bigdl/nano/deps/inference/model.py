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
from typing import List, Tuple
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
import inspect
from torch.utils.data import DataLoader
import torch


class AcceleratedLightningModule(LightningModuleFromTorch):
    def __init__(self, model):
        super().__init__(model)
        self.on_init_end()

    def on_init_end(self, *args):
        pass

    def forward(self, *inputs):
        inputs = self.on_forward_start(inputs)
        outputs = self.forward_step(*inputs)
        return self.on_forward_end(outputs)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("This model is not trainable!")
        super().train(mode)

    def on_forward_start(self, inputs):
        return inputs

    def forward_step(self, *inputs):
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        return outputs

    def get_forward_args(self):
        return get_forward_args(self)

    @staticmethod
    def tensors_to_numpy(tensors: List[torch.Tensor] or Tuple(torch.Tensor)):
        np_data = tuple(map(lambda x: x.cpu().numpy(), tensors))
        return np_data

    @staticmethod
    def numpy_to_tensors(np_array: List or Tuple):
        tensors = tuple(map(lambda x: torch.from_numpy(x), np_array))
        if len(tensors) == 1:
            tensors = tensors[0]
        return tensors


def get_forward_args(model):
    forward_args = inspect.getfullargspec(model.forward).args[1:]
    if isinstance(model, LightningModuleFromTorch):
        # forward param list for compiled model
        forward_args = get_forward_args(model.model)
    return forward_args


def get_input_example(model: LightningModuleFromTorch, input_sample):
    if isinstance(input_sample, DataLoader):
        # TODO: This assumpe the last output is y
        input_sample = tuple(next(iter(input_sample))[:-1])
    elif input_sample is None:
        if getattr(model, "example_input_array", None) is not None:
            input_sample = model.example_input_array
        elif model.trainer is not None:
            for dataloader in [model.test_dataloader(), model.train_dataloader(),
                               model.val_dataloader()]:
                if dataloader is not None:
                    # TODO: This assumpe the last output is y
                    input_sample = tuple(next(iter(dataloader))[:-1])
                    break
            if input_sample is None and model.predict_dataloader():
                input_sample = tuple(next(iter(model.predict_dataloader())))
        else:
            raise RuntimeError("You must specify an input_sample or call `Trainer.fit` "
                               "on the model first to use `eval_openvino`")

    model.example_input_array = input_sample
    return input_sample


def export(model, input_sample=None, onnx_path="model.onnx", dynamic_axes=True):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save onnx model file.
    :param **kwargs: will be passed to torch.onnx.export function.
    '''
    input_sample = get_input_example(model, input_sample)
    assert input_sample is not None,\
        'You should set either input_sample or model.example_input_array'
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

    torch.onnx.export(model, input_sample, onnx_path, **default_onnx_export_args)
