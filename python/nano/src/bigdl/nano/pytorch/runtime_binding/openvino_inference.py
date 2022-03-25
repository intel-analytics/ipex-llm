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

from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from functools import partial, wraps
import warnings
import torch
import numpy as np
import inspect
from openvino.inference_engine import IECore


OPENVINO_BINDED_COMPONENTS = ['ir_up_to_date', 'ir_model']


# internal function to build an ortsess
def _build_ir_model(self, input_sample=None, file_path="model.xml", **kwargs):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save openvino model file.
    :param **kwargs: will be passed to torch.onnx.export function.
    '''

    # quantized model will not be supported
    if "_quantized_model" in dir(self):
        self._quantized_model = None
        self._quantized_model_up_to_date = False

    # get input_sample
    if isinstance(input_sample, DataLoader):
        input_sample = tuple(next(iter(input_sample))[:-1])
    if input_sample is None and self.example_input_array:
        input_sample = self.example_input_array
    if input_sample is None and self.example_input_array is not None:
        input_sample = self.example_input_array  # use internal example_input_array

    assert input_sample is not None,\
        'You should set either input_sample or self.example_input_array'

    dynamic_axes = {}
    for forward_arg in self._forward_args:
        dynamic_axes[forward_arg] = {0: 'batch_size'}  # set all dim0 to be dynamic
    dynamic_axes['output'] = {0: 'batch_size'}

    default_onnx_export_args = {'export_params': True,
                                'opset_version': 11,  # version = 11 by default
                                'do_constant_folding': True,
                                'input_names': self._forward_args,
                                'output_names': ['output'],  # TODO: only support single output
                                # 'dynamic_axes': dynamic_axes
                                }
    default_onnx_export_args.update(kwargs)

    torch.onnx.export(self,
                      input_sample,
                      'model.onnx',
                      **default_onnx_export_args)
    infer_engine = IECore()
    net_onnx = infer_engine.read_network(model='model.onnx')
    net_onnx.serialize(file_path)
    self.ir_model = infer_engine.load_network(network=file_path, device_name='CPU')
    self.ir_up_to_date = True


def train(self, mode=True):
    self.exit_openvino()
    if mode:
        self.ir_up_to_date = False
        self.ir_model = None
    assert self.train == self._torch_train
    self.train(mode)


def forward_batch_start(*args):
    ort_inputs = []
    for ort_input_item in args:
        if isinstance(ort_input_item, torch.Tensor):
            ort_input_item = ort_input_item.numpy()
        ort_inputs.append(ort_input_item)
    return ort_inputs


def forward_batch_end(*outputs):
    return torch.from_numpy(outputs[0])


def forward(self, *args):
    args = forward_batch_start(*args)
    outputs = _forward_openvino(self, *args)
    return forward_batch_end(*outputs.values())


def _forward_openvino(self, *args):
    inputs = dict(zip(self.ir_model.input_info, args))
    return self.ir_model.infer(inputs)


def eval_openvino(self, input_sample=None, file_path="model.xml", **kwargs):
    '''
    This method change the `forward` method to an openvino backed forwarding.

    >>> model.eval_openvino(quantize=True/False)
    >>> pred = model(x)  # openvino forwarding
    >>> model.exit_openvino()

    :param input_sample: (optional) a torch dataloader, torch.Tensor or a
           list of them for the model tracing.
    :param file_path: (optional) The path to save openvino model file.
    :param **kwargs: (optional) will be passed to torch.onnx.export function.
    '''
    # change to eval mode
    self.eval()
    if not self.ir_up_to_date:
        # get input_sample
        if isinstance(input_sample, DataLoader):
            input_sample = tuple(next(iter(input_sample))[:-1])
        elif input_sample is None:
            if self.example_input_array:
                input_sample = self.example_input_array
            if self.trainer is None:
                raise RuntimeError("You must specify an input_sample or call `Trainer.fit` "
                                   "on the model first to use `eval_openvino`")
            if self.train_dataloader:
                input_sample = tuple(next(iter(self.train_dataloader()))[:-1])
        assert input_sample is not None,\
            "You must state an input_sample or fit on the model to use `eval_onnx`."
        _build_ir_model(self, input_sample=input_sample, file_path=file_path, **kwargs)
    self._torch_train = self.train
    self._torch_eval = self.eval
    self.train = partial(train, self)
    self.forward = partial(forward, self)


def exit_openvino(self):
    self.train = self._torch_train
    self.forward = self._torch_forward


def bind_openvino_methods(pl_model: LightningModule):

    # additional attributes
    pl_model.ir_up_to_date = False  # indicate if we need to build ortsess again
    pl_model.ir_model = None  # onnx graph for quantization
    if isinstance(pl_model, LightningModuleFromTorch):  # forward param list for compiled model
        pl_model._forward_args = inspect.getfullargspec(pl_model.model.forward).args[1:]
    else:  # forward param list
        pl_model._forward_args = inspect.getfullargspec(pl_model.forward).args[1:]

    # additional methods
    pl_model._torch_forward = pl_model.forward
    pl_model.eval_openvino = partial(eval_openvino, pl_model)
    pl_model.exit_openvino = partial(exit_openvino, pl_model)

    return pl_model
