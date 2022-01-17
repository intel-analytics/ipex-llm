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
import onnxruntime as ort
from functools import partial, wraps
import warnings
import torch
import math
import numpy as np
import inspect


ONNXRT_BINDED_COMPONENTS = ['_ortsess_up_to_date',
                            '_ortsess',
                            '_build_ortsess',
                            'update_ortsess',
                            '_forward_onnx']


# internal function to build an ortsess
def _build_ortsess(self,
                   input_sample=None,
                   file_path="model.onnx",
                   sess_options=None,
                   **kwargs):
    '''
    Internal function to build a ortsess and bind to the lightningmodule.

    :param input_sample: torch.Tensor or a list for the model tracing.
    :param file_path: The path to save onnx model file.
    :param sess_options: ortsess options in ort.SessionOptions type
    :param **kwargs: will be passed to torch.onnx.export function.
    '''

    if input_sample is None and self.example_input_array is not None:
        input_sample = self.example_input_array  # use internal example_input_array
    else:
        self.example_input_array = input_sample  # set example_input_array for future usage

    assert input_sample is not None,\
        'You should set either input_sample or self.example_input_array'

    dynamic_axes = {}
    for forward_arg in self._forward_args:
        dynamic_axes[forward_arg] = {0: 'batch_size'}  # set all dim0 to be dynamic
    dynamic_axes['output'] = {0: 'batch_size'}

    default_onnx_export_args = {'export_params': True,
                                'opset_version': 10,  # version = 10 by default
                                'do_constant_folding': True,
                                'input_names': self._forward_args,
                                'output_names': ['output'],  # TODO: only support single output
                                'dynamic_axes': dynamic_axes}
    default_onnx_export_args.update(kwargs)

    torch.onnx.export(self,
                      input_sample,
                      file_path,
                      **default_onnx_export_args)

    self._ortsess = ort.InferenceSession(file_path, sess_options=sess_options)
    self._ortsess_up_to_date = True


# external method to update(& rebuild) ortsess
def update_ortsess(self,
                   input_sample=None,
                   file_path="model.onnx",
                   sess_options=None,
                   **kwargs):
    '''
    Update the onnxruntime session options and rebuild the session.
    Users may also want to call this method before `inference(..., onnx=True`)`
    to avoid implicit building.

    :param input_sample: torch.Tensor for the model tracing.
    :param file_path: The path to save onnx model file.
    :param sess_options: ortsess options in ort.SessionOptions type.
    :param **kwargs: will be passed to torch.onnx.export function.
    '''
    self._build_ortsess(input_sample=input_sample,
                        file_path=file_path,
                        sess_options=sess_options,
                        **kwargs)


# on_fit_start (LightningModule method overwrite)
def _onnx_on_fit_start(self):
    self._ortsess_up_to_date = False
    self._ortsess = None
    self.exit_onnx()


def _onnx_on_train(self, mode=True):
    self.exit_onnx()
    self._ortsess_up_to_date = False


def _forward_onnx(self, *args):
    ort_inputs = {}
    for i, ort_input_item in enumerate(args):
        if isinstance(ort_input_item, torch.Tensor):
            ort_input_item = ort_input_item.numpy()
        ort_inputs[self._forward_args[i]] = ort_input_item
    ort_outs = self._ortsess.run(None, ort_inputs)
    return torch.from_numpy(ort_outs[0])


def eval_onnx(self, input_sample=None, file_path="model.onnx", sess_options=None, **kwargs):
    '''
    This method change the `forward` method to an onnxruntime backed forwarding.

    >>> model.eval_onnx()
    >>> pred = model(x)  # onnxruntime forwarding
    >>> model.exit_onnx()

    :param input_sample: (optional) a torch dataloader, torch.Tensor or a
           list of them for the model tracing.
    :param file_path: (optional) The path to save onnx model file.
    :param sess_options: (optional) ortsess options in ort.SessionOptions type.
    :param **kwargs: (optional) will be passed to torch.onnx.export function.
    '''
    # change to eval mode
    self.eval()

    # get input_sample
    if isinstance(input_sample, DataLoader):
        input_sample = tuple(next(iter(input_sample))[:-1])
    if input_sample is None and self.example_input_array:
        input_sample = self.example_input_array
    if input_sample is None and self.trainer is None:
        raise RuntimeError("You must specify an input_sample or call `Trainer.fit` "
                           "on the model first to use `eval_onnx`")
    if input_sample is None and self.trainer.train_dataloader:
        input_sample = tuple(next(iter(self.trainer.train_dataloader))[:-1])
    if input_sample is None and self.trainer.datamodule:
        input_sample = tuple(next(iter(self.trainer.datamodule.train_dataloader()))[:-1])
    assert input_sample is not None,\
        "You must state an input_sample or fit on the model to use `eval_onnx`."

    # build ortsess
    self._build_ortsess(input_sample=input_sample,
                        file_path=file_path,
                        sess_options=sess_options,
                        **kwargs)

    self.forward = self._forward_onnx


def exit_onnx(self):
    self.forward = self._torch_forward


def bind_onnxrt_methods(pl_model: LightningModule):
    # class type check
    assert isinstance(pl_model, LightningModule),\
        f"onnxruntime support is only valid for a LightningModule, but found a {type(pl_model)}."

    # check conflicts
    for component in ONNXRT_BINDED_COMPONENTS:
        if component in dir(pl_model):
            warnings.warn(f"{component} method/property will be replaced. You may rename your"
                          " customized attributes or methods and call `Trainer.compile again `"
                          "to avoid being overwrite.")

    # additional attributes
    pl_model._ortsess_up_to_date = False  # indicate if we need to build ortsess again
    pl_model._ortsess = None  # ortsess instance
    if isinstance(pl_model, LightningModuleFromTorch):  # forward param list for compiled model
        pl_model._forward_args = inspect.getfullargspec(pl_model.model.forward).args[1:]
    else:  # forward param list
        pl_model._forward_args = inspect.getfullargspec(pl_model.forward).args[1:]

    # additional methods
    pl_model._build_ortsess = partial(_build_ortsess, pl_model)
    pl_model.update_ortsess = partial(update_ortsess, pl_model)
    pl_model._onnx_on_fit_start = partial(_onnx_on_fit_start, pl_model)
    pl_model.eval_onnx = partial(eval_onnx, pl_model)
    pl_model._forward_onnx = partial(_forward_onnx, pl_model)
    pl_model.exit_onnx = partial(exit_onnx, pl_model)
    pl_model._onnx_on_train = partial(_onnx_on_train, pl_model)

    return pl_model
