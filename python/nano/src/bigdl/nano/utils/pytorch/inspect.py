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


import inspect
import torch
from bigdl.nano.pytorch.model import AcceleratedLightningModule


def get_forward_args(model):
    '''
    This function is to get all the arguments(excepts *args and **kwargs)
    It will return a list of arg name
    E.g.
    def forward(self, a, b=1, c: int = 3, *args, **kwargs):
        pass
    it will return ['a', 'b', 'c']
    '''
    from bigdl.nano.pytorch.lightning import LightningModule

    forward_args = inspect.getfullargspec(model.forward).args[1:]
    if isinstance(model, LightningModule):
        if not isinstance(model, AcceleratedLightningModule):
            # forward param list for compiled model
            forward_args = get_forward_args(model.model)
    return forward_args


def get_forward_defaults(model):
    '''
    This function is to get all the defaults
    It will return a list of default values
    E.g.
    def forward(self, a, b=1, c: int = 3, *args, **kwargs):
        pass
    it will return (1, 3)
    '''
    from bigdl.nano.pytorch.lightning import LightningModule

    forward_defaults = inspect.getfullargspec(model.forward).defaults
    if isinstance(model, LightningModule):
        if not isinstance(model, AcceleratedLightningModule):
            # forward param list for compiled model
            forward_defaults = get_forward_defaults(model.model)
    return forward_defaults


def get_forward_annotations(model):
    '''
    This function is to get all the annotations
    It will return a dict of {args: annotations}
    E.g.
    def forward(self, a, b=1, c: int = 3, *args, **kwargs):
        pass
    it will return {'c': <class 'int'>}
    '''
    from bigdl.nano.pytorch.lightning import LightningModule

    forward_annotations = inspect.getfullargspec(model.forward).annotations
    if isinstance(model, LightningModule):
        if not isinstance(model, AcceleratedLightningModule):
            # forward param list for compiled model
            forward_annotations = get_forward_annotations(model.model)
    return forward_annotations


def get_conditional_args(model, include=(torch.Tensor,), exclude=()):
    '''
    This function will return all the parameters that (might) in `condition`
    It will return a list or tensor args name
    E.g.
    def forward(self, a, b=1, c: int = 3, *args, **kwargs):
        pass
    it will return ['a'] if include=(torch.Tensor)

    :param include: tuple of type or "all".
    :param exclude: tuple of type or "all".

    Note: "all" means all the types are allowed or disallowed, except those
          stated in the opposite parameter.
    Note: exclude has higher priority if conflict instruction is provided
    '''
    include_all = True if include == "all" else False
    exclude_all = True if exclude == "all" else False

    forward_args = get_forward_args(model)
    forward_defaults = get_forward_defaults(model)
    forward_annotations = get_forward_annotations(model)
    fitted_args = []
    if forward_defaults is None:
        defaults_length = 0
    else:
        defaults_length = len(forward_defaults)
    args_length = len(forward_args)
    for i, arg in enumerate(forward_args):
        # check if type hint is provided
        flag = False
        if arg in forward_annotations:
            if include_all or forward_annotations[arg] in include:
                flag = True
            if exclude_all or forward_annotations[arg] in exclude:
                flag = False
            if flag:
                fitted_args.append(arg)
            continue
        # check if defaults is meets requirement
        default_args_start_from = args_length - defaults_length
        if i >= default_args_start_from:
            flag = False
            if include_all or type(forward_defaults[i - default_args_start_from]) in include:
                flag = True
            if exclude_all or type(forward_defaults[i - default_args_start_from]) in exclude:
                flag = False
            if flag:
                fitted_args.append(arg)
            continue
        # nothing is provided, we assume it meets requirement
        fitted_args.append(arg)
    return fitted_args
