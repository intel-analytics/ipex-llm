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

PYTORCH_LOSS_NAMES = {s for s in dir(torch.nn.modules) if s.endswith("Loss")}
PYTORCH_OPTIM_NAMES = {s for s in dir(torch.optim) if any(c.isupper() for c in s)} - {'Optimizer'}

LR_NAME = "lr"
DEFAULT_LR = 1e-3
BATCH_SIZE_NAME = "batch_size"
DEFAULT_BATCH_SIZE = 32


def validate_pytorch_loss(loss):
    import types
    if isinstance(loss, str):
        if loss in PYTORCH_LOSS_NAMES:
            return getattr(torch.nn.modules, loss)()
        raise ValueError(f'Must provide a valid torch loss name among {PYTORCH_LOSS_NAMES}')

    if isinstance(loss, torch.nn.modules.loss._Loss) or \
            isinstance(loss, types.FunctionType):
        return loss

    raise ValueError("Must provide a valid pytorch loss name or a pytorch loss instance"
                     "or a pytorch loss creator function.")


def validate_pytorch_optim(optim):
    import types
    if isinstance(optim, str):
        if optim in PYTORCH_OPTIM_NAMES:
            return getattr(torch.optim, optim)
        raise ValueError(f'Must provide a valid torch optimizer name among {PYTORCH_OPTIM_NAMES}')

    if isinstance(optim, types.FunctionType):
        return optim

    raise ValueError("Must provide a valid pytorch optimizer name "
                     "or a pytorch optimizer creator function.")
