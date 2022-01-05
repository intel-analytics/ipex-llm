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
# the following code is adapted from https://github.com/philipperemy/n-beats/
#
# MIT License
#
# Copyright (c) 2019 Philippe Rémy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer

from .utils import PYTORCH_REGRESSION_LOSS_MAP


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 stack_types=(GENERIC_BLOCK, GENERIC_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None):
        super(NBeatsNet, self).__init__()
        self.future_seq_len = future_seq_len
        self.past_seq_len = past_seq_len
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = torch.nn.ModuleList()
        self.thetas_dim = thetas_dim
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = torch.nn.ModuleList()
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet._select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.past_seq_len, self.future_seq_len, self.nb_harmonics)
            blocks.append(block)
        return blocks

    @staticmethod
    def _select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        backcast = backcast[..., 0]  # can only use the target
        forecast = 0
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return torch.unsqueeze(forecast, 2)  # return 3-dim output


def seasonality_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S)


def trend_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T)


def linear_space(past_seq_len, future_seq_len):
    ls = np.arange(-past_seq_len, future_seq_len, 1) / future_seq_len
    b_ls = np.abs(np.flip(ls[:past_seq_len]))
    f_ls = ls[past_seq_len:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.past_seq_len = past_seq_len
        self.future_seq_len = future_seq_len
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(past_seq_len, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_linspace, self.forecast_linspace = linear_space(past_seq_len, future_seq_len)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'past_seq_len={self.past_seq_len}, future_seq_len={self.future_seq_len}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, past_seq_len,
                                                   future_seq_len, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, future_seq_len, past_seq_len,
                                                   future_seq_len, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, past_seq_len,
                                         future_seq_len, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, past_seq_len=10, future_seq_len=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, past_seq_len, future_seq_len)

        self.backcast_fc = nn.Linear(thetas_dim, past_seq_len)
        self.forecast_fc = nn.Linear(thetas_dim, future_seq_len)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


def model_creator(config):
    return NBeatsNet(past_seq_len=config["past_seq_len"],
                     future_seq_len=config["future_seq_len"],
                     stack_types=config.get("stack_types", ("generic", "generic")),
                     nb_blocks_per_stack=config.get("nb_blocks_per_stack", 3),
                     thetas_dim=config.get("thetas_dim", (4, 8)),
                     share_weights_in_stack=config.get("share_weights_in_stack", False),
                     hidden_layer_units=config.get("hidden_layer_units", 256),
                     nb_harmonics=config.get("nb_harmonics", None))


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 0.001))


def loss_creator(config):
    loss_name = config.get("loss", "mse")
    if loss_name in PYTORCH_REGRESSION_LOSS_MAP:
        loss_name = PYTORCH_REGRESSION_LOSS_MAP[loss_name]
    else:
        raise RuntimeError(f"Got '{loss_name}' for loss name, "
                           "where 'mse', 'mae' or 'huber_loss' is expected")
    return getattr(torch.nn, loss_name)()


try:
    from bigdl.orca.automl.model.base_pytorch_model import PytorchBaseModel

    class NBeatsPytorch(PytorchBaseModel):
        def __init__(self, check_optional_config=False):
            super().__init__(model_creator=model_creator,
                             optimizer_creator=optimizer_creator,
                             loss_creator=loss_creator,
                             check_optional_config=check_optional_config)

        def _get_required_parameters(self):
            return {
                "past_seq_len",
                "future_seq_len"
            }

        def _get_optional_parameters(self):
            return {
                "stack_types",
                "nb_blocks_per_stack",
                "thetas_dim",
                "share_weights_in_stack",
                "hidden_layer_units",
                "nb_harmonics"
            } | super()._get_optional_parameters()
except ImportError:
    pass
