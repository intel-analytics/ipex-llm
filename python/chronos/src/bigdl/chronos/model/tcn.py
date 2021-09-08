#
# Copyright 2018 Analytics Zoo Authors.
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
# MIT License
#
# Copyright (c) 2018 CMU Locus Lab
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

# This file is adapted from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/add_test.py
import warnings

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from zoo.orca.automl.model.base_pytorch_model import PytorchBaseModel, \
    PYTORCH_REGRESSION_LOSS_MAP


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 repo_initialization=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        if repo_initialization:
            self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,
                 past_seq_len,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 num_channels,
                 kernel_size=3,
                 dropout=0.1,
                 repo_initialization=True):
        super(TemporalConvNet, self).__init__()

        num_channels.append(output_feature_num)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_feature_num if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout, repo_initialization=repo_initialization)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(past_seq_len, future_seq_len)
        if repo_initialization:
            self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = self.linear(y)
        y = y.permute(0, 2, 1)
        return y


def model_creator(config):
    if config.get("num_channels") and (config.get("nhid") and config.get("levels")):
        warnings.warn(f"WARNING: You set both num_channels and (nhid, levels) for TCN. "
                      f"Only num_channels={config['num_channels']} will be effective.")
    if config.get("num_channels"):
        num_channels = config["num_channels"]
    else:
        n_hid = config["nhid"] if config.get("nhid") else 30
        levels = config["levels"] if config.get("levels") else 8
        num_channels = [n_hid] * (levels - 1)
    return TemporalConvNet(past_seq_len=config["past_seq_len"],
                           input_feature_num=config["input_feature_num"],
                           future_seq_len=config["future_seq_len"],
                           output_feature_num=config["output_feature_num"],
                           num_channels=num_channels.copy(),
                           kernel_size=config.get("kernel_size", 7),
                           dropout=config.get("dropout", 0.2),
                           repo_initialization=config.get("repo_initialization", True))


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 4e-3))


def loss_creator(config):
    loss_name = config.get("loss", "mse")
    if loss_name in PYTORCH_REGRESSION_LOSS_MAP:
        loss_name = PYTORCH_REGRESSION_LOSS_MAP[loss_name]
    else:
        raise RuntimeError(f"Got \"{loss_name}\" for loss name,\
                           where \"mse\", \"mae\" or \"huber_loss\" is expected")
    return getattr(torch.nn, loss_name)()


class TCNPytorch(PytorchBaseModel):
    def __init__(self, check_optional_config=False):
        super().__init__(model_creator=model_creator,
                         optimizer_creator=optimizer_creator,
                         loss_creator=loss_creator,
                         check_optional_config=check_optional_config)

    def _get_required_parameters(self):
        return {
            "past_seq_len",
            "input_feature_num",
            "future_seq_len",
            "output_feature_num"
        }

    def _get_optional_parameters(self):
        return {
            "nhid",
            "levels",
            "kernel_size",
        } | super()._get_optional_parameters()
