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
from torch import nn


class Pipeline:
    def __init__(self, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def get_loss_from_fl_server(self):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        pass

    def train(self, x, y, epoch=2):
        self.model.train()
        pred = self.model(x)
        # In this step, loss is calculated from FLServer instead of local
        # when local loss_fn is called, return is a Pytorch Tensor
        # so get the tensor from FLServer, and transform to Pytorch Tensor
        # TODO: get and transform
        server_loss = self.get_loss_from_fl_server()
        loss = self.loss_fn(pred, y)
        loss.data = server_loss
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()