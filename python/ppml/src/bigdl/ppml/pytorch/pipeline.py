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
from bigdl.dllib.utils.common import JTensor, callBigDlFunc


class PytorchPipeline:
    def __init__(self, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer, algorithm,
            bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.version = 0
        self.algorithm = algorithm
        

    def server_train_step(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        y_pred = JTensor.from_ndarray(y_pred.detach().numpy())
        y_true = JTensor.from_ndarray(y_true.numpy())
        callBigDlFunc(self.bigdl_type, "pytorchTrainStep",
            y_pred, y_true, self.version, self.algorithm)

    def fit(self, x, y, epoch=2):
        for e in range(epoch):
            self.train_step(x[e], y[e])

    def train_step(self, x, y):
        self.model.train()
        pred = self.model(x)
        # In this step, loss is calculated from FLServer instead of local
        # when local loss_fn is called, return is a Pytorch Tensor
        # so get the tensor from FLServer, and transform to Pytorch Tensor
        server_loss = self.server_train_step(pred, y)
        loss = self.loss_fn(pred, y)
        loss.data = server_loss
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
