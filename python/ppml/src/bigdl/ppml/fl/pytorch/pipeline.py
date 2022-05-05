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

from grpc import server
import torch
from torch import nn
from bigdl.dllib.utils.common import JTensor, callBigDlFunc
from bigdl.ppml.fl.utils import init_fl_context


class PytorchPipeline:
    def __init__(self, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer, algorithm,
            bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.version = 0
        self.algorithm = algorithm
        init_fl_context()

    
    def add_server_model(model: nn.Module):
        # add model and pickle to server
        pass 

    def server_train_step(self, y_pred_local: torch.Tensor, y_true_global: torch.Tensor = None):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        y_pred_local = JTensor.from_ndarray(y_pred_local.detach().numpy())
        y_true_global = JTensor.from_ndarray(y_true_global.numpy())
        server_loss = callBigDlFunc(self.bigdl_type, "pytorchTrainStep",
            y_pred_local, y_true_global, self.version, self.algorithm)
        server_loss = server_loss.to_ndarray()
        return torch.from_numpy(server_loss)

    def fit(self, x, y, epoch=2):
        for i in range(epoch):
            self.model.train()
            pred = self.model(x)
            # In this step, loss is calculated from FLServer instead of local
            # when local loss_fn is called, return is a Pytorch Tensor
            # so get the tensor from FLServer, and transform to Pytorch Tensor
            server_loss = self.server_train_step(pred, y)
            loss = self.loss_fn(pred, pred)
            loss.data = server_loss
            print("Epoch number: {} and the loss : {}".format(i,loss.item()))

            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        
