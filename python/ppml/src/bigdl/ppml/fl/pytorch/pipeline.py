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

from copyreg import pickle
import logging
from grpc import server
from numpy import gradient
import torch
from torch import nn
from bigdl.dllib.utils.common import JTensor, callBigDlFunc
from bigdl.ppml.fl.pytorch.fl_client import FLClient
from bigdl.ppml.fl.utils import init_fl_context
from torch.utils.data import DataLoader
from bigdl.ppml.fl.pytorch.protobuf_utils import tensor_map_to_ndarray_map

class PytorchPipeline:
    def __init__(self, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer, algorithm=None,
            bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.version = 0
        self.algorithm = algorithm
        self.fl_client = FLClient()
        # init_fl_context()

    
    def add_server_model(self, model: nn.Module):
        # add model and pickle to server
        self.fl_client.upload_model(model)

    def train_step(self, x, y):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        y_pred_local = self.model(x)
        y_true_global = y
        data_map = {'input': y_pred_local.detach().numpy(), 'target': y_true_global.detach().numpy()}
        response = self.fl_client.train(data_map)
        response_map = tensor_map_to_ndarray_map(response.data.tensorMap)
        grad = response_map['grad']
        self.optimizer.zero_grad()
        y_pred_local.backward(gradient=torch.tensor(grad))
        self.optimizer.step()
        return response_map['loss']

    def fit(self, x, y=None, epoch=5):
        for i in range(epoch):
            self.model.train()
            if isinstance(x, DataLoader):
                size = len(x.dataset)
                for batch, (X, y) in enumerate(x):
                    loss = self.train_step(X, y)
                    current = batch * len(X)
                    if batch % 100 == 0:
                        logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    
            
            

    
        
