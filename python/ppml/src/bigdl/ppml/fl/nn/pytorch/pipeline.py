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

import logging
from numpy import ndarray
import torch
from torch import nn
import torch
from bigdl.ppml.fl.nn.fl_client import FLClient
from torch.utils.data import DataLoader
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.ppml.fl.nn.utils import tensor_map_to_ndarray_map

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
        self.loss_history = []

    
    
    def add_server_model(self, model: nn.Module, loss_fn=None, optimizer_cls=None, optimizer_args={}):
        # add model and pickle to server
        if loss_fn is None:
            logging.info(f'loss_fn on FLServer not specified, \
                using same as client: {self.loss_fn}')
            loss_fn = self.loss_fn
        if optimizer_cls is None:
            logging.info(f'optimizer on FLServer not specified, \
                using same as client: {self.optimizer} (with no args)')
            optimizer_cls = self.optimizer.__class__

        msg = self.fl_client.upload_model(model, loss_fn, optimizer_cls, optimizer_args).message
        logging.info(msg)

    def train_step(self, x, y):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        y_pred_local = self.model(x)
        y_true_global = y
        data_map = {'input': y_pred_local.detach().numpy()}
        if y_true_global is not None:
            data_map['target'] = y_true_global.detach().numpy()
        logging.debug(f'[{self.fl_client.client_uuid}] client sending train data to server')
        response = self.fl_client.train(data_map)
        logging.debug(f'[{self.fl_client.client_uuid}] client got response from server')
        response_map = tensor_map_to_ndarray_map(response.data.tensorMap)
        grad = response_map['grad']
        self.optimizer.zero_grad()
        y_pred_local.backward(gradient=torch.tensor(grad))
        self.optimizer.step()
        return response_map['loss']

    def fit(self, x, y=None, epoch=1, batch_size=4):
        for e in range(epoch):
            self.model.train()
            if isinstance(x, DataLoader):
                size = len(x.dataset)
                for batch, (X, y) in enumerate(x):
                    loss = self.train_step(X, y)
                    current = batch * len(X)
                    if batch % 100 == 0:
                        logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  \
                            epoch {e}/{epoch}")
                        self.loss_history.append(loss)
            elif isinstance(x, ndarray):
                i, size = 0, len(x)
                # just abandon last batch to reduce code
                while i < size:
                    end_index = i + batch_size if i + batch_size < size else size
                    X = torch.from_numpy(x[i:end_index])
                    Y = torch.from_numpy(y[i:end_index]) if y is not None else None
                    loss = self.train_step(X, Y)
                    i += batch_size
                    if i % 100 == 0:
                        logging.info(f"loss: {loss:>7f}  [{i:>5d}/{size:>5d}]  \
                            epoch {e}/{epoch}")
                        self.loss_history.append(loss)
            else:
                invalidInputError(False,
                                  f'got unsupported data input type: {type(x)}')
            

    def predict(self, x):
        y_pred_local = self.model(x)
        data_map = {'input': y_pred_local.detach().numpy()}
        response = self.fl_client.predict(data_map)
        return response.data.tensorMap['result']
                    
            
            

    
        
