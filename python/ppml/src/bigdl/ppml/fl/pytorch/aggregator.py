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
from torch import nn
import torch
from bigdl.ppml.fl.pytorch.protobuf_utils import ndarray_map_to_tensor_map
from bigdl.ppml.fl.pytorch.utils import set_one_like_parameter
from threading import Condition
from bigdl.dllib.utils.log4Error import invalidInputError

class Aggregator(object):
    def __init__(self, client_num=1) -> None:
        self.model = None
        self.client_data = {}
        self.server_data = None
        self.client_num = client_num
        self.condition = Condition()
        
    def add_server_model(self, model):
        if self.model is not None:
            invalidInputError(False,
                              "model already exists on server")
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def put_client_data(self, client_id, data):
        self.client_data[client_id] = data
        logging.debug(f'server receive data {len(self.client_data)}/{self.client_num}')
        if len(self.client_data) == self.client_num:
            self.condition.acquire()
            logging.debug('server received all client data, start aggregate')
            self.aggregate()
            self.condition.notify_all()
            self.condition.release()
        else:
            logging.debug('waiting')
            self.condition.wait()


    def aggregate(self):
        input, target = [], None        
        for ndarray_map in self.client_data.values():
            for k, v in ndarray_map.items():
                if k == 'input':
                    input.append(torch.from_numpy(v))
                elif k == 'target':
                    target = torch.from_numpy(v)
                else:
                    invalidInputError(False,
                                      f'Invalid type of tensor map key: {k},'
                                      f' should be input/target')
        x = torch.stack(input)
        x = torch.sum(x, dim=0)
        x.requires_grad = True
        pred = self.model(x)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        grad_map = {'grad': x.grad.numpy(), 'loss': loss.detach().numpy()}
        self.server_data = ndarray_map_to_tensor_map(grad_map)

    

