

import logging
from torch import nn
import torch
from bigdl.ppml.fl.pytorch.protobuf_utils import ndarray_map_to_tensor_map
from bigdl.ppml.fl.pytorch.utils import set_one_like_parameter
from threading import Condition

class Aggregator(object):
    def __init__(self, client_num=1) -> None:
        self.model = None
        self.client_data = {}
        self.server_data = None
        self.client_num = client_num
        self.condition = Condition()
        
    def add_server_model(self, model):
        if self.model is not None:
            raise Exception("model already exists on server")
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
                    raise Exception(f'Invalid type of tensor map key: {k}, should be input/target')
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

    

