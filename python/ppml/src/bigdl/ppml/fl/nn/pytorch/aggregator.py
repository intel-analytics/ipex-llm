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

import pickle
import logging
import threading
from torch import nn
import torch
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
from bigdl.dllib.utils.log4Error import invalidInputError, invalidOperationError
from threading import Condition
import os


class Aggregator(object):
    def __init__(self, conf) -> None:
        self.model = None
        self.client_data = {'train':{}, 'eval':{}, 'pred':{}}
        self.server_data = {'train':{}, 'eval':{}, 'pred':{}}
        self.client_num = conf['clientNum']
        self.client_num = int(self.client_num)
        self.condition = Condition()
        self._lock = threading.Lock()
        self.optimizer_cls = None
        self.optimizer_args = None
        logging.info(f"Initialized Pytorch aggregator [client_num: {self.client_num}]")

    def set_meta(self, loss_fn, optimizer):
        with self._lock:
            self.set_loss_fn(loss_fn)
            optimizer_cls = pickle.loads(optimizer.cls)
            optimizer_args = pickle.loads(optimizer.args)
            self.optimizer_cls, self.optimizer_args = optimizer_cls, optimizer_args
            self.set_optimizer(optimizer_cls, optimizer_args)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer_cls, optimizer_args):
        if len(list(self.model.parameters())) == 0:
            self.optimizer = None
            return
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def put_client_data(self, client_id, data, phase):
        self.condition.acquire()
        self.client_data[phase][client_id] = data
        logging.debug(f'server receive data [{client_id}], \
got {len(self.client_data[phase])}/{self.client_num}')
        
        if len(self.client_data[phase]) == self.client_num:            
            logging.debug('server received all client data, start aggregate')
            self.aggregate(phase)
            logging.debug('clearing client data')
            self.client_data[phase] = {}
            self.condition.notify_all()            
        else:
            logging.debug(f'[{client_id}] waiting')
            self.condition.wait()
        self.condition.release()


    def aggregate(self, phase):
    
        input, target = [], None
        # to record the order of tensors with client ID
        for cid, ndarray_map in self.client_data[phase].items():
            for k, v in ndarray_map.items():
                if k == 'input':
                    input.append((cid, torch.from_numpy(v)))
                elif k == 'target':
                    target = torch.from_numpy(v)
                else:
                    invalidInputError(False,
                                      f'Invalid type of tensor map key: {k},'
                                      f' should be input/target')
        # input is a list of tensors

        # x = torch.stack(input)
        # x = torch.sum(x, dim=0)
        # x.requires_grad = True
        # pred = self.model(x)

        # sort the input tensor list in order to keep the order info of client ID
        def sort_by_key(kv_tuple):
            return kv_tuple[0]
        
        input.sort(key=sort_by_key)
        tensor_list = []
        for cid, input_tensor in input:
            input_tensor.requires_grad = True
            tensor_list.append(input_tensor)

        if phase == 'train':
            pred = self.model(tensor_list)
            loss = self.loss_fn(pred, target)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()

            for cid, input_tensor in input:
                grad_map = {'grad': input_tensor.grad.numpy(), 'loss': loss.detach().numpy()}
                self.server_data['train'][cid] = ndarray_map_to_tensor_map(grad_map)            

        elif phase == 'eval':
            pass
        elif phase == 'pred':
            pred = self.model(tensor_list)
            for cid, input_tensor in input:
                pred_map = {'pred': pred.detach().numpy()}
                self.server_data['pred'][cid] = ndarray_map_to_tensor_map(pred_map)
        else:
            invalidInputError(False,
                              f'Invalid phase: {phase}, should be train/eval/pred')

    def load_uploaded_model(self, client_id, model_path):
        if self.model is not None:
            invalidOperationError(False,
                f"Model exists, model uploading from {client_id} ignored.")
        else:
            os.rename(model_path, f'{model_path}.pt')                
            self.model = torch.jit.load(f'{model_path}.pt')

    def save_server_model(self, model_path):
        if not os.path.exists(f"{model_path}/model.meta"):
            os.makedirs(f"{model_path}", exist_ok=True)
            with open(f"{model_path}/model.meta", 'wb') as meta_file:
                pickle.dump({'loss': self.loss_fn,
                             'optimizer': (self.optimizer_cls, self.optimizer_args)},
                            meta_file)
        m = torch.jit.script(self.model)
        torch.jit.save(m, f"{model_path}/model.pt")
        # save meta to file if not saved yet
        

    def load_server_model(self, client_id, model_path):
        if self.model is not None:
            invalidOperationError(False,
                f"Model exists, model uploading from {client_id} ignored.")
        else:
            logging.info(f"Trying to load model from {model_path}")
            self.model = torch.jit.load(f"{model_path}/model.pt")
            # if loaded, set meta here to make the optimizer bind the model
            with open(f"{model_path}/model.meta", "rb") as meta_file:
                meta = pickle.load(meta_file)
                self.loss_fn = meta['loss']
                self.set_optimizer(meta['optimizer'][0], meta['optimizer'][1])
