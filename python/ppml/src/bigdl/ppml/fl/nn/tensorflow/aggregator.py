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
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
from threading import Condition

import numpy as np
import tensorflow as tf

# TODO: tf and pytorch aggregator could be integrated to one using inherit
class Aggregator(object):
    def __init__(self,
                 client_num=1) -> None:
        self.model = None
        self.client_data = {}
        self.server_data = {}
        self.client_num = client_num
        self.condition = Condition()
        self._lock = threading.Lock()
        logging.info(f"Initialized Tensorflow aggregator [client_num: {client_num}]")


    def set_meta(self, loss_fn, optimizer):
        with self._lock:            
            self.set_loss_fn(loss_fn)
            optimizer_cls = pickle.loads(optimizer.cls)
            optimizer_args = pickle.loads(optimizer.args)
            self.set_optimizer(optimizer_cls, optimizer_args)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer_cls, optimizer_args):
        if len(list(self.model.trainable_variables)) == 0:
            self.optimizer = None
            return
        self.optimizer = optimizer_cls(**optimizer_args)


    def put_client_data(self, client_id, data):
        self.condition.acquire()
        self.client_data[client_id] = data
        logging.debug(f'server receive data [{client_id}], \
got {len(self.client_data)}/{self.client_num}')
        
        if len(self.client_data) == self.client_num:            
            logging.debug('server received all client data, start aggregate')
            self.aggregate()
            logging.debug('clearing client data')
            self.client_data = {}
            self.condition.notify_all()            
        else:
            logging.debug(f'[{client_id}] waiting')
            self.condition.wait()
        self.condition.release()


    def aggregate(self):
        input, target = [], None        
        for cid, ndarray_map in self.client_data.items():
            for k, v in ndarray_map.items():
                if k == 'input':
                    input.append((cid, tf.convert_to_tensor(v)))
                elif k == 'target':
                    target = tf.convert_to_tensor(v)
                else:
                    invalidInputError(False, f'Invalid type of tensor map key: {k}, should be input/target')
        # TODO: to be consistent with Pytorch, custom API
        
        def sort_by_key(kv_tuple):
            return kv_tuple[0]
        
        input.sort(key=sort_by_key)
        tensor_list = []
        for cid, input_tensor in input:
            input_tensor.requires_grad = True
            tensor_list.append(input_tensor)
        
        with tf.GradientTape(persistent=True) as tape:
            for tensor in tensor_list:
                tape.watch(tensor)
            pred = self.model(tensor_list)
            loss = self.loss_fn(target, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.optimizer is not None:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        for cid, input_tensor in input:
            x_grad = tape.gradient(loss, input_tensor)
            grad_map = {'grad': x_grad.numpy(), 'loss': np.array(loss.numpy())}
            self.server_data[cid] = ndarray_map_to_tensor_map(grad_map)
        
        del tape # manually delete the persistent GradientTape
    

