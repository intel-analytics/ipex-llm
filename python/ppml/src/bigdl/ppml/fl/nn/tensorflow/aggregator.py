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
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
from threading import Condition

import tensorflow as tf

# TODO: tf and pytorch aggregator could be integrated to one using inherit
class Aggregator(object):
    def __init__(self,
                 client_num=1) -> None:
        self.model = None
        self.client_data = {}
        self.server_data = None
        self.client_num = client_num
        self.condition = Condition()
        self._lock = threading.Lock()
        logging.info(f"Initialized Tensorflow aggregator [client_num: {client_num}]")

    # deprecated, use set_server_model for fully customized NN Model
    def add_server_model(self, model):
        with self._lock:
            if self.model is not None:
                logging.warn("model exists on server, the add model operation is skipped")
            else:
                self.model = model
                self.init_loss_fn()
                self.init_optimizer()

    def set_server_model(self, model, loss_fn, optimizer):
        with self._lock:
            if self.model is not None:
                logging.warn("model exists on server, the add model operation is skipped")
            else:
                self.model = model
                self.set_loss_fn(loss_fn)
                optimizer_cls = pickle.loads(optimizer.cls)
                optimizer_args = pickle.loads(optimizer.args)
                self.set_optimizer(optimizer_cls, optimizer_args)   

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer_cls, optimizer_args):
        if len(list(self.model.parameters())) == 0:
            self.optimizer = None
            return
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

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
        for ndarray_map in self.client_data.values():
            for k, v in ndarray_map.items():
                if k == 'input':
                    input.append(tf.convert_to_tensor(v))
                elif k == 'target':
                    target = tf.convert_to_tensor(v)
                else:
                    raise Exception(f'Invalid type of tensor map key: {k}, should be input/target')
        x = tf.stack(input)
        x = tf.add(x, dim=0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = self.model(x)
            loss = self.loss_fn(pred, target)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.optimizer is not None:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        x_grad = tape.gradient(loss, x)
        grad_map = {'grad': x_grad.numpy(), 'loss': loss.numpy()}
        self.server_data = ndarray_map_to_tensor_map(grad_map)

    

