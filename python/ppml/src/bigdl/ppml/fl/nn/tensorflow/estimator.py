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
import os
import shutil
import tempfile
from numpy import ndarray
from bigdl.dllib.utils.log4Error import invalidInputError

from bigdl.ppml.fl.nn.fl_client import FLClient
from bigdl.ppml.fl.nn.nn_client import NNClient
from bigdl.ppml.fl.nn.utils import file_chunk_generate, print_file_size_in_dir, tensor_map_to_ndarray_map

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import ByteChunk

class TensorflowEstimator:
    def __init__(self, 
                 model: Model, 
                 loss_fn, 
                 optimizer_cls,
                 optimizer_args,
                 bigdl_type="float", 
                 fl_client=None,
                 server_model=None):
        self.bigdl_type = bigdl_type
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_cls(**optimizer_args)
        self.version = 0
        self.fl_client = fl_client if fl_client is not None \
            else NNClient(aggregator='tf')
        self.loss_history = []
        if server_model is not None:
            self.__add_server_model(server_model, loss_fn, optimizer_cls, optimizer_args)

    
    @staticmethod
    def load_model_as_bytes(model):
        model_path = os.path.join(tempfile.mkdtemp(), "vfl_server_model")
        model.save(model_path, save_format='h5')
        logging.info(f"Client packed model file, length: {os.path.getsize(model_path)}")
        file_chunk_generator = file_chunk_generate(model_path)
        return file_chunk_generator
        


    def __add_server_model(self, model, loss_fn=None, optimizer_cls=None, optimizer_args={}):
        # add model and pickle to server
        if loss_fn is None:
            logging.info(f'loss_fn on FLServer not specified, \
                using same as client: {self.loss_fn}')
            loss_fn = self.loss_fn
        if optimizer_cls is None:
            logging.info(f'optimizer on FLServer not specified, \
                using same as client: {self.optimizer} (with no args)')
            optimizer_cls = self.optimizer.__class__

        msg_model = self.fl_client.nn_stub.upload_file(
            TensorflowEstimator.load_model_as_bytes(model))
        logging.info(msg_model)

        msg = self.fl_client.upload_meta(loss_fn, optimizer_cls, optimizer_args).message
        logging.info(msg)

    def train_step(self, x, y):
        """
        Get the loss data from FLServer and construct the identical Pytorch Tensor
        """
        with tf.GradientTape() as tape:
            y_pred_local = self.model(x)
        y_true_global = y
        data_map = {'input': y_pred_local.numpy()}
        if y_true_global is not None:
            data_map['target'] = y_true_global.numpy()
        logging.debug(f'[{self.fl_client.client_uuid}] client sending train data to server')
        response = self.fl_client.train(data_map)
        logging.debug(f'[{self.fl_client.client_uuid}] client got response from server')
        response_map = tensor_map_to_ndarray_map(response.data.tensorMap)
        server_grad = response_map['grad']
        gradients = tape.gradient(y_pred_local, self.model.trainable_variables, tf.convert_to_tensor(server_grad))
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return response_map['loss']

    def fit(self, x, y=None, epoch=1, batch_size=4):
        for e in range(epoch):
            if isinstance(x, tf.data.Dataset):
                size = len(x)
                for batch, (X, y) in enumerate(x):
                    logging.debug(f"training batch {batch}/{size}")
                    loss = self.train_step(X, y)
                    current = batch * len(X)
                    if batch % 10 == 0:
                        logging.info(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]  \
                            epoch {e}/{epoch}")
                        self.loss_history.append(loss)
            elif isinstance(x, ndarray):
                i, size = 0, len(x)
                # just abandon last batch to reduce code
                while i < size:
                    end_index = i + batch_size if i + batch_size < size else size
                    X = tf.convert_to_tensor(x[i:end_index])
                    Y = tf.convert_to_tensor(y[i:end_index]) if y is not None else None
                    loss = self.train_step(X, Y)
                    i += batch_size
                    if i % 100 == 0:
                        logging.info(f"loss: {loss:>7f}  [{i:>5d}/{size:>5d}]  \
                            epoch {e}/{epoch}")
                        self.loss_history.append(loss)
            else:
                invalidInputError(False, f'got unsupported data input type: {type(x)}')
            

    def predict(self, x):
        y_pred_local = self.model(x)
        data_map = {'input': y_pred_local.numpy()}
        response = self.fl_client.predict(data_map)
        return response.data.tensorMap['result']
                    
            
            

    
        
