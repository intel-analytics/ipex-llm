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
import bigdl.ppml.fl.nn.pytorch.aggregator as pt_agg
import bigdl.ppml.fl.nn.tensorflow.aggregator as tf_agg

from bigdl.ppml.fl.nn.generated.fl_base_pb2 import TensorMap
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, TrainResponse, PredictResponse, UploadMetaResponse
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import tensor_map_to_ndarray_map
import tensorflow as tf
from bigdl.dllib.utils.log4Error import invalidInputError
import pickle
import tempfile
import traceback
import os
import logging


class NNServiceImpl(NNServiceServicer):
    def __init__(self, client_num, **kargs) -> None:
        self.client_num = client_num
        self.aggregator_map = {
            'tf': tf_agg.Aggregator(client_num, **kargs),
            'pt': pt_agg.Aggregator(client_num, **kargs)}
        self.model_dir = tempfile.mkdtemp() # store tmp file dir
        self.model_path = os.path.join(self.model_dir, "vfl_server_model")

    def train(self, request: TrainRequest, context):
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        self.validate_client_id(client_id)
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        aggregator = self.aggregator_map[request.algorithm]
        try:
            aggregator.put_client_data(client_id, ndarray_map, 'train')            
            msg = f'[client {client_id} batch trained]'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return TrainResponse(response=msg, data=aggregator.server_data['train'][client_id], code=code)

    def evaluate(self, request, context):
        return super().evaluate(request, context)

    def predict(self, request, context):
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        self.validate_client_id(client_id)
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        aggregator = self.aggregator_map[request.algorithm]
        try:
            aggregator.put_client_data(client_id, ndarray_map, 'pred')            
            msg = f'[client {client_id} batch predicted]'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return PredictResponse(response=msg, data=aggregator.server_data['pred'][client_id], code=code)
        
    def upload_meta(self, request, context):
        try:
            loss_fn = pickle.loads(request.loss_fn)

            aggregator = self.aggregator_map[request.aggregator]
            if aggregator.model is not None:
                logging.warn(f'Model already exists, replacing...')
            if request.aggregator == 'pt':
                os.rename(self.model_path, f'{self.model_path}.pt')                
                aggregator.model = torch.jit.load(f'{self.model_path}.pt')
            elif request.aggregator == 'tf':
                os.rename(self.model_path, f'{self.model_path}.h5')
                aggregator.model = tf.keras.models.load_model(f'{self.model_path}.h5')
            else:
                invalidInputError(False, f"Invalid aggregator, got {request.aggregator}")

            aggregator.set_meta(loss_fn, request.optimizer)
            msg = "Upload meta success, server model is ready."
        except Exception as e:            
            msg = traceback.format_exc()
            logging.error(msg)
        return UploadMetaResponse(message=msg)

    def upload_file(self, request_iterator, context):
        try:
            with open(self.model_path, 'wb') as f:
                for byte_chunk in request_iterator:
                    f.write(byte_chunk.buffer)
            logging.info(f"Server received model file, length: {os.path.getsize(self.model_path)}")
            
            msg = "Upload model file sucess"
        except Exception as e:
            traceback.print_exc()
            msg = traceback.format_exc()
        return UploadMetaResponse(message=msg)
            
    def validate_client_id(self, client_id):
        try:
            client_id = int(client_id)
        except:
            invalidInputError(False, f"client ID must be a number, got {client_id}")

        if client_id <= 0 or client_id > self.client_num:
            invalidInputError(False, f"invalid client ID received: {client_id}, \
                must be in range of client number [1, {self.client_num}]")
