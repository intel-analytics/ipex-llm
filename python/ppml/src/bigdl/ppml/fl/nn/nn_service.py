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


import bigdl.ppml.fl.nn.pytorch.aggregator as pt_agg
import bigdl.ppml.fl.nn.tensorflow.aggregator as tf_agg

from bigdl.ppml.fl.nn.generated.fl_base_pb2 import TensorMap
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, TrainResponse, UploadModelResponse
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import tensor_map_to_ndarray_map
import tensorflow as tf
import pickle
import tempfile
import traceback
import os
import logging


class NNServiceImpl(NNServiceServicer):
    def __init__(self, **kargs) -> None:
        self.aggregator_map = {
            'tf': tf_agg.Aggregator(**kargs),
            'pt': pt_agg.Aggregator(**kargs)}

    def train(self, request: TrainRequest, context):
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        aggregator = self.aggregator_map[request.algorithm]
        try:
            aggregator.put_client_data(client_id, ndarray_map)            
            msg = f'[client {client_id} batch trained]'
            code = 0
        except Exception as e:            
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return TrainResponse(response=msg, data=self.aggregator.server_data[client_id], code=code)

    def evaluate(self, request, context):
        return super().evaluate(request, context)

    def predict(self, request, context):
        return super().predict(request, context)
        
    def upload_model(self, request, context):
        try:            
            model = pickle.loads(request.model_bytes) if request.aggregator == 'pt' else None
            loss_fn = pickle.loads(request.loss_fn)
            aggregator = self.aggregator_map[request.aggregator]
            aggregator.set_server(model, loss_fn, request.optimizer)
            msg = "Upload sucess"
        except Exception as e:
            msg = traceback.format_exc()
        return UploadModelResponse(message=msg)

    def upload_file(self, request_iterator, context):
        try:
            tmpdir = tempfile.mkdtemp()
            model_path = os.path.join(tmpdir, "tf_vfl_server_model.h5")
            with open(model_path, 'wb') as f:
                for byte_chunk in request_iterator:
                    f.write(byte_chunk.buffer)
            logging.info(f"Server received model file, length: {os.path.getsize(model_path)}")
            # shutil.unpack_archive(zip_path, tmpdir)
            # loaded = tf.saved_model.load(tmpdir)
            model = tf.keras.models.load_model(model_path)
            # hard code this func to use tf for now
            aggregator = self.aggregator_map['tf']
            aggregator.set_server_model(model)
            msg = "Upload model through file sucess"
        except Exception as e:
            traceback.print_exc()
            msg = traceback.format_exc()
        return UploadModelResponse(message=msg)

            