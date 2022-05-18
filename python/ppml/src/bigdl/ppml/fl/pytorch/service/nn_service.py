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


from bigdl.ppml.fl.pytorch.aggregator import Aggregator
from bigdl.ppml.fl.pytorch.generated.fl_base_pb2 import TensorMap
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2 import TrainRequest, TrainResponse, UploadModelResponse
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.pytorch.protobuf_utils import tensor_map_to_ndarray_map
import pickle
import traceback


class NNServiceImpl(NNServiceServicer):
    def __init__(self, **kargs) -> None:
        self.aggregator = Aggregator(**kargs)

    def train(self, request: TrainRequest, context):
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        try:
            self.aggregator.put_client_data(client_id, ndarray_map)            
            msg = f'[client {client_id} batch trained]'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            code = 1
        
        return TrainResponse(response=msg, data=self.aggregator.server_data, code=code)

    def evaluate(self, request, context):
        return super().evaluate(request, context)

    def predict(self, request, context):
        return super().predict(request, context)
        
    def upload_model(self, request, context):
        model = pickle.loads(request.model_bytes)
        self.aggregator.add_server_model(model)
        return UploadModelResponse(message="Upload sucess")