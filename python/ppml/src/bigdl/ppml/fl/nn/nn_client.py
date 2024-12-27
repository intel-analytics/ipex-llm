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


from bigdl.ppml.utils.safepickle import SafePickle
from bigdl.ppml.fl.nn.fl_client import FLClient
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, PredictRequest, UploadMetaRequest
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
import threading
from bigdl.dllib.utils.log4Error import invalidInputError, invalidOperationError

from bigdl.ppml.fl.nn.utils import ClassAndArgsWrapper
from numpy import ndarray
from typing import Dict, Type, Union
from nn_service_pb2 import PredictResponse, TrainResponse, UploadMetaResponse

class NNClient(object):
    _lock = threading.Lock()

    def __init__(self, aggregator: str) -> None:        
        if FLClient.channel is None:
            invalidOperationError(False, "No channel found, please make sure you called \
                init_fl_context()")
        if FLClient.client_id is None:
            invalidOperationError(False, "You have to set client_id with integer like: \
                init_fl_context(client_id=1)")
        self.nn_stub = NNServiceStub(FLClient.channel)
        self.client_uuid = FLClient.client_id
        self.aggregator = aggregator
    
    def train(self, x: Dict[str, ndarray]) -> TrainResponse:
        tensor_map = ndarray_map_to_tensor_map(x)
        train_request = TrainRequest(clientuuid=self.client_uuid,
                                     data=tensor_map,
                                     algorithm=self.aggregator)
        
        response = self.nn_stub.train(train_request)
        if response.code == 1:
            invalidInputError(False,
                              response.response)
        return response

    def predict(self, x: Dict[str, ndarray]) -> PredictResponse:
        tensor_map = ndarray_map_to_tensor_map(x)
        predict_request = PredictRequest(clientuuid=self.client_uuid,
                                     data=tensor_map,
                                     algorithm=self.aggregator)
        
        response = self.nn_stub.predict(predict_request)
        if response.code == 1:
            invalidInputError(False,
                              response.response)
        return response

    def upload_meta(
        self,
        loss_fn,
        optimizer_cls,
        optimizer_args: Dict[str, float]
    ) -> UploadMetaResponse:
        # upload model to server
        loss_fn = SafePickle.dumps(loss_fn)
        optimizer = ClassAndArgsWrapper(optimizer_cls, optimizer_args).to_protobuf()
        request = UploadMetaRequest(client_uuid=self.client_uuid,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    aggregator=self.aggregator)
        return self.nn_stub.upload_meta(request)
