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
import grpc
from numpy import ndarray
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, UploadModelRequest
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
import uuid
from torch.utils.data import DataLoader
from bigdl.dllib.utils.log4Error import invalidInputError

from bigdl.ppml.fl.nn.utils import ClassAndArgsWrapper

class FLClient(object):
    channel = None
    nn_stub = NNServiceStub(channel)
    def __init__(self, aggregator, target="localhost:8980") -> None: 
        if FLClient.channel == None:
            FLClient.channel = grpc.insecure_channel(target)
        self.client_uuid = str(uuid.uuid4())
        self.aggregator = aggregator
    
    def train(self, x):
        tensor_map = ndarray_map_to_tensor_map(x)
        train_request = TrainRequest(clientuuid=self.client_uuid,
                                     data=tensor_map,
                                     algorithm=self.aggregator)
        
        response = FLClient.nn_stub.train(train_request)
        if response.code == 1:
            invalidInputError(False,
                              response.response)
        return response

    def upload_model(self, model, loss_fn, optimizer_cls, optimizer_args):
        # upload model to server
        model = pickle.dumps(model)
        loss_fn = pickle.dumps(loss_fn)
        optimizer = ClassAndArgsWrapper(optimizer_cls, optimizer_args).to_protobuf()
        request = UploadModelRequest(client_uuid=self.client_uuid,
                                     model_bytes=model,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     aggregator=self.aggregator)
        return self.nn_stub.upload_model(request)


    
