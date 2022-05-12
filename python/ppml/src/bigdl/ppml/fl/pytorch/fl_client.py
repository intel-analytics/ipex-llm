


import pickle
import grpc
from numpy import ndarray
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2 import TrainRequest, UploadModelRequest
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.pytorch.protobuf_utils import ndarray_map_to_tensor_map
import uuid
from torch.utils.data import DataLoader

class FLClient(object):
    def __init__(self) -> None:
        self.channel = grpc.insecure_channel("localhost:8980")
        self.nn_stub = NNServiceStub(self.channel)
        self.client_uuid = str(uuid.uuid4())

    
    def train(self, x):
        tensor_map = ndarray_map_to_tensor_map(x)
        train_request = TrainRequest(clientuuid=self.client_uuid,
                                     data=tensor_map)
        
        response = self.nn_stub.train(train_request)
        if response.code == 1:
            raise Exception(response.response)
        return response

    def upload_model(self, model):
        # upload model to server
        model = pickle.dumps(model)        
        self.nn_stub.upload_model(UploadModelRequest(model_bytes=model))


    
