


import grpc
from numpy import ndarray
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2 import TrainRequest
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.pytorch.protobuf_utils import ndarray_map_to_tensor_map
import uuid

class FLClient(object):
    def __init__(self) -> None:
        self.channel = grpc.insecure_channel("localhost:8980")
        self.nn_stub = NNServiceStub(self.channel)
        self.client_uuid = str(uuid.uuid4())

    def train(self, x: dict):
        tensor_map = ndarray_map_to_tensor_map(x)
        train_request = TrainRequest(clientuuid=self.client_uuid,
                                     data=tensor_map)
        self.nn_stub.train(train_request)

    
