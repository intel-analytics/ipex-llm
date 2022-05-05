
from bigdl.ppml.fl.pytorch.aggregator import Aggregator
from bigdl.ppml.fl.pytorch.generated.fl_base_pb2 import TensorMap
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2 import TrainRequest, TrainResponse
from bigdl.ppml.fl.pytorch.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.pytorch.protobuf_utils import tensor_map_to_ndarray_map



class NNServiceImpl(NNServiceServicer):
    def __init__(self) -> None:
        self.aggregator = Aggregator()

    def train(self, request: TrainRequest, context):
        tensor_map = request.data.tensorMap
        tensor_map_to_ndarray_map(tensor_map)
        print("tensor map decoded at server")
        return TrainResponse()

    def evaluate(self, request, context):
        return super().evaluate(request, context)

    def predict(self, request, context):
        return super().predict(request, context)
        