import numpy as np
from bigdl.ppml.fl.pytorch.generated.fl_base_pb2 import FloatTensor, TensorMap

def ndarray_map_to_tensor_map(array_map: dict):
    tensor_map = {}
    for (k, v) in array_map.items():
        if not isinstance(v, np.ndarray):
            raise Exception("ndarray map element should be Numpy ndarray")
        tensor_map[k] = FloatTensor(tensor=v.flatten().tolist(),shape=v.shape)
    return TensorMap(tensorMap=tensor_map)


def tensor_map_to_ndarray_map(tensor_map: TensorMap):
    ndarray_map = {}
    for (k, v) in tensor_map.items():
        if not isinstance(v, FloatTensor):
            raise Exception("tensor map element should be protobuf type FloatTensor")
        ndarray_map[k] = np.array(v.tensor).reshape(v.shape)
    return ndarray_map