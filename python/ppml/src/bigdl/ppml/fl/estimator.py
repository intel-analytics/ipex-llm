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


import imp
import logging
from numpy import ndarray
import torch
from torch import nn
from bigdl.ppml.fl.nn.pytorch.estimator import PytorchEstimator
from torch.utils.data import DataLoader
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.ppml.fl.nn.tensorflow.estimator import TensorflowEstimator
from bigdl.ppml.fl.nn.utils import tensor_map_to_ndarray_map
from tensorflow.keras.models import Model

class Estimator:
    @staticmethod
    def from_torch(client_model: nn.Module,
                   loss_fn,
                   optimizer_cls,
                   optimizer_args={},
                   server_model=None,
                   client_model_path=None,
                   server_model_path=None):
        estimator = PytorchEstimator(model=client_model, 
                                     loss_fn=loss_fn,
                                     optimizer_cls=optimizer_cls,
                                     optimizer_args=optimizer_args,
                                     server_model=server_model,
                                     client_model_path=client_model_path,
                                     server_model_path=server_model_path)
        return estimator

    @staticmethod
    def from_keras(client_model: Model,
                   loss_fn,
                   optimizer_cls,
                   optimizer_args={},
                   server_model=None):
        estimator = TensorflowEstimator(model=client_model, 
                                        loss_fn=loss_fn, 
                                        optimizer_cls=optimizer_cls,
                                        optimizer_args=optimizer_args,
                                        server_model=server_model)        
        return estimator
