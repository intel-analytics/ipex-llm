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
from typing import Dict, Type
from numpy import ndarray
import torch
from torch import nn
from bigdl.ppml.fl.nn.pytorch.estimator import PytorchEstimator
from bigdl.ppml.fl.nn.tensorflow.estimator import TensorflowEstimator
from tensorflow.keras.models import Model

class Estimator:
    @staticmethod
    def from_torch(client_model: nn.Module,
                   loss_fn: torch.nn.modules.loss._Loss,
                   optimizer_cls: Type[torch.optim.optimizer.Optimizer],
                   optimizer_args: Dict[str, float]={},
                   server_model: nn.Module=None,
                   client_model_path: str=None,
                   server_model_path: str=None):
        ''' 
        :param client_model: client nn model

        :param loss_fn: loss function. A function that calculate the absolute difference between our prediction and the actual value.

        :param optimizer_cls: optimizer class.

        :param optimizer_args: `dict` of hypterparamters that will be passed into `optimizer_cls`

        :param server_model: optional param to upload server_model

        :param client_model_path: optional string filepath to autosave client_model. Default None and will not auto save client_model.

        :param server_model_path: optional string filepath to autosave server_model. Default None and will not auto save server_model.
        '''
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
                   server_model=None,
                   client_model_path=None,
                   server_model_path=None):
        ''' 
        :param client_model: keras Model

        :param loss_fn: loss function. A function that calculate the absolute difference between our prediction and the actual value.

        :param optimizer_cls: optimizer class.

        :param optimizer_args: `dict` of hypterparamters that will be passed into `optimizer_cls`

        :param server_model: optional param to upload server_model
        
        :param client_model_path: optional string filepath to autosave client_model. Default None and will not auto save client_model.

        :param server_model_path: optional string filepath to autosave server_model. Default None and will not auto save server_model.
        '''   
        estimator = TensorflowEstimator(model=client_model, 
                                        loss_fn=loss_fn, 
                                        optimizer_cls=optimizer_cls,
                                        optimizer_args=optimizer_args,
                                        server_model=server_model,
                                        client_model_path=client_model_path,
                                        server_model_path=server_model_path)         
        return estimator
