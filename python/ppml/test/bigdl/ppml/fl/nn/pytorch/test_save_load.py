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

from multiprocessing import Process
import time
from typing import List
import unittest
import numpy as np
import pandas as pd
import os

from bigdl.ppml.fl import *
from bigdl.ppml.fl.nn.fl_server import FLServer
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.estimator import Estimator

from torch import Tensor, nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from bigdl.ppml.fl.utils import FLTest
import shutil

resource_path = os.path.join(os.path.dirname(__file__), "../../resources")


class TestSaveLoad(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    server_model_path = '/tmp/vfl_server_model'
    client_model_path = '/tmp/vfl_client_model'
    def setUp(self) -> None:
        self.fl_server = FLServer(client_num=1)
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()
    
    def tearDown(self) -> None:
        self.fl_server.stop()
        if os.path.exists(TestSaveLoad.server_model_path):
            shutil.rmtree(TestSaveLoad.server_model_path)
        if os.path.exists(TestSaveLoad.client_model_path):
            os.remove(TestSaveLoad.client_model_path)

    def test_mnist(self) -> None:
        """
        following code is copied from pytorch quick start
        link: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
        """
        
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        batch_size = 64

        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        loss_fn = nn.CrossEntropyLoss()

        # list for result validation
        init_fl_context(1, self.target)
        vfl_model_1 = NeuralNetworkPart1()
        vfl_model_2 = NeuralNetworkPart2()
        vfl_client_ppl = Estimator.from_torch(client_model=vfl_model_1,
                                              loss_fn=loss_fn,
                                              optimizer_cls=torch.optim.SGD,
                                              optimizer_args={'lr':1e-3},
                                              server_model=vfl_model_2,
                                              server_model_path=TestSaveLoad.server_model_path,
                                              client_model_path=TestSaveLoad.client_model_path)
        vfl_client_ppl.fit(train_dataloader)
        self.fl_server.stop()
        self.setUp()
        client_model_loaded = torch.load(TestSaveLoad.client_model_path)
        ppl_from_file = Estimator.from_torch(client_model=client_model_loaded,
                                             loss_fn=loss_fn,
                                             optimizer_cls=torch.optim.SGD,
                                             optimizer_args={'lr':1e-3})
        ppl_from_file.load_server_model(TestSaveLoad.server_model_path)
        ppl_from_file.fit(train_dataloader)

        assert ppl_from_file.loss_history[-1] < 2, \
            f"Validation failed, incremental training loss does not meet requirement, \
            required < 2, current {ppl_from_file.loss_history[-1]}"
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential_1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU()
        )
        self.sequential_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential_1(x)
        x = self.sequential_2(x)
        return x

class NeuralNetworkPart1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential_1(x)
        return x

class NeuralNetworkPart2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x: List[Tensor]):
        x = x[0] # this act as interactive layer, take the first tensor
        x = self.sequential_2(x)
        return x


if __name__ == '__main__':
    unittest.main()
