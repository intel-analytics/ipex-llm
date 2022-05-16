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
import unittest
import numpy as np
import pandas as pd
import os

from bigdl.ppml.fl import *
from bigdl.ppml.fl.pytorch.fl_server import FLServer
from bigdl.ppml.fl.pytorch.fl_client import FLClient
from torch import nn
import torch.nn.functional as F

resource_path = os.path.join(os.path.dirname(__file__), "../resources")


class TestProtobufUtils(unittest.TestCase):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    def setUp(self) -> None:
        self.fl_server = FLServer()
        self.fl_server.build()
        self.fl_server.start()
        # this explicit set is needed, default value is 'fork' on Unix
        # if 'fork', the resources would be inherited and thread crash would occur
        # (to be verified)
    
    def tearDown(self) -> None:
        self.fl_server.stop()

    def test_upload_model(self) -> None:
        cli = FLClient()
        model = SimpleNNModel()
        logging.debug('uploading model to server')
        cli.upload_model(model)


class SimpleNNModel(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=10, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

if __name__ == '__main__':
    unittest.main()
