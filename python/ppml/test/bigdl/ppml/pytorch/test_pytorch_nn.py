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

import torch
from torch import nn
from multiprocessing import Process
import unittest
import numpy as np
import pandas as pd
import os

from bigdl.ppml import *
from bigdl.ppml.fl_server import FLServer
from bigdl.ppml.pytorch.pipeline import PytorchPipeline
from bigdl.ppml.utils import init_fl_context


resource_path = os.path.join(os.path.dirname(__file__), "../resources")

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Linear(2, 1)

    def forward(self, x):
        logits = self.nn(x)
        return logits





class TestPytorchNN(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls) -> None:
    #     multiprocessing.set_start_method('spawn') 

    def setUp(self) -> None:
        self.fl_server = FLServer()
        init_fl_context()

    # def tearDown(self) -> None:
    #     self.fl_server.stop()

    def test_dummy_data(self):
        model = SimpleNN()
        ppl = PytorchPipeline(model, nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=1e-3), algorithm="vfl_logistic_regression")
        x, y = np.ones([2, 2], dtype="float32"), np.ones([2], dtype="float32")
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        ppl.fit(x, y)
    

if __name__ == '__main__':
    unittest.main()
