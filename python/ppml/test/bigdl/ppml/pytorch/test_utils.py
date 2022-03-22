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
from bigdl.ppml.pytorch.pipeline import Pipeline
from bigdl.ppml.utils import init_fl_context


resource_path = os.path.join(os.path.dirname(__file__), "../resources")




class TestUtils(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls) -> None:
    #     multiprocessing.set_start_method('spawn') 

    # def setUp(self) -> None:
    #     self.fl_server = FLServer()
    #     init_fl_context()

    # def tearDown(self) -> None:
    #     self.fl_server.stop()

    # test if data in Pytorch Tensor could be directly replaced
    # currently, we use a trick to replace the data of Tensor with the result from FLServer
    # but remain other attributes, (e.g. backward steps, params) unchanged
    # this trick would only work if following test could pass
    def test_set_tensor_data(self):        
        t = torch.tensor([1, 2])
        t.data = torch.tensor([1])
        t
    

if __name__ == '__main__':
    unittest.main()
