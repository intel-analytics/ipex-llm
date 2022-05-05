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
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
from bigdl.ppml.fl.pytorch.fl_server import FLServer
from bigdl.ppml.fl.pytorch.fl_client import FLClient
from bigdl.ppml.fl.utils import init_fl_context

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

    def test_ndarray_tensor(self) -> None:
        logging.debug('client initializing')
        cli = FLClient()
        logging.debug('client initialized, start train with server')
        cli.train({'input': np.array([[1, 2], [3, 4]])})


if __name__ == '__main__':
    unittest.main()
