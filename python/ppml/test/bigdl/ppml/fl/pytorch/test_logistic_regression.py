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

import torch

from bigdl.ppml.fl.pytorch.fl_server import FLServer
from torch import nn
import logging

from bigdl.ppml.fl.pytorch.pipeline import PytorchPipeline

resource_path = os.path.join(os.path.dirname(__file__), "../resources")

def mock_process(data_train):
    df_train = pd.read_csv(os.path.join(resource_path, data_train))
    if 'Outcome' in df_train:
        df_x = df_train.drop('Outcome', 1)
        df_y = df_train.filter(items=['Outcome'])        
    else:
        df_x = df_train
        df_y = None
    x = df_x.to_numpy(dtype="float32")
    y = df_y.to_numpy(dtype="long") if df_y is not None else None

    model = LogisticRegressionNetwork(len(df_x.columns))
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ppl = PytorchPipeline(model, loss_fn, optimizer)
    response = ppl.fit(x, y)
    logging.info(response)


class TestLogisticRegression(unittest.TestCase):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    def setUp(self) -> None:
        self.fl_server = FLServer()
        self.fl_server.build() 
        self.fl_server.start()


    def test_two_party_logistic_regression(self) -> None:
        mock_party1 = Process(target=mock_process, 
            args=('diabetes-vfl-1.csv',))
        mock_party1.start()
        mock_party2 = Process(target=mock_process, 
            args=('diabetes-vfl-2.csv',))
        mock_party2.start()
        mock_party1.join()
        mock_party2.join()


class LogisticRegressionNetwork(nn.Module):
    def __init__(self, num_feature) -> None:
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        x = self.dense(x)
        return x

if __name__ == '__main__':
    unittest.main()