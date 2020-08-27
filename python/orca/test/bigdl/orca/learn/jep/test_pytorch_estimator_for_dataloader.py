#
# Copyright 2018 Analytics Zoo Authors.
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
import os
from unittest import TestCase

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np

from zoo.orca.data.pandas import read_csv
from zoo.orca.learn.pytorch import Estimator
from zoo.common.nncontext import *
from bigdl.optim.optimizer import Adam
from zoo.orca.learn.metrics import *
from zoo.orca.learn.trigger import EveryEpoch

resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")


class TestEstimatorForSparkDataLoader(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_bigdl_pytorch_estimator_dataloader(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = self.bn1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        model = SimpleModel()

        estimator = Estimator.from_torch(model=model, loss=nn.BCELoss(),
                                         optimizer=Adam(), backend="bigdl")

        inputs = torch.Tensor([[1, 2], [1, 3], [3, 2], [5, 6], [8, 9], [1, 9]])
        targets = torch.Tensor([[0], [0], [0], [1], [1], [1]])
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
        )
        val_loader = torch.utils.data.DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
        )
        estimator.fit(data=train_loader, epochs=2, validation_data=val_loader,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
        estimator.evaluate(data=val_loader, validation_methods=[Accuracy()])


if __name__ == "__main__":
    pytest.main([__file__])
