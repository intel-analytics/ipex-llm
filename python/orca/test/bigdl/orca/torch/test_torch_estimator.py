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
import torchvision
import pytest

from unittest import TestCase
from bigdl.orca.torch import TorchModel, TorchLoss
from bigdl.dllib.nncontext import *
from torch.utils.data import TensorDataset, DataLoader
from bigdl.dllib.estimator import *
from bigdl.dllib.keras.optimizers import Adam
from bigdl.dllib.optim.optimizer import MaxEpoch, EveryEpoch
from bigdl.dllib.keras.metrics import Accuracy
from bigdl.dllib.feature.common import FeatureSet


class TestPytorch(TestCase):

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

    def test_train_model_with_bn_creator(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = self.bn1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        self.sc.stop()
        self.sc = init_nncontext()
        torch_model = SimpleTorchModel()
        loss_fn = torch.nn.BCELoss()
        az_model = TorchModel.from_pytorch(torch_model)
        zoo_loss = TorchLoss.from_pytorch(loss_fn)

        def train_dataloader():
            inputs = torch.Tensor([[1, 2], [1, 3], [3, 2],
                                   [5, 6], [8, 9], [1, 9]])
            targets = torch.Tensor([[0], [0], [0],
                                    [1], [1], [1]])
            return DataLoader(TensorDataset(inputs, targets), batch_size=2)

        train_featureset = FeatureSet.pytorch_dataloader(train_dataloader)
        val_featureset = FeatureSet.pytorch_dataloader(train_dataloader)
        zooOptimizer = Adam()
        estimator = Estimator(az_model, optim_methods=zooOptimizer)
        estimator.train_minibatch(train_featureset, zoo_loss, end_trigger=MaxEpoch(4),
                                  checkpoint_trigger=EveryEpoch(),
                                  validation_set=val_featureset,
                                  validation_method=[Accuracy()])

        trained_model = az_model.to_pytorch()


if __name__ == "__main__":
    pytest.main([__file__])
