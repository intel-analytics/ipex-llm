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

import torch
from torch import nn
import torchvision
import pytest

from unittest import TestCase
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.common.nncontext import *


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

    def test_torchmodel_constructor(self):
        class TwoInputModel(nn.Module):
            def __init__(self):
                super(TwoInputModel, self).__init__()
                self.dense1 = nn.Linear(2, 2)
                self.dense2 = nn.Linear(3, 1)

            def forward(self, x1, x2):
                x1 = self.dense1(x1)
                x2 = self.dense2(x2)
                return x1, x2

        TorchModel.from_pytorch(TwoInputModel())

    def test_torchloss_constructor(self):
        criterion = nn.MSELoss()
        TorchLoss.from_pytorch(criterion)

    def test_torch_net_predict_resnet(self):
        torch.random.manual_seed(1)
        pytorch_model = torchvision.models.resnet18(pretrained=False).eval()
        zoo_model = TorchModel.from_pytorch(pytorch_model)
        zoo_model.evaluate()

        dummy_input = torch.ones(1, 3, 224, 224)
        pytorch_result = pytorch_model(dummy_input).data.numpy()
        zoo_result = zoo_model.forward(dummy_input.numpy())
        print(pytorch_result)
        print(zoo_result)
        assert np.allclose(pytorch_result, zoo_result, rtol=1.e-6, atol=1.e-6)

    def test_model_to_pytorch(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        torch_model = SimpleTorchModel()
        az_model = TorchModel.from_pytorch(torch_model)

        weights = az_model.get_weights()
        weights[0][0] = 1.0
        az_model.set_weights(weights)

        exported_model = az_model.to_pytorch()
        p = list(exported_model.parameters())
        assert p[0][0][0] == 1.0

    def test_model_with_bn_to_pytorch(self):
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

        torch_model = SimpleTorchModel()
        az_model = TorchModel.from_pytorch(torch_model)
        dummy_input = torch.ones(16, 2)
        zoo_result = az_model.forward(dummy_input.numpy())

        exported_model = az_model.to_pytorch()
        print(list(exported_model.named_buffers()))


if __name__ == "__main__":
    pytest.main([__file__])
