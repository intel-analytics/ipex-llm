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
from unittest import TestCase

import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")


class TestEstimatorForSaveAndLoad(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_orca_context(cores=4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

    def test_bigdl_pytorch_estimator_save_and_load(self):
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

                self.fc1 = nn.Linear(28 * 28, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        model = Network()
        model.train()
        criterion = nn.NLLLoss()
        adam = torch.optim.Adam(model.parameters(), 0.001)

        dir = "/tmp/dataset/"
        batch_size = 320

        images = torch.randn(1000 * 28 * 28, dtype=torch.float32).view(1000, 1, 28, 28)
        labels = torch.randint(0, 10, (1000,), dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(images, labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # epoch 1
        est = Estimator.from_torch(model=model, optimizer=adam, loss=criterion,
                                   metrics=[Accuracy()])

        est.fit(data=train_loader, epochs=1, validation_data=test_loader,
                checkpoint_trigger=EveryEpoch())
        paras1 = list(est.get_model().named_parameters())
        est.save("model_epoch_1")

        # epoch 2
        est.fit(data=train_loader, epochs=2, validation_data=test_loader,
                checkpoint_trigger=EveryEpoch())
        paras2 = list(est.get_model().named_parameters())
        est.load("model_epoch_1")
        paras3 = list(est.get_model().named_parameters())

        load_success = 0
        for i in range(len(paras2)):
            name2, para2 = paras2[i]
            name3, para3 = paras3[i]
            if not torch.all(torch.eq(para2, para3)):
                load_success = 1
                break
        if not load_success:
            raise Exception("Load failed. Parameters did not change after loading.")

        for i in range(len(paras1)):
            name1, para1 = paras1[i]
            name3, para3 = paras3[i]
            if not torch.all(torch.eq(para1, para3)):
                raise Exception("After reloading the model," + name1 + "does not match.")
        print("pass")


if __name__ == "__main__":
    pytest.main([__file__])
