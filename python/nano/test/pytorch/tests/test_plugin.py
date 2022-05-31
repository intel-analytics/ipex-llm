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

import os
import pytest
from unittest import TestCase

import torch
import torchmetrics
from torch import nn

from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch import Trainer

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from test.pytorch.utils._train_torch_lightning import create_test_data_loader
from test.pytorch.tests.test_lightning import ResNet18

import copy

num_classes = 10
batch_size = 32
dataset_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class TestPlugin(TestCase):
    model = ResNet18(pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_loader = create_data_loader(data_dir, batch_size, num_workers,
                                     data_transform, subset=dataset_size)
    test_data_loader = create_test_data_loader(data_dir, batch_size, num_workers,
                                               data_transform, subset=dataset_size)

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        project_test_dir = os.path.abspath(
            os.path.join(os.path.join(os.path.join(test_dir, ".."), ".."), "..")
        )
        os.environ['PYTHONPATH'] = project_test_dir

    def test_trainer_subprocess_plugin(self):
        pl_model = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer = Trainer(num_processes=2, distributed_backend="subprocess",
                          max_epochs=4)
        trainer.fit(pl_model, self.data_loader, self.test_data_loader)
        trainer.test(pl_model, self.test_data_loader)

    def test_trainer_subprocess_correctness(self):
        dataloader_1 = create_data_loader(data_dir, batch_size, num_workers,
                                     data_transform, subset=dataset_size, shuffle=False)
        pl_model_dis = LightningModuleFromTorch(
            self.model, self.loss, self.optimizer,
            metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        )
        trainer_dis = Trainer(num_processes=2, distributed_backend="subprocess", max_epochs=4)
        trainer_dis.tune(model=pl_model_dis, train_dataloaders=dataloader_1, scale_batch_size_kwargs={'max_trials':2})
        trainer_dis.fit(pl_model_dis, dataloader_1, dataloader_1)
        res_dis = trainer_dis.test(pl_model_dis, dataloader_1)
        print("distributed result", res_dis)

        # dataloader_2 = create_data_loader(data_dir, batch_size, num_workers,
        #                              data_transform, subset=dataset_size, shuffle=False)
        dataloader_2 = copy.deepcopy(dataloader_1)
        # pl_model_single = LightningModuleFromTorch(
        #     self.model, self.loss, self.optimizer,
        #     metrics=[torchmetrics.F1(num_classes), torchmetrics.Accuracy(num_classes=10)]
        # )
        pl_model_single = copy.deepcopy(pl_model_dis)
        trainer_single = Trainer(num_processes=1, max_epochs=4)
        trainer_single.tune(model=pl_model_single, train_dataloaders=dataloader_2, scale_batch_size_kwargs={'max_trials':3})
        trainer_single.fit(pl_model_single, dataloader_2, dataloader_2)
        
        res_single = trainer_single.test(pl_model_single, dataloader_2)
        print("single result", res_single)

        acc_single = res_single[0]['test/Accuracy_1']
        acc_dis = res_dis[0]['test/Accuracy_1']

        assert abs((acc_single-acc_dis))/max(acc_dis, acc_single) < 0.3, "distributed trained model accuracy should be close to non-distributed-trained model"
        return 

if __name__ == '__main__':
    pytest.main([__file__])
