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
import pytest
import os
import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from bigdl.nano.pytorch.onnx.onnxruntime_support import onnxruntime_support

# adaptted from 
# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
# model definition
@onnxruntime_support()
class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

# data preparation (not using mnist due to the poor network connection)
def dataloader_creator(size=1000, batch_size=64):
    random_mnist_x = torch.rand(size, 1, 28, 28)
    random_mnist_y = torch.randint(9, (size, ))
    return DataLoader(TensorDataset(random_mnist_x,
                                    random_mnist_y),
                      batch_size=batch_size,
                      shuffle=True)

class TestAutoTrainer(TestCase):
    mnist_train = dataloader_creator(size=1000, batch_size=64)
    mnist_test = dataloader_creator(size=100, batch_size=1)
    model = LitMNIST()
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, mnist_train)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_inference_onnx(self):
        random_mnist_x = torch.rand(1, 1, 28, 28)
        random_mnist_x_numpy = random_mnist_x.numpy()
        random_mnist_y_onnx = self.model.inference(random_mnist_x_numpy, backend="onnx")
        random_mnist_y = self.model.inference(random_mnist_x, backend=None).numpy()
        np.testing.assert_almost_equal(random_mnist_y, random_mnist_y_onnx, decimal=5)

    def test_trainer_predict(self):
        self.trainer.predict(self.model, dataloaders=self.mnist_test)

    def test_update_onnx_session(self):
        random_mnist_x = torch.rand(1, 1, 28, 28)
        random_mnist_x_numpy = random_mnist_x.numpy()
        random_mnist_y_onnx = self.model.inference(random_mnist_x_numpy, backend="onnx")
        assert self.model._ortsess_up_to_date is True
        self.trainer.fit(model, mnist_train)
        assert self.model._ortsess_up_to_date is False