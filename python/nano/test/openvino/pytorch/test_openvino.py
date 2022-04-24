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
from torchmetrics import F1
from bigdl.nano.pytorch.trainer import Trainer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import os
import shutil
from bigdl.nano.deps.openvino.pytorch.pytorch_openvino_model import PytorchOpenVINOModel


class TestOpenVINO(TestCase):
    def test_trainer_trace_openvino(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)
        
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        # trace a torch model 
        openvino_model = trainer.trace(model, x, 'openvino')
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        # trace pytorch-lightning model
        pl_model = Trainer.compile(model, loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, dataloader)

        openvino_model = trainer.trace(model, accelerator='openvino')
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        openvino_model.save('saved_openvino')
        assert len(os.listdir('saved_openvino')) > 0
        loaded_openvino_model = PytorchOpenVINOModel.load('saved_openvino')
        y_hat = loaded_openvino_model(x)
        assert y_hat.shape == (10, 10)

        shutil.rmtree('saved_openvino')

    def test_trainer_save_openvino(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)
        x = torch.rand((10, 3, 256, 256))

        # save and load pytorch model
        openvino_model = trainer.trace(model, accelerator='openvino', input_sample=x)
        trainer.save(openvino_model, 'trainer_save_openvino_model')
        assert len(os.listdir('trainer_save_openvino_model')) > 0
        loaded_openvino_model = trainer.load('trainer_save_openvino_model')
        y_hat = loaded_openvino_model(x)
        assert y_hat.shape == (10, 10)
        shutil.rmtree('trainer_save_openvino_model')
