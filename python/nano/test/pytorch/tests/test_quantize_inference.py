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


import pytest
import os
import tempfile
from unittest import TestCase

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch.vision.models import vision

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)

class TestQuantizeInference(TestCase):

    def test_quantized_model_inference(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        train_loader = create_data_loader(data_dir, batch_size, \
                                          num_workers, data_transform, subset=200)
        trainer.fit(pl_model, train_loader)
        pl_model = trainer.quantize(pl_model, train_loader)
        print(pl_model._quantized_model_up_to_date)

        for x, y in train_loader:
            quantized_res = pl_model.inference(x, backend=None, quantize=True).numpy()  # quantized
            pl_model.eval(quantize=True)
            with torch.no_grad():
                forward_res = pl_model(x).numpy()
            assert pl_model._quantized_model_up_to_date is True  # qmodel is up-to-date while inferencing
            np.testing.assert_almost_equal(quantized_res, forward_res, decimal=5)  # same result
        
        trainer.fit(pl_model, train_loader)
        assert pl_model._quantized_model_up_to_date is False  # qmodel is not up-to-date after training

        # test save/load dict
        pl_model = trainer.quantize(pl_model, train_loader)
        assert pl_model._quantized_model_up_to_date is True  # qmodel is up-to-date after building

        model_load = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        pl_model_load = Trainer.compile(model_load)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, ".ckpt")
            torch.save(pl_model.quantized_state_dict(), ckpt_name)
            pl_model_load.load_quantized_state_dict(torch.load(ckpt_name))
        
        for x, y in train_loader:
            quantized_res = pl_model.inference(x, backend=None, quantize=True).numpy()  # quantized
            quantized_res_load = pl_model_load.inference(x, backend=None, quantize=True).numpy()  # quantized
            np.testing.assert_almost_equal(quantized_res, quantized_res_load, decimal=5)  # same result
