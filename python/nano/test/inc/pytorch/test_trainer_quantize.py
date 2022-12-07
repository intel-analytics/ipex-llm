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
import tempfile
from unittest import TestCase

import pytest
import torch
from pytorch_lightning import LightningModule
from torch import nn
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
import torchmetrics

from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.vision.models import vision

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


class LitResNet18(LightningModule):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.classify = nn.Sequential(backbone, head)

    def forward(self, *args):
        return self.classify(args[0])


class TestTrainer(TestCase):
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    user_defined_pl_model = LitResNet18(10)

    def test_quantize_inc_ptq_compiled(self):
        # Test if a Lightning Module compiled by nano works
        train_loader_iter = iter(self.train_loader)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        x = next(train_loader_iter)[0]

        # Case 1: Default
        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=self.train_loader)
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

        # Case 2: Override by arguments
        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=self.train_loader,
                                             metric=torchmetrics.F1(10),
                                             approach='static',
                                             tuning_strategy='basic',
                                             accuracy_criterion={'relative': 0.99,
                                                                 'higher_is_better': True})

        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

        # Case 3: Dynamic quantization
        qmodel = InferenceOptimizer.quantize(pl_model, approach='dynamic')
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

        # Case 4: Invalid approach
        invalid_approach = 'qat'
        with pytest.raises(RuntimeError, match="Approach should be 'static' or 'dynamic', "
                                               "{} is invalid.".format(invalid_approach)):
            InferenceOptimizer.quantize(pl_model, approach=invalid_approach)

        # Case 5: Test if registered metric can be fetched successfully
        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=self.train_loader,
                                             metric=torchmetrics.F1(10),
                                             accuracy_criterion={'relative': 0.99,
                                                                'higher_is_better': True})
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

        trainer.validate(qmodel, self.train_loader)
        trainer.test(qmodel, self.train_loader)
        trainer.predict(qmodel, self.train_loader)

        # save and load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(qmodel, tmp_dir_name)
            loaded_qmodel = InferenceOptimizer.load(tmp_dir_name, pl_model)
            assert loaded_qmodel
            out = loaded_qmodel(x)
            assert out.shape == torch.Size([256, 10])

    def test_quantize_inc_ptq_customized(self):
        # Test if a Lightning Module not compiled by nano works
        train_loader_iter = iter(self.train_loader)
        x = next(train_loader_iter)[0]
        trainer = Trainer(max_epochs=1)

        qmodel = InferenceOptimizer.quantize(self.user_defined_pl_model,
                                             calib_data=self.train_loader)
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

        # save and load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(qmodel, tmp_dir_name)
            loaded_qmodel = InferenceOptimizer.load(tmp_dir_name, self.user_defined_pl_model)
            assert loaded_qmodel
            out = loaded_qmodel(x)
            assert out.shape == torch.Size([256, 10])

    def test_quantize_inc_ptq_with_tensor(self):
        train_loader_iter = iter(self.train_loader)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        x = next(train_loader_iter)[0]

        # Case 1: quantize with single tensor
        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=x)
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])
        
        # Case 2: quantize with tensor tuple
        qmodel = InferenceOptimizer.quantize(pl_model,
                                             # fake a label
                                             calib_data=(x, torch.ones(1)))
        assert qmodel
        out = qmodel(x)
        assert out.shape == torch.Size([256, 10])

    def test_quantize_inc_ptq_compiled_context_manager(self):
        # Test if a Lightning Module compiled by nano works
        train_loader_iter = iter(self.train_loader)
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        x = next(train_loader_iter)[0]

        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=self.train_loader,
                                             thread_num=2)
        assert qmodel

        with InferenceOptimizer.get_context(qmodel):
            assert torch.get_num_threads() == 2
            out = qmodel(x)
        assert out.shape == torch.Size([256, 10])
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(qmodel, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name, pl_model)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            out = model(x)
        assert out.shape == torch.Size([256, 10])

    def test_quantize_inc_ptq_compiled_additional_attributes(self):
        # Test if a Lightning Module compiled by nano works
        train_loader_iter = iter(self.train_loader)
        pl_model = Trainer.compile(self.model, self.loss, self.optimizer)
        # patch a attribute
        pl_model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        pl_model.hello = hello
        x = next(train_loader_iter)[0]

        qmodel = InferenceOptimizer.quantize(pl_model,
                                             calib_data=self.train_loader,
                                             thread_num=2)
        assert qmodel
        assert qmodel.channels == 3
        qmodel.hello()

        with InferenceOptimizer.get_context(qmodel):
            assert torch.get_num_threads() == 2
            out = qmodel(x)
        assert out.shape == torch.Size([256, 10])
