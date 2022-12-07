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
from unittest import TestCase
import tempfile

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

import numpy as np

from bigdl.nano.pytorch import Trainer, InferenceOptimizer
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


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(28 * 28, 128)
        self.layer_3 = nn.Linear(256, 2)

    def forward(self, x1, x2):
        x1 = self.layer_1(x1)
        x2 = self.layer_2(x2)
        x = torch.cat([x1, x2], axis=1)

        return self.layer_3(x)


class TestOnnx(TestCase):
    def test_trainer_trace_onnx(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, train_loader)

        onnx_model = InferenceOptimizer.trace(pl_model, accelerator="onnxruntime", input_sample=train_loader)

        for x, y in train_loader:
            model.eval()
            with torch.no_grad():
                forward_res_pytorch = pl_model(x).numpy()
            forward_res_onnx = onnx_model(x).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_pytorch, decimal=5)

        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)

    def test_trainer_trace_multiple_input_onnx(self):
        model = MultiInputModel()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x1 = torch.randn(100, 28 * 28)
        x2 = torch.randn(100, 28 * 28)
        y = torch.zeros(100).long()
        y[0:50] = 1
        train_loader = DataLoader(TensorDataset(x1, x2, y), batch_size=32, shuffle=True)
        trainer.fit(pl_model, train_loader)

        onnx_model = InferenceOptimizer.trace(pl_model, accelerator="onnxruntime",
                                              input_sample=train_loader)

        for x1, x2, y in train_loader:
            model.eval()
            with torch.no_grad():
                forward_res_pytorch = pl_model(x1, x2).numpy()
            forward_res_onnx = onnx_model(x1, x2).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_pytorch, decimal=5)

        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)
        trainer.predict(onnx_model, train_loader)

    def test_onnx_trainer_save_load(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, train_loader)

        onnx_model = InferenceOptimizer.trace(pl_model,
                                              accelerator="onnxruntime",
                                              input_sample=train_loader,
                                              thread_num=1)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            onnx_model_new = InferenceOptimizer.load(tmp_dir_name)

        assert onnx_model_new.session_options.intra_op_num_threads == 1
        assert onnx_model_new.session_options.inter_op_num_threads == 1

        for x, y in train_loader:
            forward_res_onnx = onnx_model(x).numpy()
            forward_res_onnx_new = onnx_model_new(x).numpy()
            np.testing.assert_almost_equal(forward_res_onnx, forward_res_onnx_new, decimal=5)

    def test_onnx_context_manager(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, train_loader)

        onnx_model = InferenceOptimizer.trace(pl_model,
                                              accelerator="onnxruntime",
                                              input_sample=train_loader,
                                              thread_num=1)

        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)
    
    def test_onnx_additional_attributes(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, train_loader)
        # patch a attribute
        pl_model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        pl_model.hello = hello

        onnx_model = InferenceOptimizer.trace(pl_model,
                                              accelerator="onnxruntime",
                                              input_sample=train_loader,
                                              thread_num=1)

        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 1
            output = onnx_model(x)

        assert onnx_model.channels == 3
        onnx_model.hello()
        with pytest.raises(AttributeError):
            onnx_model.width

    def test_onnx_default_values(self):
        # default bool values
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, a=True, b=False):
                if a:
                    return x+1
                if b:
                    return x-1
                return x

        model = Net()

        data = torch.rand(1,3,1,1)
        result_true = model(data)
        # sample with only required parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="onnxruntime",
                                            input_sample=(torch.rand(2,3,1,1),))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="onnxruntime",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        data = torch.rand(1,3,1,1)
        result_true = model(data, False, True)
        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="onnxruntime",
                                            input_sample=(torch.rand(2,3,1,1),False,True))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        # typehint model
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x: torch.Tensor, y: int = 3):
                return x+y

        model = Net()

        x = torch.rand(1,3,1,1)
        y = 3
        result_true = model(x, y)
        # sample with only required parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="onnxruntime",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(x, np.array([y]))  # TODO: make y work here
        assert torch.equal(result_true, result_m)

        # sample with only all parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="onnxruntime",
                                            input_sample=(torch.rand(2,3,1,1),3))
        result_m = accmodel(x, np.array([y]))  # TODO: make y work here
        assert torch.equal(result_true, result_m)


if __name__ == '__main__':
    pytest.main([__file__])
