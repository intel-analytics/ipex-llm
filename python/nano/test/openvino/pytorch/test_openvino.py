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
from tempfile import TemporaryDirectory
from unittest import TestCase
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import os
import tempfile
import pytest


class TestOpenVINO(TestCase):
    def test_trainer_trace_openvino(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        # trace a torch model
        openvino_model = InferenceOptimizer.trace(model, x, 'openvino')
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        # trace pytorch-lightning model
        pl_model = Trainer.compile(model, loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)
        trainer.fit(pl_model, dataloader)

        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino')
        y_hat = openvino_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = openvino_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(openvino_model, dataloader)
        trainer.test(openvino_model, dataloader)
        trainer.predict(openvino_model, dataloader)

    def test_trainer_save_openvino(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)
        x = torch.rand((10, 3, 256, 256))

        # save and load pytorch model
        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)
        with TemporaryDirectory() as saved_root:
            InferenceOptimizer.save(openvino_model, saved_root)
            assert len(os.listdir(saved_root)) > 0
            loaded_openvino_model = InferenceOptimizer.load(saved_root)
            y_hat = loaded_openvino_model(x[0:3])
            assert y_hat.shape == (3, 10)
            y_hat = loaded_openvino_model(x)
            assert y_hat.shape == (10, 10)

    def test_pytorch_openvino_model_async_predict(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x, accelerator='openvino')
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        # do async_predict use dataloader as input
        result = openvino_model.async_predict(dataloader, num_requests=3)
        for res in result:
            assert res.shape == (2, 10)

        # do async_predict use List of Tensor as input
        x = [torch.rand((10, 3, 256, 256)) for i in range(3)]
        result = openvino_model.async_predict(x, num_requests=3)
        for res in result:
            assert res.shape == (10, 10)

    def test_pytorch_openvino_model_option(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
                                                  openvino_config={"PERFORMANCE_HINT": "LATENCY"})

        result = openvino_model(x[0:1])

    def test_pytorch_openvino_model_context_manager(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
                                                  thread_num=2)

        with InferenceOptimizer.get_context(openvino_model):
            assert torch.get_num_threads() == 2
            y1 = openvino_model(x[0:1])

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            y2 = model(x[0:1])

    def test_pytorch_openvino_model_additional_attrs(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)
        # patch a attribute
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        openvino_model = InferenceOptimizer.trace(model, input_sample=x,
                                                  accelerator='openvino',
                                                  thread_num=2)
        assert openvino_model.channels == 3
        openvino_model.hello()
        with pytest.raises(AttributeError):
            openvino_model.width

        with InferenceOptimizer.get_context(openvino_model):
            assert torch.get_num_threads() == 2
            y1 = openvino_model(x[0:1])

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            y2 = model(x[0:1])

    def test_openvino_default_values(self):
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
                                            accelerator="openvino",
                                            input_sample=(torch.rand(2,3,1,1),))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(data)
        assert torch.equal(result_true, result_m)

        data = torch.rand(1,3,1,1)
        result_true = model(data, False, True)
        # sample with only required parameters
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
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
                                            accelerator="openvino",
                                            input_sample=torch.rand(2,3,1,1))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)

        # sample with only all parameters (in a tuple)
        accmodel = InferenceOptimizer.trace(model,
                                            accelerator="openvino",
                                            input_sample=(torch.rand(2,3,1,1), 3))
        result_m = accmodel(x, y)
        assert torch.equal(result_true, result_m)
