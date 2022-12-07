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
from bigdl.nano.pytorch import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import tempfile
import pytest


class TestOpenVINO(TestCase):
    def test_trainer_quantize_openvino(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        # Case1: Trace and quantize
        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)
        optimized_model = InferenceOptimizer.quantize(openvino_model, accelerator='openvino',
                                                      calib_data=dataloader)
        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        # Case2: Quantize directly from pytorch
        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader)

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)

    def test_trainer_quantize_openvino_with_tuning(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader,
                                                      metric=F1(10))

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)

    def test_trainer_quantize_openvino_option(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=dataloader,
                                                      openvino_config={"PERFORMANCE_HINT": "LATENCY"})

        optimized_model(x[0:1])
    
    def test_quantize_openvino_with_tensor(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((1, 3, 256, 256))
        y = torch.ones((1, ), dtype=torch.long)

        optimized_model = InferenceOptimizer.quantize(model, accelerator='openvino',
                                                      calib_data=(x, y))

        optimized_model(x[0:1])

    def test_pytorch_openvino_model_context_manager(self):
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.quantize(model,
                                                     accelerator='openvino',
                                                     calib_data=dataloader,
                                                     metric=F1(10),
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
        model = mobilenet_v3_small(num_classes=10)
        # patch a attribute
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        openvino_model = InferenceOptimizer.quantize(model,
                                                     accelerator='openvino',
                                                     calib_data=dataloader,
                                                     metric=F1(10),
                                                     thread_num=2)
        assert openvino_model.channels == 3
        openvino_model.hello()
        with pytest.raises(AttributeError):
            openvino_model.width

        with InferenceOptimizer.get_context(openvino_model):
            assert torch.get_num_threads() == 2
            y1 = openvino_model(x[0:1])
