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


def customized_collate_fn(batch):
    batch, targets = zip(*batch)
    batch = torch.stack(batch, dim=0)
    targets = torch.stack(targets, dim=0)
    batch = batch.permute(0, 3, 1, 2).contiguous()
    return batch, targets


class TestOnnx(TestCase):

    def test_trainer_compile_with_onnx_quantize(self):
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

        # normal usage without tunning
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader)
        for x, y in train_loader:
            forward_res = onnx_model(x).numpy()
            # np.testing.assert_almost_equal(y.numpy(), forward_res, decimal=5)  # same result
  
        #  quantization with tensor
        x = torch.rand((1, 3, 256, 256))
        y = torch.ones((1, ), dtype=torch.long)
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=(x, y))

        # quantization with tunning
        pl_model.eval()
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader,
                                                 metric=torchmetrics.F1(10),
                                                 accuracy_criterion={'relative': 0.99,
                                                                    'higher_is_better': True})
        for x, y in train_loader:
            forward_res = onnx_model(x).numpy()
            # np.testing.assert_almost_equal(y.numpy(), forward_res, decimal=5)  # same result

        # test with pytorch-lightning trainer functions
        trainer.validate(onnx_model, train_loader)
        trainer.test(onnx_model, train_loader)
        trainer.predict(onnx_model, train_loader)

        # save the quantized model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            loaded_onnx_model = InferenceOptimizer.load(tmp_dir_name)

        for x, y in train_loader:
            forward_res = loaded_onnx_model(x)

    def test_trainer_compile_with_onnx_quantize_customized_collate_fn(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)

        # normal usage without tunning
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader)
        for x, y in train_loader:
            forward_res = onnx_model(x).numpy()
            # np.testing.assert_almost_equal(y.numpy(), forward_res, decimal=5)  # same result

        # quantization with tunning
        pl_model.eval()
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader,
                                                 metric=torchmetrics.F1(10),
                                                 accuracy_criterion={'relative': 0.99,
                                                                     'higher_is_better': True})
        for x, y in train_loader:
            forward_res = onnx_model(x).numpy()
            # np.testing.assert_almost_equal(y.numpy(), forward_res, decimal=5)  # same result

        # save the quantized model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            loaded_onnx_model = InferenceOptimizer.load(tmp_dir_name)

        for x, y in train_loader:
            forward_res = loaded_onnx_model(x)

    def test_trainer_compile_with_onnx_quantize_context_manager(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)

        # normal usage without tunning
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader,
                                                 thread_num=2)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 2
            x = torch.rand((2, 3, 256, 256))
            output = onnx_model(x)
            assert output.shape == (2, 10)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            model = InferenceOptimizer.load(tmp_dir_name)

        with InferenceOptimizer.get_context(model):
            assert torch.get_num_threads() == 2
            output = model(x)

    def test_trainer_compile_with_onnx_quantize_additional_attributes(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(max_epochs=1)

        pl_model = Trainer.compile(model, loss, optimizer)
        x = torch.rand((10, 256, 256, 3))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=2, collate_fn=customized_collate_fn)
        trainer.fit(pl_model, train_loader)
        # patch a attribute
        pl_model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        pl_model.hello = hello

        # normal usage without tunning
        onnx_model = InferenceOptimizer.quantize(pl_model,
                                                 accelerator='onnxruntime',
                                                 method='qlinear',
                                                 calib_data=train_loader,
                                                 thread_num=2)
        with InferenceOptimizer.get_context(onnx_model):
            assert torch.get_num_threads() == 2
            x = torch.rand((2, 3, 256, 256))
            output = onnx_model(x)
            assert output.shape == (2, 10)

        assert onnx_model.channels == 3
        onnx_model.hello()
        with pytest.raises(AttributeError):
            onnx_model.width


if __name__ == '__main__':
    pytest.main([__file__])
