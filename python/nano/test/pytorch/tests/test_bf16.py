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
from unittest import TestCase
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from bigdl.nano.pytorch import Trainer
from torchvision.models.resnet import resnet18
from unittest.mock import MagicMock, PropertyMock, patch
from bigdl.nano.pytorch import utils


class TestLightningLess1_6(TestCase):
    @patch.object(utils, 'LIGHTNING_VERSION_LESS_1_6', True)
    @patch.object(utils, 'TORCH_VERSION_LESS_1_10', False)
    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', False)
    def test_bf16_lightning_less_1_6(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.12 and pytorch-lightning>=1.6.0."
        ):
            trainer.quantize(model, precision='bf16')


class TestPytorchLess1_10(TestCase):
    @patch.object(utils, 'TORCH_VERSION_LESS_1_10', True)
    def test_bf16_pytorch_less_1_10(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.12 and pytorch-lightning>=1.6.0."
        ):
            trainer.quantize(model, precision='bf16')


class TestPytorchLess1_12(TestCase):
    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', True)
    @patch.object(utils, 'TORCH_VERSION_LESS_1_10', False)
    def test_bf16_pytorch_less_1_12(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.12 and pytorch-lightning>=1.6.0."
        ):
            trainer.quantize(model, precision='bf16')


class TestBF16(TestCase):
    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', False)
    def test_bf16_with_amx_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        if utils.TORCH_VERSION_LESS_1_10:
            with pytest.raises(
                RuntimeError,
                match="Require torch>=1.12 and pytorch-lightning>=1.6.0."):
                bf16_model = trainer.quantize(model, precision='bf16')
            return
        bf16_model = trainer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AMX")
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', False)
    def test_bf16_with_avx512_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        if utils.TORCH_VERSION_LESS_1_10 or utils.LIGHTNING_VERSION_LESS_1_6:
            with pytest.raises(
                RuntimeError,
                match="Require torch>=1.12 and pytorch-lightning>=1.6.0."):
                bf16_model = trainer.quantize(model, precision='bf16')
            return
        bf16_model = trainer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AVX512")
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', False)
    def test_non_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        if utils.TORCH_VERSION_LESS_1_10 or utils.LIGHTNING_VERSION_LESS_1_6:
            with pytest.raises(
                RuntimeError,
                match="Require torch>=1.12 and pytorch-lightning>=1.6.0."):
                bf16_model = trainer.quantize(model, precision='bf16')
            return
        bf16_model = trainer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value=None)
            with pytest.raises(
                RuntimeError,
                match="BF16 ISA support is not enabled under current context."
            ):
                y_hat = bf16_model(x)

    @patch.object(utils, 'TORCH_VERSION_LESS_1_12', False)
    def test_bf16_common(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        if utils.TORCH_VERSION_LESS_1_10 or utils.LIGHTNING_VERSION_LESS_1_6:
            with pytest.raises(
                RuntimeError,
                match="Require torch>=1.12 and pytorch-lightning>=1.6.0."):
                bf16_model = trainer.quantize(model, precision='bf16')
            return
        bf16_model = trainer.quantize(model, precision='bf16')
        # Debug mode to test functionality, make sure forward is called sucessfully
        os.environ["ALLOW_NON_BF16_ISA"] = "1"
        y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        os.environ["ALLOW_NON_BF16_ISA"] = "0"

if __name__ == '__main__':
    pytest.main([__file__])
