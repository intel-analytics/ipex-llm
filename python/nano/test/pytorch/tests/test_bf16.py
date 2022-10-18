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
from unittest.mock import MagicMock, Mock, PropertyMock, patch
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_12


class Pytorch1_9:
    """Pytorch version < 1.10, bfloat16 precision is not supported."""
    def test_bf16_pytorch_less_1_10(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.10 to convert type as bfloat16."
        ):
            trainer.quantize(model, precision='bf16')


class Pytorch1_11:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """
    def test_bf16_pytorch_1_11(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = trainer.quantize(model, precision='bf16')
        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.12 to obtain bfloat16 acceleration."
        ):
            y_hat = bf16_model(x)


class Pytorch1_12:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """
    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa", new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_has_bf16_isa):
        mocked_has_bf16_isa.return_value = False
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = trainer.quantize(model, precision='bf16')
        with pytest.raises(RuntimeError,
                           match="Your machine or OS doesn't support BF16 instructions."):
            y_hat = bf16_model(x)

    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._max_bf16_isa", return_value=None)
    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa", new_callable=PropertyMock)
    def test_not_executed_on_bf16(self, mocked_has_bf16_isa, mocked_max_bf16_isa):
        """
        Pytorch version is correct and bf16 instructions are detected.
        But somehow in the run, there is no bf16 instructions used.
        """
        mocked_has_bf16_isa.return_value = True
        mocked_max_bf16_isa.return_value = None
        
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))

        bf16_model = trainer.quantize(model, precision='bf16')
        with pytest.raises(
            RuntimeError,
            match="BF16 ISA support is not enabled under current context."
        ):
            bf16_model(x)

    @patch.dict('os.environ', {"ALLOW_NON_BF16_ISA": "1"})
    def test_bf16_common(self):
        """
        Debug mode. Allow run bf16 forward without bf16 instruction support.
        """
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16')
        # Debug mode to test functionality, make sure forward is called sucessfully
        y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16


    def test_bf16_with_amx_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AMX")
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_avx512_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AVX512")
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

TORCH_VERSION_CLS = Pytorch1_12
if TORCH_VERSION_LESS_1_12:
    print("1.11")
    TORCH_VERSION_CLS = Pytorch1_11
if TORCH_VERSION_LESS_1_10:
    print("1.9")
    TORCH_VERSION_CLS = Pytorch1_9


class TestBF16(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
