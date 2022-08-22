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
import torch
from bigdl.nano.pytorch import Trainer
from torchvision.models.resnet import resnet18
from unittest.mock import MagicMock, Mock, PropertyMock, patch
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_11, TORCH_VERSION_LESS_1_12
from bigdl.nano.common import check_avx512


class Pytorch1_9:
    """Pytorch version < 1.10, bfloat16 precision is not supported."""
    def test_bf16_pytorch_less_1_10(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        with pytest.raises(
            RuntimeError,
            match="torch version should >=1.10 to use ipex"
        ):
            trainer.quantize(model, precision='bf16', use_ipex=True)


class CaseWithoutAVX512:
    @patch('bigdl.nano.deps.ipex.ipex_inference_bf16_model.PytorchIPEXJITBF16Model._has_bf16_isa', new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_has_bf16_isa):
        mocked_has_bf16_isa.return_value = False
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        with pytest.raises(RuntimeError,
                           match="Your machine or OS doesn't support BF16 instructions."):
            bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)

    @patch.dict('os.environ', {'ALLOW_NON_BF16_ISA': "1"})
    @patch('bigdl.nano.deps.ipex.ipex_inference_bf16_model.PytorchIPEXJITBF16Model._has_bf16_isa', new_callable=PropertyMock)
    def test_bf16_common(self, mocked_has_bf16_isa):
        """
        Debug mode. Allow run model without IPEX BF16 optimization.
        """
        mocked_has_bf16_isa.return_value = False
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)
        # Debug mode to test functionality, make sure forward is called sucessfully
        y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.float32


class Pytorch1_11(CaseWithoutAVX512):
    @patch("bigdl.nano.deps.ipex.ipex_inference_bf16_model.PytorchIPEXJITBF16Model._max_bf16_isa", return_value=None)
    def test_not_executed_on_bf16(self, mocked_max_bf16_isa):
        mocked_max_bf16_isa.return_value = None

        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))

        bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)
        with pytest.raises(
            RuntimeError,
            match="BF16 ISA support is not enabled under current context."
        ):
            y_hat = bf16_model(x)
            assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_avx512_core(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)
        y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16


class Pytorch1_12(Pytorch1_11):
    def test_bf16_with_avx512_bf16(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)

        bf16_model._max_bf16_isa = MagicMock(return_value="AVX512_BF16")
        y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_amx(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = trainer.quantize(model, precision='bf16', use_ipex=True)

        bf16_model._max_bf16_isa = MagicMock(return_value="AMX")
        y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16


TORCH_VERSION_CLS = Pytorch1_12

if not check_avx512():
    print("IPEX Inference Model Without AVX512")
    TORCH_VERSION_CLS = CaseWithoutAVX512
elif TORCH_VERSION_LESS_1_10:
    print("ipex 1.9")
    TORCH_VERSION_CLS = Pytorch1_9
elif TORCH_VERSION_LESS_1_12:
    print("ipex 1.11")
    TORCH_VERSION_CLS = Pytorch1_11


class TestBF16(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
