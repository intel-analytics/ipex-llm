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
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.resnet import resnet18
from unittest.mock import MagicMock, PropertyMock, patch
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_12
import tempfile


class Pytorch1_11:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """
    def test_bf16_pytorch_1_11(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        with pytest.raises(
            RuntimeError,
            match="Require torch>=1.12 to obtain bfloat16 acceleration."
        ):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16')


class Pytorch1_12:
    """
    Pytorch version >= 1.10 and <1.12, bfloat16 precision is supported.
    But there is no optimization on bfloat16.
    """
    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa", new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_has_bf16_isa):
        mocked_has_bf16_isa.return_value = False
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)

    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._max_bf16_isa", return_value=None)
    @patch("bigdl.nano.pytorch.amp.bfloat16.BF16Model._has_bf16_isa", new_callable=PropertyMock)
    @pytest.mark.skip(reason="Disable dnnl log check if torch==1.12")
    def test_not_executed_on_bf16(self, mocked_has_bf16_isa, mocked_max_bf16_isa):
        """
        Pytorch version is correct and bf16 instructions are detected.
        But somehow in the run, there is no bf16 instructions used.
        """
        mocked_has_bf16_isa.return_value = True
        mocked_max_bf16_isa.return_value = None

        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            bf16_model(x)

    @patch.dict('os.environ', {"ALLOW_NON_BF16_ISA": "1"})
    def test_bf16_common(self):
        """
        Debug mode. Allow run bf16 forward without bf16 instruction support.
        """
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        # Debug mode to test functionality, make sure forward is called sucessfully
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_amx_bf16(self):
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AMX")
            with InferenceOptimizer.get_context(bf16_model):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_avx512_bf16(self):
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with patch.object(type(bf16_model), "_has_bf16_isa", PropertyMock(return_value=True)):
            bf16_model._max_bf16_isa = MagicMock(return_value="AVX512")
            with InferenceOptimizer.get_context(bf16_model):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_save_and_load(self):
        model = resnet18(num_classes=10)

        # test bf16
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)

        with InferenceOptimizer.get_context(load_model):
            y_hat2 = load_model(x)
        assert y_hat2.shape == (10, 10) and y_hat2.dtype == torch.bfloat16
        assert y_hat1.equal(y_hat2)

        # test bf16 + channels_last
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                 channels_last=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)

        with InferenceOptimizer.get_context(load_model):
            y_hat2 = load_model(x)
        assert y_hat2.shape == (10, 10) and y_hat2.dtype == torch.bfloat16
        assert y_hat1.equal(y_hat2)

    def test_bf16_additional_attrs(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        # patch a attribute
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        # test bf16
        bf16_model = InferenceOptimizer.quantize(model,
                                                 precision='bf16')
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        assert bf16_model.channels == 3
        bf16_model.hello()

        # test bf16 + channels_last
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                 channels_last=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat1 = bf16_model(x)
        assert y_hat1.shape == (10, 10) and y_hat1.dtype == torch.bfloat16
        assert bf16_model.channels == 3
        bf16_model.hello()
        with pytest.raises(AttributeError):
            bf16_model.width


TORCH_VERSION_CLS = Pytorch1_12
if TORCH_VERSION_LESS_1_12:
    print("1.11")
    TORCH_VERSION_CLS = Pytorch1_11


class TestBF16(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
