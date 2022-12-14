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
from unittest.mock import PropertyMock, patch
from bigdl.nano.common import check_avx512
import tempfile


class CaseWithoutAVX512:
    def test_unsupported_HW_or_OS(self):
        model = resnet18(num_classes=10)

        with pytest.raises(RuntimeError,
                           match="Applying IPEX BF16 optimization needs the cpu support avx512."):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)


class Pytorch1_11:
    @patch('bigdl.nano.deps.ipex.ipex_inference_bf16_model.PytorchIPEXJITBF16Model._check_cpu_isa', new_callable=PropertyMock)
    def test_unsupported_HW_or_OS(self, mocked_check_cpu_isa):
        mocked_check_cpu_isa.return_value = False
        model = resnet18(num_classes=10)

        with pytest.raises(RuntimeError,
                           match="Applying IPEX BF16 optimization needs the cpu support avx512."):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)
    
    def test_bf16_inference_with_jit(self):
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                 accelerator="jit",
                                                 input_sample=x)
        with InferenceOptimizer.get_context(bf16_model):
            for i in range(10):
                y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            for i in range(10):
                y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_with_avx512_core(self):
        model = resnet18(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_ipex_save_load(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                 use_ipex=True)
        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)

        with InferenceOptimizer.get_context(load_model):
            y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_jit_save_load(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        bf16_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                 use_ipex=True, accelerator="jit",
                                                 input_sample=x)

        with InferenceOptimizer.get_context(bf16_model):
            y_hat = bf16_model(x)

        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16
        
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(load_model):
            y_hat_ = load_model(x)
        assert y_hat_.shape == (10, 10) and y_hat_.dtype == torch.bfloat16
        assert y_hat.equal(y_hat_)

    def test_bf16_ipex_jit_additional_attrs(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        #  patch a attr
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello
        new_model = InferenceOptimizer.trace(model, precision='bf16',
                                             accelerator="jit", use_ipex=True,
                                             input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()

        new_model = InferenceOptimizer.trace(model, precision='bf16',
                                             accelerator="jit",
                                             input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()

        new_model = InferenceOptimizer.trace(model, precision='bf16',
                                             use_ipex=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width


TORCH_VERSION_CLS = Pytorch1_11


if not check_avx512():
    print("IPEX Inference Model Without AVX512")
    TORCH_VERSION_CLS = CaseWithoutAVX512


class TestIPEXBF16(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
