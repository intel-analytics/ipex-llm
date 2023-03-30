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
from torch import nn
from bigdl.nano.pytorch import InferenceOptimizer
from torchvision.models.resnet import resnet18
from unittest.mock import PropertyMock, patch
from bigdl.nano.utils.common import _avx512_checker
import tempfile
from typing import List


class CaseWithoutAVX512:
    def test_unsupported_HW_or_OS(self):
        model = resnet18(num_classes=10)

        with pytest.raises(RuntimeError,
                           match="Applying IPEX BF16 optimization needs the cpu support avx512."):
            bf16_model = InferenceOptimizer.quantize(model, precision='bf16', use_ipex=True)


class DummyMultiInputModel(nn.Module):
    """
    A simple model for test various inputs of channels last format
    """
    def __init__(self):
        super(DummyMultiInputModel, self).__init__()

    def forward(self, x1, x2, x3: List[float]):
        return x1, x2, x3


class DummyModelWith3d(nn.Module):
    """
    A simple model for test various inputs of channels last format
    """
    def __init__(self):
        super(DummyModelWith3d, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 33, 3, stride=2)

    def forward(self, x1, x2:int):
        return self.conv3d_1(x1), x2


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

        # test jit + ipex + bf16
        new_model = InferenceOptimizer.quantize(model,
                                                precision='bf16',
                                                accelerator="jit",
                                                use_ipex=True,
                                                input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        # save & load with original model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(
            AttributeError,
            match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"
        ):
            load_model.strange_call()

        # test jit + bf16
        new_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                accelerator="jit",
                                                input_sample=x)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        # save & load with original model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(
            AttributeError,
            match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"
        ):
            load_model.strange_call()

        # test ipex + bf16
        new_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                use_ipex=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width
        # save & load with original model
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(new_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model=model)
        assert load_model.channels == 3
        load_model.hello()
        with pytest.raises(
            AttributeError,
            match="'PytorchIPEXJITBF16Model' object has no attribute 'strange_call'"
        ):
            load_model.strange_call()

    def test_bf16_ipex_jit_method(self):

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.arange(len(x))

        model = Net()

        input_sample = torch.rand(1,3,1,1)
        input = torch.rand(5,3,1,1)
        expected_output_len = 5

        # test with jit.script (with ipex)
        accmodel = InferenceOptimizer.quantize(model, precision='bf16',
                                               accelerator='jit', 
                                               use_ipex=True,
                                               input_sample=input_sample, 
                                               jit_method='script')
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] == expected_output_len

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(accmodel, tmp_dir_name)
            loaded_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(loaded_model):
            output = loaded_model(input)
        assert output.shape[0] == expected_output_len
        assert loaded_model.jit_method == 'script'

        # test with jit.trace (with ipex)
        accmodel = InferenceOptimizer.quantize(model, precision='bf16',
                                               accelerator='jit', 
                                               use_ipex=True,
                                               input_sample=input_sample, 
                                               jit_method='trace')
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] != expected_output_len

        # test with deafult jit_method
        accmodel = InferenceOptimizer.quantize(model, precision='bf16',
                                               accelerator='jit',
                                               input_sample=input_sample)
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] != expected_output_len

        # test with invalidInputError
        with pytest.raises(RuntimeError):
            InferenceOptimizer.quantize(model, precision='bf16',
                                        accelerator='jit',
                                        input_sample=input_sample,
                                        jit_method='scriptttt')

    def test_ipex_jit_inference_weights_prepack(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        # test jit + ipex
        model = InferenceOptimizer.quantize(model, precision='bf16',
                                            accelerator="jit",
                                            use_ipex=True,
                                            input_sample=x,
                                            weights_prepack=False)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.weights_prepack is False

    def test_bf16_ipex_channels_last_various_input_sample(self):
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8) # 3-dim input test
        x2 = torch.rand(1, 3, 8, 8) # 4-dim input test
        x3 = [1, 2, 3, 4] # input without .to() method
        bf16_ipex_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                                    channels_last=True, use_ipex=True)
        with InferenceOptimizer.get_context(bf16_ipex_channels_last_model):
            bf16_ipex_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_ipex_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_bf16_jit_channels_last_various_input_sample(self):
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8) # 3-dim input test
        x2 = torch.rand(1, 3, 8, 8) # 4-dim input test
        x3 = [1, 2, 3, 4] # input without .to() method
        bf16_jit_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                                   channels_last=True, accelerator="jit")
        with InferenceOptimizer.get_context(bf16_jit_channels_last_model):
            bf16_jit_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_bf16_ipex_jit_channels_last_various_input_sample(self):
        model = DummyMultiInputModel()
        x1 = torch.rand(1, 8, 8) # 3-dim input test
        x2 = torch.rand(1, 3, 8, 8) # 4-dim input test
        x3 = [1, 2, 3, 4] # input without .to() method
        bf16_ipex_jit_channels_last_model = InferenceOptimizer.quantize(model, precision='bf16',
                                                                        channels_last=True,
                                                                        use_ipex=True, accelerator="jit")
        with InferenceOptimizer.get_context(bf16_ipex_jit_channels_last_model):
            bf16_ipex_jit_channels_last_model(x1, x2, x3)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_ipex_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            load_model(x1, x2, x3)

    def test_ipex_jit_inference_onednn(self):
        model = resnet18(num_classes=10)
        x = torch.rand((10, 3, 256, 256))
        # test jit + ipex
        model = InferenceOptimizer.quantize(model, precision='bf16',
                                            accelerator="jit",
                                            use_ipex=True,
                                            input_sample=x,
                                            enable_onednn=True)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.enable_onednn is True
        
        model = InferenceOptimizer.quantize(model, precision='bf16',
                                            accelerator="jit",
                                            use_ipex=True,
                                            input_sample=x,
                                            enable_onednn=False)
        with InferenceOptimizer.get_context(model):
            model(x)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(x)
            assert new_model.enable_onednn is False

    def test_ipex_jit_channels_last_3d_inference(self):
        model = DummyModelWith3d()
        x1 = torch.rand(32, 3, 3, 224, 224) # 5-dim input test
        x2 = 3
        ipex_jit_channels_last_model = InferenceOptimizer.quantize(model,
                                                                   accelerator="jit", 
                                                                   use_ipex=True,
                                                                   precision='bf16',
                                                                   input_sample=(x1, x2),
                                                                   enable_onednn=True,
                                                                   channels_last=True)
        with InferenceOptimizer.get_context(ipex_jit_channels_last_model):
            ipex_jit_channels_last_model(x1, x2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(ipex_jit_channels_last_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name)
            with InferenceOptimizer.get_context(load_model):
                load_model(x1, x2)


TORCH_VERSION_CLS = Pytorch1_11


if not _avx512_checker():
    print("IPEX Inference Model Without AVX512")
    TORCH_VERSION_CLS = CaseWithoutAVX512


class TestIPEXBF16(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
