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
from torch.utils.data import TensorDataset, DataLoader
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from torch import nn
import operator
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
from bigdl.nano.utils.util import compare_version
import tempfile

batch_size = 256
num_workers = 0
data_dir = "/tmp/data"


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


class MultipleInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        return self.dense1(x1) + self.dense2(x2)


class MultipleInputWithKwargsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(10, 1)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, x1, x2, x3=10):
        return self.dense1(x1) + self.dense2(x2) + x3


class IPEXJITInference_gt_1_10:
    model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
    data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    data_sample = next(iter(data_loader))[0]

    def test_ipex_inference(self):
        model = InferenceOptimizer.trace(self.model, accelerator=None, use_ipex=True)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, self.model)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)

    def test_jit_inference(self):
        model = InferenceOptimizer.trace(self.model, accelerator="jit",
                                         use_ipex=False, input_sample=self.data_sample)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)

    def test_ipex_jit_inference(self):
        model = InferenceOptimizer.trace(self.model, accelerator="jit",
                                         use_ipex=True, input_sample=self.data_sample)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
    
    def test_ipex_jit_inference_additional_attrs(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        #  patch a attr
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        # test jit + ipex
        new_model = InferenceOptimizer.trace(model, accelerator="jit",
                                             use_ipex=True,
                                             input_sample=self.data_sample)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()

        # test jit + ipex + inplace
        new_model = InferenceOptimizer.trace(model, accelerator="jit",
                                             use_ipex=True,
                                             inplace=True,
                                             input_sample=self.data_sample)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()

        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        #  patch a attr
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        # test jit
        new_model = InferenceOptimizer.trace(model, accelerator="jit",
                                             input_sample=self.data_sample)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()

        # test ipex
        new_model = InferenceOptimizer.trace(model, use_ipex=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width

        # test ipex inplace
        new_model = InferenceOptimizer.trace(model, use_ipex=True, inplace=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width

        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        #  patch a attr
        model.channels = 3
        def hello():
            print("hello world!")
        # patch a function
        model.hello = hello

        # test channels_last
        new_model = InferenceOptimizer.trace(model, channels_last=True)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
        assert new_model.channels == 3
        new_model.hello()
        with pytest.raises(AttributeError):
            new_model.width

    def test_ipex_jit_inference_strict(self):
        model = InferenceOptimizer.trace(self.model, accelerator="jit",
                                         jit_strict=False, input_sample=self.data_sample)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
            assert new_model.jit_strict is False
    
    def test_ipex_quantization(self):
        model = ResNet18(10, pretrained=False, include_top=False, freeze=True)
        # test dataloader contains x+y
        data_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
        inc_model = InferenceOptimizer.quantize(model,
                                                precision='int8',
                                                accelerator=None,
                                                method='ipex',
                                                calib_data=data_loader)
        # test context manager
        with InferenceOptimizer.get_context(inc_model):
            inc_model(self.data_sample)

        # test save & load
        import operator
        if compare_version("neural_compressor", operator.ge, "2.0"):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                InferenceOptimizer.save(inc_model, tmp_dir_name)
                new_model = InferenceOptimizer.load(tmp_dir_name, model,
                                                    input_sample=next(iter(data_loader))[0])
            with InferenceOptimizer.get_context(new_model):
                new_model(self.data_sample)

        # test dataloader only contains x
        from torchvision.models import resnet18
        model = resnet18()
        x = torch.rand((10, 3, 256, 256))
        ds = TensorDataset(x)
        dataloader = DataLoader(ds, batch_size=2)
        model = InferenceOptimizer.quantize(model,
                                            precision='int8',
                                            accelerator=None,
                                            method='ipex',
                                            calib_data=dataloader)
        with InferenceOptimizer.get_context(model):
            model(x)
        
        # test single sample
        from torchvision.models import resnet18
        model = resnet18()
        x = torch.rand((10, 3, 256, 256))
        model = InferenceOptimizer.quantize(model,
                                            precision='int8',
                                            accelerator=None,
                                            method='ipex',
                                            calib_data=x)
        with InferenceOptimizer.get_context(model):
            model(x)
        
        # test multi input
        for model_class in [MultipleInputNet, MultipleInputWithKwargsNet]:
            net = model_class()
            x1 = torch.randn(32, 10)
            x2 = torch.randn(32, 10)
            y = torch.randn(32, 1)
            if isinstance(net, MultipleInputNet):
                dataloader = DataLoader(TensorDataset(x1, x2, y), batch_size=1)
            else:
                x3 = torch.randn(32, 1)
                dataloader = DataLoader(TensorDataset(x1, x2, x3, y), batch_size=1)

            model = InferenceOptimizer.quantize(net,
                                                precision='int8',
                                                accelerator=None,
                                                method='ipex',
                                                calib_data=dataloader)
            with InferenceOptimizer.get_context(model):
                if isinstance(net, MultipleInputNet):
                    model(x1, x2)
                else:
                    model(x1, x2, x3)

    def test_ipex_jit_inference_jit_method(self):
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
        accmodel = InferenceOptimizer.trace(model, accelerator='jit',
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
        accmodel = InferenceOptimizer.trace(model, accelerator='jit', 
                                      use_ipex=True,
                                      input_sample=input_sample, 
                                      jit_method='trace')
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output.shape[0] != expected_output_len

        # test with deafult jit_method
        accmodel = InferenceOptimizer.trace(model, accelerator='jit', 
                                      input_sample=input_sample)
        with InferenceOptimizer.get_context(accmodel):
            output = accmodel(input)
        assert output != output.shape[0] != expected_output_len

        # test with invalidInputError
        with pytest.raises(RuntimeError):
            InferenceOptimizer.trace(model, accelerator='jit', 
                               input_sample=input_sample,
                               jit_method='scriptttt')

    def test_ipex_jit_inference_weights_prepack(self):
        # test jit + ipex
        model = InferenceOptimizer.trace(self.model, accelerator="jit",
                                         use_ipex=True, input_sample=self.data_sample,
                                         weights_prepack=False)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
            assert new_model.weights_prepack is False

        # test ipex
        model = InferenceOptimizer.trace(self.model, accelerator=None,
                                         use_ipex=True, weights_prepack=False)
        with InferenceOptimizer.get_context(model):
            model(self.data_sample)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, self.model)
        with InferenceOptimizer.get_context(new_model):
            new_model(self.data_sample)
            assert new_model.weights_prepack is False

class IPEXJITInference_lt_1_10:
    def test_placeholder(self):
        pass


TORCH_VERSION_CLS = IPEXJITInference_gt_1_10
if TORCH_VERSION_LESS_1_10:
    TORCH_VERSION_CLS = IPEXJITInference_lt_1_10


class TestIPEXJITInference(TORCH_VERSION_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
