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
from test.pytorch.utils._train_torch_lightning import create_data_loader, data_transform
from torch import nn

from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10
import tempfile

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../data")


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


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
