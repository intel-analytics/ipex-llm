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
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_2_0
from bigdl.nano.utils.common import _avx2_checker


batch_size = 256
num_workers = 0
data_dir = "/tmp/data"


class VisionIPEX:

    def test_resnet18_ipex(self):
        resnet18 = vision.resnet18(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_resnet34_ipex(self):
        resnet34 = vision.resnet34(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet34, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_resnet50_ipex(self):
        resnet50 = vision.resnet50(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet50, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_mobilenet_v3_large_ipex(self):
        mobilenet = vision.mobilenet_v3_large(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_mobilenet_v3_small_ipex(self):
        mobilenet = vision.mobilenet_v3_small(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_mobilenet_v2_ipex(self):
        mobilenet = vision.mobilenet_v2(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            use_ipex=True)

    def test_shufflenet_ipex(self):
        shufflenet = vision.shufflenet_v2_x1_0(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            shufflenet, batch_size, num_workers, data_dir,
            use_ipex=True)


TORCH_CLS = VisionIPEX


class CaseWithoutAVX2:
    def test_placeholder(self):
        pass


if not TORCH_VERSION_LESS_2_0 and not _avx2_checker():
    print("Vision IPEX Without AVX2")
    # Intel® Extension for PyTorch* only works on machines with instruction sets equal or newer than AVX2
    TORCH_CLS = CaseWithoutAVX2


class TestVisionIPEX(TORCH_CLS, TestCase):
    pass


if __name__ == '__main__':
    pytest.main([__file__])
