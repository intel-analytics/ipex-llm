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


batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../../data")


class TestVisionIPEX(TestCase):
    from bigdl.nano.pytorch.accelerators.ipex_accelerator import IPEXAccelerator
    ipex_accelerator = IPEXAccelerator()
    def test_resnet18_ipex(self):
        resnet18 = vision.resnet18(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_resnet34_ipex(self):
        resnet34 = vision.resnet34(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet34, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_resnet50_ipex(self):
        resnet50 = vision.resnet50(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet50, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_mobilenet_v3_large_ipex(self):
        mobilenet = vision.mobilenet_v3_large(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_mobilenet_v3_small_ipex(self):
        mobilenet = vision.mobilenet_v3_small(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_mobilenet_v2_ipex(self):
        mobilenet = vision.mobilenet_v2(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)

    def test_shufflenet_ipex(self):
        shufflenet = vision.shufflenet_v2_x1_0(
            pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(
            shufflenet, batch_size, num_workers, data_dir,
            accelerator=self.ipex_accelerator)


if __name__ == '__main__':
    pytest.main([__file__])
