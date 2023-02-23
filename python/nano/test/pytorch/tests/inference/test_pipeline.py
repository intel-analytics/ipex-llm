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

import torch
from unittest import TestCase
from torchvision.models import resnet18
from bigdl.nano.pytorch import Pipeline


class TestPipeline(TestCase):
    def test_pipeline_create(self):
        def stage1():
            pass
        _pipeline = Pipeline([
            ("preprocess", stage1, {}),
            ("inference", stage1, {}),
        ])

    def test_pipeline_inference(self):
        def preprocess(_i):
            return torch.rand((1, 3, 224, 224))
        model = resnet18(num_classes=10)
        pipeline = Pipeline([
            ("preprocess", preprocess, {}),
            ("inference", model, {}),
        ])
        inputs = list(range(10))
        outputs = pipeline.run(inputs)
        assert len(outputs) == 10 and all(map(lambda o: o.shape == (1, 10), outputs))
