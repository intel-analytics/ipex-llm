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

import platform
import pytest
import torch
from unittest import TestCase
from torchvision.models import resnet18
from bigdl.nano.pytorch import Pipeline


def empty_stage():
    pass


def preprocess(_i):
    return torch.rand((1, 3, 224, 224))


class TestPipeline(TestCase):
    def test_pipeline_create(self):
        _pipeline = Pipeline([
            ("preprocess", empty_stage, {}),
            ("inference", empty_stage, {}),
        ])

    def test_pipeline_inference(self):
        model = resnet18(num_classes=10)
        pipeline = Pipeline([
            ("preprocess", preprocess, {}),
            ("inference", model, {}),
        ])
        inputs = list(range(10))
        outputs = pipeline.run(inputs)
        assert len(outputs) == 10 and all(map(lambda o: o.shape == (1, 10), outputs))

    @pytest.mark.skipif(platform.system() == "Windows",
                        reason=("os.sched_getaffinity() is unavaiable on Windows, "
                                "and Windows doesn't support pickle local function"))
    def test_pipeline_core_control(self):
        def preprocess(_i):
            import os
            affinity = os.environ.get("KMP_AFFINITY", "")
            beg = affinity.find("[")
            end = affinity.find("]", beg) + 1
            return affinity[beg: end]
        def inference(i):
            import os
            affinity = os.environ.get("KMP_AFFINITY", "")
            beg = affinity.find("[")
            end = affinity.find("]", beg) + 1
            return (i, affinity[beg: end])
        pipeline = Pipeline([
            ("preprocess", preprocess, {"core_num": 1}),
            ("inference", inference, {"core_num": 1}),
        ])
        output = pipeline.run([None])[0]
        # The first stage's affinity should be '[0]', and the second stage's affinity should be '[1]'
        assert output == ('[0]', '[1]')

