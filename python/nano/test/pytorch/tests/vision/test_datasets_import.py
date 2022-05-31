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
from unittest import TestCase

class TestDatasetsImport(TestCase):
    def test_datasets_replace(self):
        from torchvision import datasets
        origin_set = set(datasets.__all__)
        del datasets
        from bigdl.nano.pytorch.vision import datasets
        new_set = set(dir(datasets))
        assert origin_set.issubset(new_set)

    def test_datasets_ImageFolder_version(self):
        from bigdl.nano.pytorch.vision import datasets
        assert datasets.__name__ in datasets.ImageFolder.__module__

if __name__ == '__main__':
    pytest.main([__file__])
