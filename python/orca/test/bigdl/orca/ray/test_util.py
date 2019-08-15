#
# Copyright 2018 Analytics Zoo Authors.
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

import numpy as np
import pytest

import zoo.ray.util.utils as rutils

np.random.seed(1337)  # for reproducibility


class TestUtil(TestCase):

    # def test_split(self):
    #     vector = np.ones([10])
    #     result = rutils.split(vector, 4)
    #     assert len(result) == 4
    #     assert len(result[0]) == 3
    #     assert len(result[1]) == 3
    #     assert len(result[2]) == 2
    #     assert len(result[3]) == 2

    def test_resource_to_bytes(self):
        assert 10 == rutils.resourceToBytes("10b")
        assert 10000 == rutils.resourceToBytes("10k")
        assert 10000000 == rutils.resourceToBytes("10m")
        assert 10000000000 == rutils.resourceToBytes("10g")


if __name__ == "__main__":
    pytest.main([__file__])
