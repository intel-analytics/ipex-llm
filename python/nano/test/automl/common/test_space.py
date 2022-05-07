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

import bigdl.nano.automl.hpo.space as space

class TestHPOSpace(TestCase):

    def test_categorical(self):
        choices = ['a','b','c','d']
        param = space.Categorical(*choices)
        assert(param.cs)
        assert(param.rand in choices)
        assert(param.default in choices)

    def test_real(self):
        min = 0.001
        max = 0.1
        param = space.Real(min, max)
        assert(param.rand >= min and param.rand <= max )
        assert(param.default >= min and param.default <= max)

    def test_int(self):
        min = 1
        max = 1e6
        param = space.Real(min, max)
        assert(param.rand >= min and param.rand <= max )
        assert(param.default >= min and param.default <= max)

    def test_list(self):
        #TODO
        pass

    def test_dict(self):
        #TODO
        pass


if __name__ == '__main__':
    pytest.main([__file__])