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
from random import randrange
from bigdl.nano.automl.hpo.decorator import *
import bigdl.nano.automl.hpo.space as space

class TestHPODecorator(TestCase):

    def test_func(self):
        @func()
        def dummy_func(arg1=None, arg2=None):
            return arg1, arg2

        range_arg1 = (randrange(10,20), randrange(100,200))
        range_arg2 = (randrange(10,20), randrange(100,200))
        autofunc = dummy_func(arg1=space.Int(range_arg1[0], range_arg1[1]),
                   arg2=space.Int(range_arg2[0], range_arg2[1]))
        assert(autofunc)

        arg1, arg2 = autofunc.sample()
        assert(arg1 >= range_arg1[0] and arg1 <= range_arg1[1])
        assert(arg2 >= range_arg2[0] and arg2 <= range_arg2[1])

    def test_obj(self):
        #TODO
        pass

    def test_tfmodel(self):
        #TODO
        pass

    def test_plmodel(self):
        #TODO
        pass

if __name__ == '__main__':
    pytest.main([__file__])