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

from bigdl.nano.automl.hpo.backend import create_hpo_backend
from bigdl.nano.automl.hpo.backend import SamplerType,PrunerType

class TestHPOBackend(TestCase):

    def test_create_sampler(self):
        backend = create_hpo_backend()
        sampler = backend.create_sampler(SamplerType.TPE,{})
        assert(sampler)

    def test_create_pruner(self):
        backend = create_hpo_backend()
        pruner = backend.create_pruner(PrunerType.HyperBand,{})
        assert(pruner)

    def test_instantiate(self):
        #TODO
        pass



if __name__ == '__main__':
    pytest.main([__file__])