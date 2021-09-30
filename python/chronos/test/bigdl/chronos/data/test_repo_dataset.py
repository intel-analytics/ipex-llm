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
import pandas as pd
import random

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.data.repo_dataset import get_public_dataset


class TestRepoDataset(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_init_dataset(self):
        name = random.sample([x for x in range(10)], 5)
        path = '~/.chronos/dataset'
        with pytest.raises(AssertionError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'nyc_taxi'
        path = random.sample([x for x in range(10)], 5)
        with pytest.raises(AssertionError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'chronos_dataset'
        path = '~/.chorons/dataset/'
        with pytest.raises(NameError):
            get_public_dataset(name, path=path, redownload=False)
