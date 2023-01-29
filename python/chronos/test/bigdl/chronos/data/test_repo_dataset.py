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

from unittest import TestCase
from bigdl.chronos.data import get_public_dataset, gen_synthetic_data
from .. import op_torch, op_tf2


@op_torch
@op_tf2
class TestRepoDataset(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_init_dataset(self):
        name = random.sample([x for x in range(10)], 5)
        path = '~/.chronos/dataset'
        with pytest.raises(RuntimeError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'nyc_taxi'
        path = random.sample([x for x in range(10)], 5)
        with pytest.raises(RuntimeError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'chronos_dataset'
        path = '~/.chorons/dataset/'
        with pytest.raises(RuntimeError):
            get_public_dataset(name, path=path, redownload=False)

        name = 'tsinghua_electricity'
        path = '~/.chorons/dataset/'
        with pytest.raises(RuntimeError):
            get_public_dataset(name, path=path, redownload=False)

    def test_gen_synthetic_data(self):
        with pytest.raises(RuntimeError):
            gen_synthetic_data(len="10000")
        with pytest.raises(RuntimeError):
            gen_synthetic_data(sine_amplitude="10")
        with pytest.raises(RuntimeError):
            gen_synthetic_data(angular_freq="0.01")
        with pytest.raises(RuntimeError):
            gen_synthetic_data(noise_amplitude="0.01")
        with pytest.raises(RuntimeError):
            gen_synthetic_data(noise_scale="1.0")
        with pytest.raises(RuntimeError):
            gen_synthetic_data(seed=0.01)
        with pytest.raises(RuntimeError):
            gen_synthetic_data(time_freq=1)

        tsdata = gen_synthetic_data()
        assert tsdata._id_list == ['0']
        assert tsdata.target_col == ["target"]
        assert tsdata.dt_col == "datetime"

        tsdata = gen_synthetic_data(len=5000)
        assert len(tsdata.to_pandas()) == 5000

        tsdata_train, tsdata_val, tsdata_test = gen_synthetic_data(with_split=True,
                                                                   val_ratio=0.1, test_ratio=0.1)
        assert len(tsdata_train.to_pandas()) == 10000*0.8
        assert len(tsdata_val.to_pandas()) == 10000*0.1
        assert len(tsdata_test.to_pandas()) == 10000*0.1
