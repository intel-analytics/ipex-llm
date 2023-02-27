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
from bigdl.chronos.aiops.config_generator import ConfigGenerator, triggerbyclock
import time
from .. import op_diff_set_all


class TestConfigGenerator(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @op_diff_set_all
    def test_triggerbyclock(self):

        class MyConfigGenerator(ConfigGenerator):
            def __init__(self, sweetpoint):
                self.sweetpoint = sweetpoint
                super().__init__()

            def genConfig(self):
                return self.sweetpoint

            @triggerbyclock(2)
            def update_sweetpoint(self):
                self.sweetpoint += 1

        mycg = MyConfigGenerator(5)
        time.sleep(4)
        assert mycg.genConfig() > 5
