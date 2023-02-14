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

from unittest import TestCase
import pytest

from bigdl.orca import init_orca_context, stop_orca_context

class TestImport(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_orca_context(cores=4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

    def test_data_import(self):
        from bigdl.orca.data import file, transformer

    def test_tf2_import(self):
        from bigdl.orca.learn.tf2 import Estimator, pyspark_estimator, spark_runner

    def test_pytorch_import(self):
        from bigdl.orca.learn.pytorch import Estimator, pytorch_metrics
        from bigdl.orca.learn.pytorch import pytorch_pyspark_estimator, torch_runner

if __name__ == "__main__":
    pytest.main([__file__])
