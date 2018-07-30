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

from bigdl.util.engine import *
import pytest
import os
import glob
import pyspark


class TestEngineEnv():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.spark_home = '/home/megaspoon/Documents/spark-2.0.1-bin-hadoop2.7'

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        pass

    def test___prepare_bigdl_env(self):
        with pytest.raises(KeyError):
            __prepare_bigdl_env()
    def test___sys_path_insert(self):
        py4j = glob.glob(os.path.join(self.spark_home, 'python/lib', 'py4j-*.zip'))[0]
        pyspark = glob.glob(os.path.join(self.spark_home, 'python/lib', 'pyspark*.zip'))[0]
        with pytest.raises(RuntimeError):
            __sys_path_insert(py4j)
        with pytest.raises(RuntimeError):
            __sys_path_insert(pyspark)

if __name__ == '__main__':
    pytest.main()
