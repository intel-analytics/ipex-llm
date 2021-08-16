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
from bigdl.util.common import *


class TestEngineEnv():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        pass

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        pass

    def test___prepare_bigdl_env(self):
        # BigDL will automatically execute 'prepare_env()' function which
        # includes '__prepare_bigdl_env()'. To test if there's no more duplicate
        #  adding jar path message, just do prepare_env()' again
        # to see if the log is correct and the environment variables should not vary.

        from bigdl.util.engine import prepare_env

        bigdl_jars_env_1 = os.environ.get("BIGDL_JARS", None)
        spark_class_path_1 = os.environ.get("SPARK_CLASSPATH", None)
        sys_path_1 = sys.path
        prepare_env()
        # there should be no duplicate messages about adding jar path to
        # the environment var "BIGDL_JARS"
        # environment variables should remain the same
        bigdl_jars_env_2 = os.environ.get("BIGDL_JARS", None)
        spark_class_path_2 = os.environ.get("SPARK_CLASSPATH", None)
        sys_path_2 = sys.path
        assert bigdl_jars_env_1 == bigdl_jars_env_2
        assert spark_class_path_1 == spark_class_path_2
        assert sys_path_1 == sys_path_2

if __name__ == '__main__':
    pytest.main()
