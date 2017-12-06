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

from bigdl.nn.layer import *
from bigdl.nn.initialization_method import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import _py2java
from bigdl.nn.initialization_method import *
from bigdl.dataset import movielens
import numpy as np
import tempfile
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from bigdl.util.engine import compare_version
np.random.seed(1337)  # for reproducibility


class TestPickler():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        JavaCreator.set_creator_class("com.intel.analytics.bigdl.python.api.PythonBigDLValidator")
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        init_engine()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_activity_with_jtensor(self):
        back = callBigDlFunc("float", "testActivityWithTensor")
        back = back.toPy()
        assert isinstance(back.value, np.ndarray)

    def test_activity_with_table_of_tensor(self):
        back = callBigDlFunc("float", "testActivityWithTableOfTensor")
        back = back.toPy()
        assert isinstance(back.value, list)
        assert isinstance(back.value[0], np.ndarray)
        assert back.value[0][0] < back.value[1][0]
        assert back.value[1][0] < back.value[2][0]

    def test_activity_with_table_of_table(self):
        back = callBigDlFunc("float", "testActivityWithTableOfTable")
        back = back.toPy()
        assert isinstance(back.value, list)
        assert isinstance(back.value[0], list)
        assert isinstance(back.value[0][0], np.ndarray)

        back_again = callBigDlFunc("float", "testPyToJavaActivity", back)
        back_again = back_again.toPy()
        assert isinstance(back_again.value, list)
        assert isinstance(back_again.value[0], list)
        assert isinstance(back_again.value[0][0], np.ndarray)

    def test_activity_without_serialization(self):
        back = callBigDlFunc("float", "testActivityWithTableOfTable")
        assert isinstance(back, JavaObject)
        back_again = callBigDlFunc("float", "testPyToJavaActivity", back)
        assert isinstance(back_again, JavaObject)
        back_again = back_again.toPy()
        assert isinstance(back_again.value, list)
        assert isinstance(back_again.value[0], list)
        assert isinstance(back_again.value[0][0], np.ndarray)

    def test_activity_py_to_java(self):
        back = callBigDlFunc("float", "testPyToJavaActivity",
                             JActivity(np.random.random_sample([2, 3])))
        back = back.toPy()
        assert back.value.shape == (2, 3)

    def test_nested_activity_py_to_java(self):
        value = np.random.random_sample([2, 3])
        list_value = [value, value]
        nested1 = JActivity(list_value)
        nested2 = JActivity([list_value, list_value])
        back = callBigDlFunc("float", "testPyToJavaActivity",
                             nested1)
        back = back.toPy()
        assert back.value[0].shape == (2, 3)
        back = callBigDlFunc("float", "testPyToJavaActivity",
                             nested2)
        back = back.toPy()
        assert back.value[0][0].shape == (2, 3)

if __name__ == "__main__":
    pytest.main([__file__])
