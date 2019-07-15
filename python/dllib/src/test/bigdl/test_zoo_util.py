#
# Copyright 2018 Analytics Zoo Authors.
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

from bigdl.util.common import get_node_and_core_number
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.common import set_core_number


class TestUtil(ZooTestCase):

    def test_set_core_num(self):
        _, core_num = get_node_and_core_number()

        set_core_number(core_num + 1)

        _, new_core_num = get_node_and_core_number()

        assert new_core_num == core_num + 1, \
            "set_core_num failed, set the core" \
            " number to be {} but got {}".format(core_num + 1, new_core_num)

        set_core_number(core_num)


if __name__ == "__main__":
    pytest.main([__file__])
