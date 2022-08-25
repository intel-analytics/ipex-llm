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

import unittest
from uuid import uuid4


from bigdl.ppml.fl.psi.utils import *

from bigdl.ppml.fl.utils import FLTest

class TestHashing(FLTest):           
    def test_hashing(self):        
        ids = ['1', '2', '4', '5']
        salt = str(uuid4())
        hex_string = to_hex_string(ids, salt)
        hex_string


if __name__ == '__main__':
    unittest.main()
