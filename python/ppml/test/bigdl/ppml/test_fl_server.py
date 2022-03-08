#
<<<<<<< HEAD
# Copyright 2021 The BigDL Authors.
=======
# Copyright 2016 The BigDL Authors.
>>>>>>> e7467572f8032567954e0976268e6356d814accd
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
from bigdl.ppml import FLServer



class TestFLServer(unittest.TestCase):
    def test_fl_server_default_config(self):
        fl_server = FLServer()
        fl_server.build()
        fl_server.start()

    def test_fl_server_custom_config(self):
        pass


if __name__ == '__main__':
    unittest.main()
