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


from bigdl.ppml.fl.psi.psi_client import PSI
from bigdl.ppml.fl.nn.fl_server import FLServer
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.utils import FLTest

class TestPSI(FLTest):
    def setUp(self) -> None:
        self.fl_server = FLServer(1)
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()

    def tearDown(self) -> None:
        self.fl_server.stop()
   
        
    def test_psi_get_salt(self):        
        init_fl_context(1, self.target)
        psi = PSI()
        salt = psi.get_salt()
        assert (isinstance(salt, str))

    def test_psi_pipeline(self):
        init_fl_context(1, self.target)
        psi = PSI()
        salt = psi.get_salt()
        key = ["k1", "k2"]
        psi.upload_set(key, salt)
        intersection = psi.download_intersection()
        assert (isinstance(intersection, list))
        self.assertEqual(key, intersection)

if __name__ == '__main__':
    unittest.main()
