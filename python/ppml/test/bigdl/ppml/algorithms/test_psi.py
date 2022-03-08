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


from bigdl.ppml.algorithms.psi import PSI
from bigdl.ppml.fl_server import FLServer
from bigdl.ppml import *


class TestPSI(unittest.TestCase):
    def test_psi_create(self):
        psi = PSI()
        
    def test_psi_get_salt(self):
        fl_server = FLServer()
        fl_server.build()
        fl_server.start()
        init_fl_context()
        psi = PSI()
        salt = psi.get_salt()
        fl_server.stop()
        assert (isinstance(salt, str))

    def test_psi_pipeline(self):
        fl_server = FLServer()
        fl_server.build()
        fl_server.start()
        init_fl_context()
        psi = PSI()
        salt = psi.get_salt()
        key = ["k1", "k2"]
        psi.upload_set(key, salt)
        intersection = psi.download_intersection()
        fl_server.stop()
        assert (isinstance(intersection, list))
        self.assertEqual(len(intersection), 2)

if __name__ == '__main__':
    unittest.main()
