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
import os
import random
import shutil

from bigdl.ppml.ppml_context import *

resource_path = os.path.join(os.path.dirname(__file__), "resources")


class TestPPMLContext(unittest.TestCase):
    # generate app_id and app_key
    app_id = ''.join([str(random.randint(0, 9)) for i in range(12)])
    app_key = ''.join([str(random.randint(0, 9)) for j in range(12)])

    df = None
    data_content = None
    csv_content = None
    sc = None

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(resource_path):
            os.mkdir(resource_path)

        args = {"kms_type": "SimpleKeyManagementService",
                "simple_app_id": "429738913832",
                "simple_app_key": "273066367938",
                "primary_key_path": "/home/zehuan/test/keys/primaryKey",
                "data_key_path": "/home/zehuan/test/keys/dataKey"
                }

        cls.sc = PPMLContext("testApp", args)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(resource_path):
            shutil.rmtree(resource_path)

    def test_one(self):
        print("**************** TEST DEBUG ******************")
        print(self.sc.spark.sparkContext.getConf().getAll())



if __name__ == "__main__":
    unittest.main()
