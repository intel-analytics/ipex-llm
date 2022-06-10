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

from bigdl.ppml import PPMLContext
from pyspark.sql import SparkSession

resource_path = os.path.join(os.path.dirname(__file__), "/resources")


class TestPPMLContext(unittest.TestCase):
    def setUp(self) -> None:
        self.args = {"kms_type": "SimpleKeyManagementService",
                     "simple_app_id": "465227134889",
                     "simple_app_key": "799072978028",
                     "primary_key_path": os.path.join(resource_path, "primaryKey"),
                     "data_key_path": os.path.join(resource_path, "dataKey")
                     }
        self.sc = PPMLContext(None, "testApp", self.args)

    def test_read_plain_file(self):
        print("test read plain text file")
        input_path = os.path.join(resource_path, "people.csv")
        print(input_path)
        df = self.sc.read("plain_text") \
            .option("header", "true") \
            .csv(input_path)
        self.assertEqual(df.count(), 100)

    def test_read_encrypt_file(self):
        print("test read encrypt text file")
        input_path = os.path.join(resource_path, "encrypt-people")
        print(input_path)
        df = self.sc.read("AES/CBC/PKCS5Padding") \
            .option("header", "true") \
            .csv(input_path)
        self.assertEqual(df.count(), 100)

    def test_write_plain_file(self):
        print("test write plain text file")
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        spark = SparkSession.builder.appName('testApp').getOrCreate()
        df = spark.createDataFrame(data).toDF("language", "user")
        self.sc.write(df, "plain_text") \
            .mode('overwrite') \
            .option("header", True) \
            .csv(os.path.join(resource_path, "out"))

    def test_write_encrypt_file(self):
        print("test write encrypt text file")
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        spark = SparkSession.builder.appName('testApp').getOrCreate()
        df = spark.createDataFrame(data).toDF("language", "user")
        self.sc.write(df, "AES/CBC/PKCS5Padding") \
            .mode('overwrite') \
            .option("header", True) \
            .csv(os.path.join(resource_path, "out"))


if __name__ == "__main__":
    unittest.main()
