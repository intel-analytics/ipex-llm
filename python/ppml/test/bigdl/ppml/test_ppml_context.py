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
import csv

from bigdl.ppml.ppml_context import PPMLContext
from pyspark.sql import SparkSession

resource_path = os.path.join(os.path.dirname(__file__), "resources")


class TestPPMLContext(unittest.TestCase):
    def setUp(self) -> None:
        self.args = {"kms_type": "SimpleKeyManagementService",
                     "simple_app_id": "465227134889",
                     "simple_app_key": "799072978028",
                     "primary_key_path": os.path.join(resource_path, "primaryKey"),
                     "data_key_path": os.path.join(resource_path, "dataKey")
                     }
        self.sc = PPMLContext("testApp", self.args)

    def test_read_plain_file(self):
        # create a tmp csv file
        with open(os.path.join(resource_path, "people.csv"), "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["name", "age", "job"])
            csv_writer.writerow(["jack", "18", "Developer"])
            csv_writer.writerow(["alex", "20", "Researcher"])
            csv_writer.writerow(["xuoui", "25", "Developer"])
            csv_writer.writerow(["hlsgu", "29", "Researcher"])
            csv_writer.writerow(["xvehlbm", "45", "Developer"])
            csv_writer.writerow(["ehhxoni", "23", "Developer"])
            csv_writer.writerow(["capom", "60", "Developer"])
            csv_writer.writerow(["pjt", "24", "Developer"])
            f.close()

        input_path = os.path.join(resource_path, "people.csv")
        df = self.sc.read("plain_text") \
            .option("header", "true") \
            .csv(input_path)
        self.assertEqual(df.count(), 8)

        if os.path.isfile(input_path):
            os.remove(input_path)

    def test_read_encrypt_file(self):
        input_path = os.path.join(resource_path, "encrypt-people")
        df = self.sc.read("AES/CBC/PKCS5Padding") \
            .option("header", "true") \
            .csv(input_path)
        self.assertEqual(df.count(), 100)

    def test_write_plain_file(self):
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        spark = SparkSession.builder.appName('testApp').getOrCreate()
        df = spark.createDataFrame(data).toDF("language", "user")
        self.sc.write(df, "plain_text") \
            .mode('overwrite') \
            .option("header", True) \
            .csv(os.path.join(resource_path, "out/plain"))

    def test_write_encrypt_file(self):
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        spark = SparkSession.builder.appName('testApp').getOrCreate()
        df = spark.createDataFrame(data).toDF("language", "user")
        self.sc.write(df, "AES/CBC/PKCS5Padding") \
            .mode('overwrite') \
            .option("header", True) \
            .csv(os.path.join(resource_path, "out/encrypt"))


if __name__ == "__main__":
    unittest.main()
