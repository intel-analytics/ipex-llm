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
    sc = None

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(resource_path):
            os.mkdir(resource_path)

        # set key path
        primary_key_path = os.path.join(resource_path, "primaryKey")
        data_key_path = os.path.join(resource_path, "dataKey")

        # init a SparkContext
        conf = {"spark.app.name": "PPML TEST",
                "spark.hadoop.io.compression.codecs": "com.intel.analytics.bigdl.ppml.crypto.CryptoCodec",
                "spark.bigdl.kms.type": "SimpleKeyManagementService",
                "spark.bigdl.kms.simple.id": cls.app_id,
                "spark.bigdl.kms.simple.key": cls.app_key,
                "spark.bigdl.kms.key.primary": primary_key_path,
                "spark.bigdl.kms.key.data": data_key_path
                }
        init_spark_on_local(conf=conf)

        # generate primaryKey and dataKey
        init_keys(cls.app_id, cls.app_key,
                  primary_key_path, data_key_path)

        args = {"kms_type": "SimpleKeyManagementService",
                "simple_app_id": cls.app_id,
                "simple_app_key": cls.app_key,
                "primary_key_path": primary_key_path,
                "data_key_path": data_key_path
                }

        # init a PPMLContext
        cls.sc = PPMLContext("testApp", args)

        # generate a DataFrame for test
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        cls.df = cls.sc.spark.createDataFrame(data).toDF("language", "user")
        cls.df = cls.df.repartition(1)
        cls.data_content = '\n'.join([str(v['language']) + "," + str(v['user'])
                                      for v in cls.df.orderBy('language').collect()])

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(resource_path):
            shutil.rmtree(resource_path)

    def test_write_and_read_plain_csv(self):
        path = os.path.join(resource_path, "csv/plain")
        # write as plain csv file
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT) \
            .mode('overwrite') \
            .option("header", True) \
            .csv(path)

        # read from a plain csv file
        df = self.sc.read(CryptoMode.PLAIN_TEXT) \
            .option("header", "true") \
            .csv(path)

        csv_content = '\n'.join([str(v['language']) + "," + str(v['user'])
                                 for v in df.orderBy('language').collect()])

        self.assertEqual(csv_content, self.data_content)

    def test_write_and_read_encrypted_csv(self):
        path = os.path.join(resource_path, "csv/encrypted")
        # write as encrypted csv file
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING) \
            .mode('overwrite') \
            .option("header", True) \
            .csv(path)

        # read from an encrypted csv file
        df = self.sc.read(CryptoMode.AES_CBC_PKCS5PADDING) \
            .option("header", "true") \
            .csv(path)

        csv_content = '\n'.join([str(v['language']) + "," + str(v['user'])
                                 for v in df.orderBy('language').collect()])

        self.assertEqual(csv_content, self.data_content)

    def test_write_and_read_plain_parquet(self):
        parquet_path = os.path.join(resource_path, "parquet/plain-parquet")
        # write as a parquet
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT) \
            .mode('overwrite') \
            .parquet(parquet_path)

        # read from a parquet
        df_from_parquet = self.sc.read(CryptoMode.PLAIN_TEXT) \
            .parquet(parquet_path)

        content = '\n'.join([str(v['language']) + "," + str(v['user'])
                             for v in df_from_parquet.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_write_and_read_encrypted_parquet(self):
        parquet_path = os.path.join(resource_path, "parquet/en-parquet")
        # write as a parquet
        self.sc.write(self.df, CryptoMode.AES_GCM_CTR_V1) \
            .mode('overwrite') \
            .parquet(parquet_path)

        # read from a parquet
        df_from_parquet = self.sc.read(CryptoMode.AES_GCM_CTR_V1) \
            .parquet(parquet_path)

        content = '\n'.join([str(v['language']) + "," + str(v['user'])
                             for v in df_from_parquet.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_plain_text_file(self):
        path = os.path.join(resource_path, "csv/plain")
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT) \
            .mode('overwrite') \
            .option("header", True) \
            .csv(path)

        rdd = self.sc.textfile(path)
        rdd_content = '\n'.join([line for line in rdd.collect()])

        self.assertEqual(rdd_content, "language,user\n" + self.data_content)

    def test_encrypted_text_file(self):
        path = os.path.join(resource_path, "csv/encrypted")
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING) \
            .mode('overwrite') \
            .option("header", True) \
            .csv(path)

        rdd = self.sc.textfile(path=path, crypto_mode=CryptoMode.AES_CBC_PKCS5PADDING)
        rdd_content = '\n'.join([line for line in rdd.collect()])
        self.assertEqual(rdd_content, "language,user\n" + self.data_content)

    def test_write_and_read_plain_json(self):
        json_path = os.path.join(resource_path, "json/plain-json")
        # write as a json
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT) \
            .mode('overwrite') \
            .json(json_path)

        # read from a json
        df_from_json = self.sc.read(CryptoMode.PLAIN_TEXT) \
            .json(json_path)

        content = '\n'.join([str(v['language']) + "," + str(v['user'])
                             for v in df_from_json.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_write_and_read_encrypted_json(self):
        json_path = os.path.join(resource_path, "json/en-json")
        # write as a json
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING) \
            .mode('overwrite') \
            .json(json_path)

        # read from a json
        df_from_json = self.sc.read(CryptoMode.AES_CBC_PKCS5PADDING) \
            .json(json_path)

        content = '\n'.join([str(v['language']) + "," + str(v['user'])
                             for v in df_from_json.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)


if __name__ == "__main__":
    unittest.main()
