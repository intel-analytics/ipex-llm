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

from bigdl.ppml.ppml_context import *
from pyspark import SparkConf

ppml_args = {"kms_type": "SimpleKeyManagementService",
             "app_id": "123456654321",
             "api_key": "123456654321",
             "primary_key_material": "/opt/occlum_spark/data/key/simple_encrypted_primary_key",
             }

import sys
plain_csv_path = sys.argv[1]
encrypt_output_path = sys.argv[2]
conf = SparkConf()
conf.setMaster("local[4]")
sc = PPMLContext("MyApp", ppml_args, conf)
# import
from bigdl.ppml.ppml_context import *

# read a plain csv file and return a DataFrame
df1 = sc.read(CryptoMode.PLAIN_TEXT).option("header", "true").csv(plain_csv_path)
# write a DataFrame as an encrypted csv file
sc.write(df1, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').option("header", "true").csv(encrypt_output_path)
