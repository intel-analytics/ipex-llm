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

import argparse

from bigdl.ppml.ppml_context import *
from bigdl.ppml.kms.utils.kms_argument_parser import KmsArgumentParser

"""
execute the following command to run this example on local

python simple_query_example.py \
--app_id your_simple_kms_app_id \
--api_key your_simple_kms_app_key \
--primary_key_material /your/primary/key/path/primaryKey \
--input_path /your/file/input/path \
--output_path /your/file/output/path \
--input_encrypt_mode AES/CBC/PKCS5Padding \
--output_encrypt_mode plain_text

"""

args = KmsArgumentParser().get_arg_dict()

sc = PPMLContext('pyspark-simple-query', args)

# create a DataFrame
data = [("Tom", "20", "Developer"), ("Jane", "21", "Developer"), ("Tony", "19", "Developer")]
df = sc.spark.createDataFrame(data).toDF("name", "age", "job")

# write DataFrame as an encrypted csv file
sc.write(df, args["input_encrypt_mode"]) \
    .mode('overwrite') \
    .option("header", True) \
    .csv(args["input_path"])

# get a DataFrame from an encrypted csv file
df = sc.read(args["input_encrypt_mode"]) \
    .option("header", "true") \
    .csv(args["input_path"])

df.select("name").count()

df.select(df["name"], df["age"] + 1).show()

developers = df.filter((df["job"] == "Developer") & df["age"]
                       .between(20, 40)).toDF("name", "age", "job").repartition(1)

sc.write(developers, args["output_encrypt_mode"]) \
    .mode('overwrite') \
    .option("header", True) \
    .csv(args["output_path"])
