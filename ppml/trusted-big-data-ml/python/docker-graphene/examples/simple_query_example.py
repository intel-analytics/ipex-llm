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

"""
execute the following command to run this example on local

python simple_query_example.py \
--simple_app_id your_app_id \
--simple_app_key your_app_key \
--primary_key_path /your/primary/key/path/primaryKey \
--data_key_path /your/data/key/path/dataKey \
--input_path /your/file/input/path \
--output_path /your/file/output/path \
--input_encrypt_mode AES/CBC/PKCS5Padding \
--output_encrypt_mode plain_text

"""

parser = argparse.ArgumentParser()
parser.add_argument("--simple_app_id", type=str, help="simple app id")
parser.add_argument("--simple_app_key", type=str, help="simple app key")
parser.add_argument("--kms_server_ip", type=str, help="ehsm kms server ip")
parser.add_argument("--kms_server_port", type=str, help="ehsm kms server port")
parser.add_argument("--ehsm_app_id", type=str, help="ehsm app id")
parser.add_argument("--ehsm_app_key", type=str, help="ehsm app key")
parser.add_argument("--azure_vault", type=str, help="Azure Key Vault name")
parser.add_argument("--azure_client_id", type=str, default="", help="Azure client id")
parser.add_argument("--primary_key_path", type=str, required=True, help="primary key path")
parser.add_argument("--data_key_path", type=str, required=True, help="data key path")
parser.add_argument("--input_encrypt_mode", type=str, required=True, help="input encrypt mode")
parser.add_argument("--output_encrypt_mode", type=str, required=True, help="output encrypt mode")
parser.add_argument("--input_path", type=str, required=True, help="input path")
parser.add_argument("--output_path", type=str, required=True, help="output path")
parser.add_argument("--kms_type", type=str, required=True, default="SimpleKeyManagementService",
                    help="SimpleKeyManagementService, EHSMKeyManagementService or AzureKeyManagementService")
args = parser.parse_args()
arg_dict = vars(args)

sc = PPMLContext('testApp', arg_dict)

# create a DataFrame
data = [("Tom", "20", "Developer"), ("Jane", "21", "Developer"), ("Tony", "19", "Developer")]
df = sc.spark.createDataFrame(data).toDF("name", "age", "job")

# write DataFrame as an encrypted csv file
sc.write(df, args.input_encrypt_mode) \
    .mode('overwrite') \
    .option("header", True) \
    .csv(args.input_path)

# get a DataFrame from an encrypted csv file
df = sc.read(args.input_encrypt_mode) \
    .option("header", "true") \
    .csv(args.input_path)

df.select("name").count()

df.select(df["name"], df["age"] + 1).show()

developers = df.filter((df["job"] == "Developer") & df["age"]
                       .between(20, 40)).toDF("name", "age", "job").repartition(1)

sc.write(developers, args.output_encrypt_mode) \
    .mode('overwrite') \
    .option("header", True) \
    .csv(args.output_path)
