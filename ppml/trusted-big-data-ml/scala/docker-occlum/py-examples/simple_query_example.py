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

from bigdl.ppml.kms.utils.kms_argument_parser import KmsArgumentParser
from bigdl.ppml.ppml_context import *

args = KmsArgumentParser().get_arg_dict()

sc = PPMLContext('pyspark-simple-query', args)

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
    .option("header", "true") \
    .csv(args["output_path"])
