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
import os

from bigdl.ppml.ppml_context import *
from bigdl.ppml.utils.supportive import timing

from pyspark.sql.functions import desc, col

"""
execute the following command to run this example on local

python simple_query_example.py \
--simple_app_id your_app_id \
--simple_app_key your_app_key \
--primary_key_path /your/primary/key/path/primaryKey \
--data_key_path /your/data/key/path/dataKey \
--input_path /your/file/input/path \
--output_path /your/file/output/path \
--input_encrypt_mode plain_text \
--output_encrypt_mode AES/CBC/PKCS5Padding

"""


@timing("1/4 load data from csv")
def read_from_csv(context, mode, path):
    return context.read(mode)\
        .option("header", "true")\
        .csv(os.path.join(path, "people.csv"))


@timing("2/4 load data from parquet")
def read_from_parquet(context, mode, path):
    return context.read(mode).parquet(os.path.join(path, "people.parquet"))


@timing("3/4 do sql operation")
def do_sql_operation(csv_df, parquet_df):
    # union
    union_df = csv_df.unionByName(parquet_df)\
        .withColumnRenamed("age", "age_string")\
        .withColumn("age", col("age_string").cast("int"))\
        .drop("age_string")
    # filter
    filter_df = union_df.filter(union_df["age"].between(20, 40))
    # count people in each job
    count_df = filter_df.groupby("job").count()
    # calculate average age in each job
    avg_df = filter_df.groupby("job")\
        .avg("age")\
        .withColumnRenamed("avg(age)", "average_age")
    # join and sort
    result_df = count_df.join(avg_df, "job").sort(desc("average_age"))

    return result_df


@timing("4/4 encrypt and save outputs")
def save(context, df, encrypted_mode, path):
    context.write(df, encrypted_mode).mode('overwrite').json(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_app_id", type=str, required=True, help="simple app id")
    parser.add_argument("--simple_app_key", type=str, required=True, help="simple app key")
    parser.add_argument("--primary_key_path", type=str, required=True, help="primary key path")
    parser.add_argument("--data_key_path", type=str, required=True, help="data key path")
    parser.add_argument("--input_encrypt_mode", type=str, required=True, help="input encrypt mode")
    parser.add_argument("--output_encrypt_mode", type=str, required=True, help="output encrypt mode")
    parser.add_argument("--input_path", type=str, required=True, help="input path")
    parser.add_argument("--output_path", type=str, required=True, help="output path")
    parser.add_argument("--kms_type", type=str, default="SimpleKeyManagementService",
                        help="SimpleKeyManagementService or EHSMKeyManagementService")
    args = parser.parse_args()
    arg_dict = vars(args)

    # create a PPMLContext
    sc = PPMLContext('testApp', arg_dict)

    # 1.read data from csv
    df1 = read_from_csv(sc, args.input_encrypt_mode, args.input_path)

    # 2.read data from parquet
    df2 = read_from_parquet(sc, args.input_encrypt_mode, args.input_path)

    # 3.do sql operation
    result = do_sql_operation(df1, df2)

    result.show()

    # 4.save as json file
    save(sc, result, args.output_encrypt_mode, args.output_path)
