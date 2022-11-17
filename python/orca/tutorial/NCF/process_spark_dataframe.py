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
import os.path

import tensorflow as tf
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructField, StructType, IntegerType, ArrayType, StringType
from pyspark.sql.functions import udf, lit, collect_list, explode


def read_data(data_dir, spark):
    schema = StructType([StructField('user', IntegerType(), False),
                         StructField('item', IntegerType(), False)])
    # Need spark3 to support delimiter with more than one character.
    df = spark.read.csv(os.path.join(data_dir, "ratings.dat"), sep="::", schema=schema, header=False)
    df = df.withColumn('label', lit(1.0))
    return df


def generate_neg_sample(df):
    embedding_in_dim = {}
    for i, c, in enumerate(['user', 'item']):
        print(f'[INFO] ==> begin calculate {c} embedding_in_dim')
        embedding_in_dim[c] = df.agg({c: "max"}).collect()[0][f"max({c})"]
    print(embedding_in_dim)

    def neg_sample(x):
        import random
        neg_scale = 4
        neg_res = []
        for _ in x:
            for i in range(neg_scale):
                neg_item_index = random.randint(1, embedding_in_dim['item'])
                while neg_item_index in x:
                    neg_item_index = random.randint(1, embedding_in_dim['item'])
                neg_res.append(neg_item_index)
        return neg_res

    neg_sample_udf = udf(neg_sample, ArrayType(IntegerType(), False))

    df_neg = df.groupBy('user').agg(neg_sample_udf(collect_list('item')).alias('item_list'))
    df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
    df_neg = df_neg.withColumn('label', lit(0.0))
    df_neg = df_neg.withColumnRenamed('col', 'item')
    df = df.unionByName(df_neg)
    df = df.repartition(df.rdd.getNumPartitions())

    return df, embedding_in_dim


def split_dataset(df):
    train_df, val_df = df.randomSplit([0.8, 0.2], 100)
    train_size = train_df.count()
    val_size = val_df.count()
    return train_df, val_df, train_size, val_size
