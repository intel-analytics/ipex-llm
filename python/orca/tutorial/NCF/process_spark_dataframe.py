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

from pyspark.sql.types import StructField, StructType, IntegerType, ArrayType
from pyspark.sql.functions import udf, lit, collect_list, explode

from bigdl.orca import OrcaContext


def read_data(data_dir):
    spark = OrcaContext.get_spark_session()
    schema = StructType([StructField('user', IntegerType(), False),
                         StructField('item', IntegerType(), False)])
    # Need spark3 to support delimiter with more than one character.
    df = spark.read.csv(
        os.path.join(data_dir, "ratings.dat"), sep="::", schema=schema, header=False)
    return df


def generate_neg_sample(df, item_num):
    def neg_sample(x):
        import random
        neg_scale = 4
        neg_res = []
        for _ in x:
            for i in range(neg_scale):
                neg_item_index = random.randint(1, item_num - 1)
                while neg_item_index in x:
                    neg_item_index = random.randint(1, item_num - 1)
                neg_res.append(neg_item_index)
        return neg_res

    df = df.withColumn('label', lit(1.0))

    neg_sample_udf = udf(neg_sample, ArrayType(IntegerType(), False))
    df_neg = df.groupBy('user').agg(neg_sample_udf(collect_list('item')).alias('item_list'))
    df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
    df_neg = df_neg.withColumn('label', lit(0.0))
    df_neg = df_neg.withColumnRenamed('col', 'item')

    df = df.unionByName(df_neg)
    df = df.repartition(df.rdd.getNumPartitions())

    return df


if __name__ == "__main__":
    from bigdl.orca import init_orca_context, stop_orca_context

    sc = init_orca_context()
    df = read_data('./ml-1m')
    item_num = df.agg({'item': "max"}).collect()[0]["max(item)"]
    df = generate_neg_sample(df, item_num)
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=100)
    train_data.write.csv('./train_dataframe', header=True, sep=',', mode='overwrite')
    test_data.write.csv('./test_dataframe', header=True, sep=',', mode='overwrite')
    stop_orca_context()
