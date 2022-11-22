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

from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, IntegerType, ArrayType, StringType
from pyspark.sql.functions import udf, lit, collect_list, explode

from bigdl.orca import OrcaContext


def read_data(data_dir):
    spark = OrcaContext.get_spark_session()
    schema = StructType([StructField('user', IntegerType(), False),
                         StructField('item', IntegerType(), False)])
    schema_user = StructType(
        [
            StructField('user', IntegerType(), False),
            StructField('gender', StringType(), False),
            StructField('age', IntegerType(), False),
            StructField('occupation', IntegerType(), False),
            StructField('zipcode', StringType(), False)
        ]
    )
    schema_item = StructType(
        [
            StructField('item', IntegerType(), False),
            StructField('title', StringType(), False),
            StructField('category', StringType(), False)
        ]
    )
    # Need spark3 to support delimiter with more than one character.
    df = spark.read.csv(os.path.join(data_dir, 'ratings.dat'),
                        sep="::", schema=schema, header=False)
    df_user = spark.read.csv(os.path.join(data_dir, 'users.dat'),
                             sep="::", schema=schema_user, header=False)
    df_item = spark.read.csv(os.path.join(data_dir, 'movies.dat'),
                             sep="::", schema=schema_item, header=False).drop('title')
    return df, df_user, df_item


def generate_neg_sample(df, item_num, neg_scale):
    def neg_sample(x):
        import random
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


def add_feature(df, df_user, df_item, cat_feature, num_feature):
    df_feat = df.join(df_user, 'user', "inner")
    df_feat = df_feat.join(df_item, 'item', "inner")
    embedding_in_dim = []
    for i in cat_feature:
        indexer = StringIndexer(inputCol=i, outputCol=i + '_index').fit(df_feat)
        df_feat = indexer.transform(df_feat)
        df_feat = df_feat.drop(i).withColumnRenamed(i + '_index', i)
        df_feat = df_feat.withColumn(i, df_feat[i].cast('int'))
        embedding_in_dim.append(df_feat.agg({i: "max"}).collect()[0][f"max({i})"])
    for i in num_feature:
        assembler = VectorAssembler(inputCols=[i], outputCol=i + '_vec')
        scaler = MinMaxScaler(inputCol=i + '_vec', outputCol=i + '_scaler')
        pipeline = Pipeline(stages=[assembler, scaler])
        scalerModel = pipeline.fit(df_feat)
        df_feat = scalerModel.transform(df_feat)
        df_feat = df_feat.drop(i, i + '_vec').withColumnRenamed(i + '_scaler', i)
        df_feat.show()
    return df_feat, embedding_in_dim


def data_process(data_dir, cat_feature, num_feature, neg_scale=4):
    df, df_user, df_item = read_data(data_dir)
    user_num = df.agg({'user': "max"}).collect()[0]["max(user)"] + 1
    item_num = df.agg({'item': "max"}).collect()[0]["max(item)"] + 1

    df = generate_neg_sample(df, item_num, neg_scale=neg_scale)
    df_add_feature, embedding_in_dim = add_feature(df, df_user, df_item, cat_feature, num_feature)

    train_df, val_df = df_add_feature.randomSplit([0.8, 0.2], seed=100)

    return train_df, val_df, embedding_in_dim, user_num, item_num


if __name__ == "__main__":
    from bigdl.orca import init_orca_context, stop_orca_context

    sc = init_orca_context()

    cat_feature = ['zipcode', 'gender', 'occupation', 'category']
    num_feature = ['age']

    train_data, test_data, embedding_in_dim, user_num, item_num = \
        data_process("ml-1m", cat_feature, num_feature, neg_scale=4)
    train_data.write.csv('./train_dataframe', header=True, sep=',', mode='overwrite')
    test_data.write.csv('./test_dataframe', header=True, sep=',', mode='overwrite')
    stop_orca_context()
