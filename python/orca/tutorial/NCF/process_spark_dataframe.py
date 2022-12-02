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
    df_rating = spark.read.csv(os.path.join(data_dir, 'ratings.dat'),
                               sep="::", schema=schema, header=False)
    df_user = spark.read.csv(os.path.join(data_dir, 'users.dat'),
                             sep="::", schema=schema_user, header=False)
    df_item = spark.read.csv(os.path.join(data_dir, 'movies.dat'),
                             sep="::", schema=schema_item, header=False)
    return df_rating, df_user, df_item


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


def string_index(df, col):
    indexer = StringIndexer(inputCol=col, outputCol=col + '_index').fit(df)
    df = indexer.transform(df)
    df = df.drop(col).withColumnRenamed(col + '_index', col)
    # The StringIndexer output is float type.
    # Change to 1-based index with 0 reversed for unknown features.
    df = df.withColumn(col, df[col].cast('int') + 1)
    embed_dim = df.agg({col: "max"}).collect()[0][f"max({col})"] + 1
    return df, embed_dim


def num_scale(df, col):
    assembler = VectorAssembler(inputCols=[col], outputCol=col + '_vec')
    scaler = MinMaxScaler(inputCol=col + '_vec', outputCol=col + '_scaled')
    pipeline = Pipeline(stages=[assembler, scaler])
    scalerModel = pipeline.fit(df)
    df = scalerModel.transform(df)
    df = df.drop(col, col + '_vec').withColumnRenamed(col + '_scaled', col)
    return df


def add_feature(df, df_user, df_item, sparse_features, dense_features):
    sparse_feats_input_dims = []
    for i in sparse_features:
        if i in df_user.columns:
            df_user, embed_dim = string_index(df_user, i)
        else:
            df_item, embed_dim = string_index(df_item, i)
        sparse_feats_input_dims.append(embed_dim)
    for i in dense_features:
        if i in df_user.columns:
            df_user = num_scale(df_user, i)
        else:
            df_item = num_scale(df_item, i)
    df_feat = df.join(df_user, 'user', "inner")
    df_feat = df_feat.join(df_item, 'item', "inner")
    return df_feat, sparse_feats_input_dims


def prepare_data(data_dir, neg_scale=4):
    sparse_features = ['zipcode', 'gender', 'category']
    dense_features = ['age']
    df_rating, df_user, df_item = read_data(data_dir)

    user_num = df_rating.agg({'user': "max"}).collect()[0]["max(user)"] + 1
    item_num = df_rating.agg({'item': "max"}).collect()[0]["max(item)"] + 1

    df_rating = generate_neg_sample(df_rating, item_num, neg_scale=neg_scale)
    df, sparse_feats_input_dims = \
        add_feature(df_rating, df_user, df_item, sparse_features, dense_features)
    # Occupation is already indexed.
    sparse_features.append('occupation')
    occupation_num = df.agg({'occupation': 'max'}).collect()[0]['max(occupation)'] + 1
    sparse_feats_input_dims.append(occupation_num)
    feature_cols = ['user', 'item'] + sparse_features + dense_features
    label_cols = ['label']

    train_df, val_df = df.randomSplit([0.8, 0.2], seed=100)

    return train_df, val_df, user_num, item_num, \
        sparse_feats_input_dims, len(dense_features), feature_cols, label_cols


if __name__ == "__main__":
    from bigdl.orca import init_orca_context, stop_orca_context

    sc = init_orca_context()

    train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
        feature_cols, label_cols = prepare_data("./ml-1m")
    train_data.write.parquet('./train_dataframe', mode='overwrite')
    test_data.write.parquet('./test_dataframe', mode='overwrite')
    stop_orca_context()
