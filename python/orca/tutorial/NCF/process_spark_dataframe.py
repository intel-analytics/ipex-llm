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
import os

from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, IntegerType, LongType, ArrayType, StringType
from pyspark.sql.functions import udf, lit, collect_list, explode, concat

from bigdl.orca import OrcaContext

# user/item ids and sparse features are converted to LongType to be compatible with
# lower versions of PyTorch such as 1.7.1.

sparse_features = ["zipcode", "gender", "category", "occupation"]
dense_features = ["age"]


def read_data(data_dir, dataset):
    spark = OrcaContext.get_spark_session()
    print("Loading data...")

    schema = StructType([StructField("user", LongType(), False),
                         StructField("item", LongType(), False)])
    if dataset == "ml-1m":
        schema_user = StructType(
            [
                StructField("user", LongType(), False),
                StructField("gender", StringType(), False),
                StructField("age", IntegerType(), False),
                StructField("occupation", StringType(), False),
                StructField("zipcode", StringType(), False)
            ]
        )
        schema_item = StructType(
            [
                StructField("item", LongType(), False),
                StructField("title", StringType(), False),
                StructField("category", StringType(), False)
            ]
        )
        # Need spark3 to support delimiter with more than one character.
        df_rating = spark.read.csv(os.path.join(data_dir, dataset, "ratings.dat"),
                                   sep="::", schema=schema, header=False)
        df_user = spark.read.csv(os.path.join(data_dir, dataset, "users.dat"),
                                 sep="::", schema=schema_user, header=False)
        df_item = spark.read.csv(os.path.join(data_dir, dataset, "movies.dat"),
                                 sep="::", schema=schema_item, header=False)
    else:  # ml-100k
        schema_user = StructType(
            [
                StructField("user", LongType(), False),
                StructField("age", IntegerType(), False),
                StructField("gender", StringType(), False),
                StructField("occupation", StringType(), False),
                StructField("zipcode", StringType(), False)
            ]
        )
        schema_item = StructType(
            [
                StructField("item", LongType(), False),
                StructField("title", StringType(), False),
                StructField("date", StringType(), False),
                StructField("vdate", StringType(), False),
                StructField("URL", StringType(), False),
            ]
            + [StructField(f"col{i}", StringType(), False) for i in range(19)]
        )
        df_rating = spark.read.csv(os.path.join(data_dir, dataset, "u.data"),
                                   sep="\t", schema=schema, header=False)
        df_user = spark.read.csv(os.path.join(data_dir, dataset, "u.user"),
                                 sep="|", schema=schema_user, header=False)
        df_item = spark.read.csv(os.path.join(data_dir, dataset, "u.item"),
                                 sep="|", schema=schema_item, header=False)
        # Merge multiple one-hot columns into one movie category column
        df_item = df_item.select(
            df_item.item,
            concat(*[df_item[f"col{i}"] for i in range(19)]).alias("category")
        )
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

    df = df.withColumn("label", lit(1.0))

    neg_sample_udf = udf(neg_sample, ArrayType(LongType(), False))
    df_neg = df.groupBy("user").agg(neg_sample_udf(collect_list("item")).alias("item_list"))
    df_neg = df_neg.select(df_neg.user, explode(df_neg.item_list))
    df_neg = df_neg.withColumn("label", lit(0.0))
    df_neg = df_neg.withColumnRenamed("col", "item")

    df = df.unionByName(df_neg)
    df = df.repartition(df.rdd.getNumPartitions())
    return df


def string_index(df, col):
    indexer = StringIndexer(inputCol=col, outputCol=col + "_index").fit(df)
    df = indexer.transform(df)
    df = df.drop(col).withColumnRenamed(col + "_index", col)
    # The StringIndexer output is float type.
    # Change to 1-based index with 0 reversed for unknown features.
    df = df.withColumn(col, df[col].cast("long") + 1)
    embed_dim = df.agg({col: "max"}).collect()[0][f"max({col})"] + 1
    return df, embed_dim


def min_max_scale(df, col):
    assembler = VectorAssembler(inputCols=[col], outputCol=col + "_vec")
    scaler = MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled")
    pipeline = Pipeline(stages=[assembler, scaler])
    scalerModel = pipeline.fit(df)
    df = scalerModel.transform(df)
    df = df.drop(col, col + "_vec").withColumnRenamed(col + "_scaled", col)
    return df


def merge_features(df, df_user, df_item, sparse_features, dense_features):
    sparse_feats_input_dims = []
    for i in sparse_features:
        if i in df_user.columns:
            df_user, embed_dim = string_index(df_user, i)
        else:
            df_item, embed_dim = string_index(df_item, i)
        sparse_feats_input_dims.append(embed_dim)
    for i in dense_features:
        if i in df_user.columns:
            df_user = min_max_scale(df_user, i)
        else:
            df_item = min_max_scale(df_item, i)
    df_feat = df.join(df_user, "user", "inner")
    df_feat = df_feat.join(df_item, "item", "inner")
    return df_feat, sparse_feats_input_dims


def prepare_data(data_dir="./", dataset="ml-1m", neg_scale=4):
    df_rating, df_user, df_item = read_data(data_dir, dataset)

    user_num = df_rating.agg({"user": "max"}).collect()[0]["max(user)"] + 1
    item_num = df_rating.agg({"item": "max"}).collect()[0]["max(item)"] + 1

    df_rating = generate_neg_sample(df_rating, item_num, neg_scale=neg_scale)
    df, sparse_feats_input_dims = \
        merge_features(df_rating, df_user, df_item, sparse_features, dense_features)

    train_df, val_df = df.randomSplit([0.8, 0.2], seed=100)
    return train_df, val_df, user_num, item_num, \
        sparse_feats_input_dims, len(dense_features), get_feature_cols(), get_label_cols()


def get_feature_cols():
    return ["user", "item"] + sparse_features + dense_features


def get_label_cols():
    return ['label']


if __name__ == "__main__":
    from utils import init_orca, stop_orca_context

    init_orca("local")
    train_df, test_df, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
        feature_cols, label_cols = prepare_data()
    train_df.write.parquet("./train_processed_dataframe.parquet", mode="overwrite")
    test_df.write.parquet("./test_processed_dataframe.parquet", mode="overwrite")
    stop_orca_context()
