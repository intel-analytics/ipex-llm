from pyspark.sql import SparkSession

from bigdl.orca import init_orca_context, stop_orca_context

# Please use 0.10.0 version for Spark3.2 and 0.9.5-13-d1b51517-SNAPSHOT version for Spark3.1
import pyspark
import os

spark_conf = {"spark.app.name": "recsys-lightGBM",
              "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.9.5-13-d1b51517-SNAPSHOT",
              "spark.jars.repositories": "https://mmlspark.azureedge.net/maven"}

spark = pyspark.sql.SparkSession.builder.appName("MyApp")\
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5-13-d1b51517-SNAPSHOT")\
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")\
    .getOrCreate()
import synapse.ml

df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(
        "/home/arda/intelWork/data/bankruptcy/data.csv"
    )
)
# print dataset size
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()

train, test = df.randomSplit([0.85, 0.15], seed=1)
from pyspark.ml.feature import VectorAssembler
feature_cols = df.columns[1:]
featurizer = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)
train_data = featurizer.transform(train)['Bankrupt?', 'features']
test_data = featurizer.transform(test)['Bankrupt?', 'features']

train_data.printSchema()
train_data.show(10)
print(train_data.count())

from synapse.ml.lightgbm import LightGBMClassifier
model = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True)

model = model.fit(train_data)


from synapse.ml.lightgbm import LightGBMClassificationModel

# if os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":
#     model.saveNativeModel("/models/lgbmclassifier.model")
#     model = LightGBMClassificationModel.loadNativeModelFromFile("/models/lgbmclassifier.model")
# else:
#     model.saveNativeModel("/lgbmclassifier.model")
#     model = LightGBMClassificationModel.loadNativeModelFromFile("/lgbmclassifier.model")

predictions = model.transform(test_data)
# predictions.show(10)
# print(predictions.count())

#
# # from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
# # sc = init_orca_context("local")
# # spark = OrcaContext.get_spark_session
input = "/home/arda/intelWork/data/tweet/xgb_processed/train/"
df = spark.read.parquet(input)
df.printSchema()
print(df.count())
# df.repartition(2)
# # print dataset size
# print("records read: " + str(df.count()))
# print("Schema: ")
# df.printSchema()
# df.cache()
#
# train, test = df.randomSplit([0.85, 0.15], seed=1)
#
# from synapse.ml.lightgbm import LightGBMClassifier
#
# model = LightGBMClassifier(
#     objective="binary", featuresCol="features", labelCol="label")
#
# model = model.fit(train)
# print("after fitting")
# predictions = model.transform(test)
# predictions.show(10)

import time
import os
import pandas as pd
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
from bigdl.orca.learn.tf2.estimator import Estimator
from friesian.example.wnd.train.wnd_train_recsys import ColumnFeatureInfo, build_model
from bigdl.orca.data.file import exists, makedirs
import tensorflow as tf

ratings = movielens.get_id_ratings(data_dir)
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])
ratings_tbl = FeatureTable.from_pandas(ratings) .cast(["user", "item", "rate"], "int")
ratings_tbl.cache()

user_tbl = FeatureTable.read_csv(data_dir + "/ml-1m/users.dat", delimiter=":") \
    .select("_c0", "_c2", "_c4", "_c6"
                                                                                      , "_c8") \
    .rena \
    me({"_c0": "user", "_c2": "gender", "_c4": "age", "_c6": "occupation", "_c8": "zip"}) \
    .cast(["user"], "int")
user_tbl.cache()

user_tbl = user_tbl.fillna("0", "zip")

user_stats = ratings_tbl.group_by("user", agg={"item": "count", "rate": "mean"}) \
    .rena \
    me({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

item_stats = ratings_tbl.group_by("item", agg={"user": "count", "rate": "mean"}) \
    .rena \
    me({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
item_stats, item_min_max = item_stats.min_max_scale(["item_visits", "item_mean_rate"])

user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zip", "occupation"])

item_size = item_stats.select("item").distinct().size()
ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item",
                                               label_col="label", neg_num=1)

user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zip"]], [50, 200])

user_tbl = user_tbl.join(user_stats, on="user")
full = ratings_tbl.join(user_tbl, on="user").join(item_stats, on="item")
stats = full.get_stats(cat_cols, "max")
wide_dims = [stats[key] for key in wide_cols]
wide_cross_dims = [stats[key] for key in wide_cross_cols]
embed_dims = [stats[key] for key in embed_cols]
indicator_dims = [stats[key] for key in indicator_cols]
column_info = ColumnFeatureInfo(wide_base_cols=wide_cols,
                                wide_base_dims=wide_dims,
                                wide_cross_cols=wide_cross_cols,
                                wide_cross_dims=wide_cross_dims,
                                indicator_cols=indicator_cols,
                                indicator_dims=indicator_dims,
                                embed_cols=embed_cols,
                                embed_in_dims=embed_dims,
                                embed_out_dims=[8] * len(embed_dims),
                                continuous_cols=num_cols,
                                label="label")

train_tbl, test_tbl = full.select("label", *column_info.feature_cols).random_split([0.8, 0.2])
train_count, test_count = train_tbl.size(), test_tbl.size()
data_dir = "./movielens"
model_dir = "./wnd"
wide_cols = ["gender", "age", "occupation", "zip"]
wide_cross_cols = ["gender_age", "age_zip"]
indicator_cols = ["gender", "age", "occupation"]
embed_cols = ["user", "item"]
num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
cat_cols = wide_cols + wide_cross_cols + embed_cols
batch_size = 1024

train_tbl.df.show(10)
train_tbl.df.printSchema()
