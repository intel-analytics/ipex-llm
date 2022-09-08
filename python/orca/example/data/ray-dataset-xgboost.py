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

# XGBoost on ray is needed to run this example.
# Please refer to https://docs.ray.io/en/latest/xgboost-ray.html to install it.

from xgboost_ray import RayDMatrix, train, RayParams, RayXGBRegressor
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data import spark_df_to_ray_dataset


def data_process(df):
    df = df.withColumnRenamed(" FIPS", "FIPS")\
        .withColumnRenamed("Age-Adjusted Incidence Rate(Ê) - cases per 100,000",
                           "Age-Adjusted Incidence Rate") \
        .withColumnRenamed("Recent 5-Year Trend () in Incidence Rates", "Recent 5-Year Trend") \
        .withColumnRenamed("Lower 95% Confidence Interval3", "Lower 95% Confidence Interval") \
        .withColumnRenamed("Upper 95% Confidence Interval4", "Upper 95% Confidence Interval") \
        .cache()

    # removes rows with cell value _, # or *
    df = df.where(~(col('Age-Adjusted Incidence Rate').contains('_') | col(
        'Age-Adjusted Incidence Rate').contains('#') | col('Recent Trend').contains('*')))

    df = df.withColumn("FIPS", df["FIPS"].cast("long")) \
        .withColumn("Lower 95% Confidence Interval",
                    df["Lower 95% Confidence Interval"].cast("double")) \
        .withColumn("Upper 95% Confidence Interval",
                    df["Upper 95% Confidence Interval"].cast("double")) \
        .withColumn("Average Annual Count", df["Average Annual Count"].cast("double")) \
        .withColumn("Recent 5-Year Trend", df["Recent 5-Year Trend"].cast("double")) \
        .withColumn("Age-Adjusted Incidence Rate",
                    df["Age-Adjusted Incidence Rate"].cast("double")) \
        .cache()

    feature_cols = [
        "FIPS",
        "Lower 95% Confidence Interval",
        "Upper 95% Confidence Interval",
        "Average Annual Count",
        "Recent 5-Year Trend"]
    target_col = "Age-Adjusted Incidence Rate"

    useful_columns = feature_cols + [target_col]
    df = df.select(useful_columns)

    splits = df.randomSplit([0.8, 0.2], seed=24)
    train_df = splits[0]
    test_df = splits[1]

    print("Number of records in train_df", train_df.count())
    print("Number of records in test_df", test_df.count())

    return train_df, test_df, feature_cols, target_col


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ray dataset xgboost example')
    parser.add_argument('-p', '--path', type=str,
                        default="./data/incd.csv",
                        help='Training data path')
    parser.add_argument('--cluster_mode',
                        type=str,
                        default='local',
                        choices=['local', 'yarn', "spark-submit"],
                        help='The mode for the Spark cluster.')
    parser.add_argument('--cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('-n', '--num_workers', type=int, default=2,
                        help='The number of workers to be launched.')

    opt = parser.parse_args()
    if opt.cluster_mode == "yarn":
        sc = init_orca_context(cluster_mode="yarn-client",
                               num_nodes=opt.num_workers,
                               cores=opt.cores,
                               init_ray_on_spark=True)
    else:
        sc = init_orca_context(cluster_mode=opt.cluster_mode,
                               cores=opt.cores,
                               init_ray_on_spark=True)
    spark = SparkSession(sc)
    df = spark.read.option('encoding', 'ISO-8859-1')\
        .csv(opt.path, header=True)\
        .repartition(opt.num_workers)

    train_df, test_df, feature_cols, target_col = data_process(df)

    train_dataset = spark_df_to_ray_dataset(train_df)
    test_dataset = spark_df_to_ray_dataset(test_df)
    print(train_dataset)
    print(test_dataset)

    # Then convert them into DMatrix used by xgboost
    dtrain = RayDMatrix(train_dataset, label=target_col, feature_names=feature_cols)
    dtest = RayDMatrix(test_dataset, label=target_col, feature_names=feature_cols)
    # Configure the XGBoost model
    config = {
        "tree_method": "hist",
        "eval_metric": ["rmse"],
    }
    evals_result = {}
    # Train the model
    bst = train(
        config,
        dtrain,
        evals=[(dtest, "eval")],
        evals_result=evals_result,
        ray_params=RayParams(max_actor_restarts=1,
                             num_actors=opt.num_workers,
                             cpus_per_actor=1),
        num_boost_round=10)
    # print evaluation stats
    print("Final validation error: {:.4f}".format(evals_result["eval"]["rmse"][-1]))
