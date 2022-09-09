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
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.automl.xgboost import AutoXGBRegressor
from bigdl.orca.automl import hp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


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
        description='AutoXGBRegressor example')
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
    parser.add_argument('-m', '--mode', type=str, default='gridrandom',
                        choices=['gridrandom', 'skopt', 'sigopt'],
                        help='Search algorithms',
                        )

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
    df = spark.read.option('encoding', 'ISO-8859-1').csv(opt.path, header=True).cache()

    train_df, test_df, feature_cols, target_col = data_process(df)

    config = {'random_state': 2}

    recipe = None

    num_rand_samples = 1
    n_estimators_range = (800, 1000)
    max_depth_range = (10, 15)
    lr = (1e-4, 1e-1)
    min_child_weight = [1, 2, 3]

    if opt.mode == 'skopt':
        search_space = {
            "n_estimators": hp.randint(n_estimators_range[0], n_estimators_range[1]),
            "max_depth": hp.randint(max_depth_range[0], max_depth_range[1]),
            "lr": hp.loguniform(lr[0], lr[-1]),
            "min_child_weight": hp.choice(min_child_weight),
        }
        search_alg = "skopt"
        search_alg_params = None
        scheduler = "AsyncHyperBand"
        scheduler_params = dict(
            max_t=50,
            grace_period=1,
            reduction_factor=3,
            brackets=3,
        )
    elif opt.mode == 'sigopt':
        if "SIGOPT_KEY" not in os.environ:
            invalidInputError(False, "Environment Variable 'SIGOPT_KEY' is not set")
        space = [
            {
                "name": "n_estimators",
                "type": "int",
                "bounds": {
                    "min": n_estimators_range[0],
                    "max": n_estimators_range[1]
                },
            },
            {
                "name": "max_depth",
                "type": "int",
                "bounds": {
                    "min": max_depth_range[0],
                    "max": max_depth_range[1]
                },
            },
            {
                "name": "lr",
                "type": "double",
                "bounds": {
                    "min": lr[0],
                    "max": lr[1]
                },
            },
            {
                "name": "min_child_weight",
                "type": "int",
                "bounds": {
                    "min": min_child_weight[0],
                    "max": min_child_weight[-1]
                },
            },

        ]

        # could customize by yourselves, make sure project_id exists
        experiment_name = "AutoXGBoost SigOpt Experiment"
        search_alg_params = dict(
            space=space,
            name=experiment_name,
            max_concurrent=1)
        search_space = dict()
        search_alg = "sigopt"
        search_alg_params = search_alg_params
        scheduler = "AsyncHyperBand"
        scheduler_params = dict(
            max_t=50,
            grace_period=1,
            reduction_factor=3,
            brackets=3,
        )
    else:
        search_space = {
            "n_estimators": hp.grid_search(list(n_estimators_range)),
            "max_depth": hp.grid_search(list(max_depth_range)),
            "lr": hp.loguniform(1e-4, 1e-1),
            "min_child_weight": hp.choice(min_child_weight),
        }
        search_alg = None
        search_alg_params = None
        scheduler = None
        scheduler_params = None

    remote_dir = "hdfs:///tmp/auto_xgb_regressor" if opt.cluster_mode == "yarn" else None

    auto_xgb_reg = AutoXGBRegressor(
        cpus_per_trial=2,
        name="auto_xgb_regressor",
        remote_dir=remote_dir,
        **config)
    auto_xgb_reg.fit(data=train_df,
                     validation_data=test_df,
                     metric="rmse",
                     n_sampling=num_rand_samples,
                     search_space=search_space,
                     search_alg=search_alg,
                     search_alg_params=None,
                     scheduler=scheduler,
                     scheduler_params=scheduler_params,
                     feature_cols=feature_cols,
                     label_cols=target_col,
                     )

    print("Training completed.")
    best_model = auto_xgb_reg.get_best_model()
    x_test = test_df.select(feature_cols).toPandas()
    y_hat = best_model.predict(x_test)
    y_test = test_df.select(target_col).toPandas()

    from bigdl.orca.automl.metrics import Evaluator
    rmse = Evaluator.evaluate(metric="rmse", y_true=y_test, y_pred=y_hat)
    print(f"Evaluate: the square root of mean square error is {rmse:.2f}")
    stop_orca_context()
