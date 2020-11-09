#
# Copyright 2018 Analytics Zoo Authors.
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

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray import RayContext
from zoo.automl.model.XGBoost import XGBoost
from zoo.orca.automl.AutoXGBoost import AutoXGBoost
from zoo.automl.config.recipe import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AutoXGBRegressor example')
    parser.add_argument('-p', '--path', type=str,
                        help='Training data path')
    parser.add_argument('--hadoop_conf', type=str,
                        help='The path to the hadoop configuration folder. Required if you '
                             'wish to run on yarn clusters. Otherwise, run in local mode.')
    parser.add_argument('--conda_name', type=str,
                        help='The name of conda environment. Required if you '
                             'wish to run on yarn clusters.')
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('-n', '--num_workers', type=int, default=2,
                        help='The number of workers to be launched.')
    opt = parser.parse_args()
    if opt.hadoop_conf:
        assert opt.conda_name is not None, "conda_name must be specified for yarn mode"
        sc = init_spark_on_yarn(
            hadoop_conf=opt.hadoop_conf,
            conda_name=opt.conda_name,
            num_executors=opt.num_workers,
            executor_cores=opt.executor_cores)
    else:
        sc = init_spark_on_local(cores="*")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    input_cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]

    # path = "./airline_14col.data"
    # num_rows = 2500000  # number of rows to be used in this notebook; max: 115000000
    num_rows = 2500

    dataset_config = {
        "nrows": num_rows,  # Max Rows in dataset: 115000000
        "delayed_threshold": 10,
    }
    pdf = pd.read_csv(opt.path, names=input_cols, nrows=dataset_config["nrows"])

    pdf["ArrDelayBinary"] = 1.0 * (
        pdf["ArrDelay"] > dataset_config["delayed_threshold"]
    )

    # drop non-binary label column [ delay time ]
    pdf = pdf[pdf.columns.difference(["ArrDelay"])]

    # encode categoricals as numeric
    for col in pdf.select_dtypes(["object"]).columns:
        pdf[col] = pdf[col].astype("category").cat.codes.astype(np.int32)

    # cast all columns to int32
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(np.float32)  # needed for random forest

    # put target/label column first [ classic XGBoost standard ]
    output_cols = ["ArrDelayBinary"] + input_cols[:-1]

    pdf = pdf.reindex(columns=output_cols)

    train_df, val_df = train_test_split(pdf, test_size=0.2, random_state=2)

    n_estimators_range = (50, 1000)
    max_depth_range = (2, 15)
    # max_features_range = (0.1, 0.8)

    input_cols.remove("ArrDelay")
    config = {"tree_method": 'hist', "learning_rate": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1, "n_jobs": 4}

    estimator = AutoXGBoost().classifier(feature_cols=input_cols,
                                         target_col="ArrDelayBinary",
                                         config=config,
                                         search_alg="skopt",
                                         search_alg_params=None,
                                         scheduler="AsyncHyperBand",
                                         scheduler_params=dict(
                                             max_t=50,
                                             grace_period=1,
                                             reduction_factor=3,
                                             brackets=3,
                                         ),
                                         )

    import time
    start = time.time()
    pipeline = estimator.fit(train_df,
                             validation_df=val_df,
                             metric="error",
                             recipe=XgbRegressorSkOptRecipe(num_rand_samples=3))
    end = time.time()
    print("elapse: ", (end-start))
    accuracy = pipeline.evaluate(val_df, metrics=["accuracy"])
    print("Evaluate: accuracy is", accuracy)
