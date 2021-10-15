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

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.automl.xgboost import AutoXGBClassifier
from bigdl.orca.automl import hp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AutoXGBRegressor example')
    parser.add_argument('-p', '--path', type=str,
                        default="./data/airline_14col.data",
                        help='Training data path')
    parser.add_argument('--cluster_mode',
                        type=str,
                        default='local',
                        choices=['local', 'yarn'],
                        help='The mode for the Spark cluster.')
    parser.add_argument('--cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('-n', '--num_workers', type=int, default=2,
                        help='The number of workers to be launched.')
    parser.add_argument('-m', '--mode', type=str, default='gridrandom',
                        choices=['gridrandom', 'skopt'],
                        help='The search algorithm to use.')
    opt = parser.parse_args()
    if opt.cluster_mode == "yarn":
        init_orca_context(cluster_mode="yarn-client",
                          num_nodes=opt.num_workers,
                          cores=opt.cores,
                          init_ray_on_spark=True)
    else:
        init_orca_context(cores=opt.cores, init_ray_on_spark=True)

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

    # feature cols = input_cols - "ArrDelay"
    X = pdf[input_cols[:-1]]
    y = pdf[["ArrDelayBinary"]]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

    num_rand_samples = 1
    n_estimators_range = (50, 1000)
    max_depth_range = (2, 15)

    config = {"tree_method": 'hist', "learning_rate": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1}

    if opt.mode == 'skopt':
        search_space = {
            "n_estimators": hp.randint(n_estimators_range[0], n_estimators_range[1]),
            "max_depth": hp.randint(max_depth_range[0], max_depth_range[1]),
            "lr": hp.loguniform(1e-4, 1e-1)
        }
        search_alg = "skopt"
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
            "lr": hp.loguniform(1e-4, 1e-1)
        }
        search_alg = None
        scheduler = None
        scheduler_params = None

    auto_xgb_clf = AutoXGBClassifier(cpus_per_trial=4, name="auto_xgb_classifier", **config)
    import time
    start = time.time()
    auto_xgb_clf.fit(data=(X_train, y_train),
                     validation_data=(X_val, y_val),
                     metric="error",
                     metric_mode="min",
                     n_sampling=num_rand_samples,
                     search_space=search_space,
                     search_alg=search_alg,
                     search_alg_params=None,
                     scheduler=scheduler,
                     scheduler_params=scheduler_params)
    end = time.time()
    print("elapse: ", (end-start), "s")
    best_model = auto_xgb_clf.get_best_model()

    y_hat = best_model.predict(X_val)
    from bigdl.orca.automl.metrics import Evaluator
    accuracy = Evaluator.evaluate(metric="accuracy", y_true=y_val, y_pred=y_hat)
    print("Evaluate: accuracy is", accuracy)
    stop_orca_context()
