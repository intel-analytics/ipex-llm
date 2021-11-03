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
from sklearn.model_selection import train_test_split
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.automl.xgboost import AutoXGBRegressor
from bigdl.orca.automl import hp


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
        init_orca_context(cluster_mode="yarn-client",
                          num_nodes=opt.num_workers,
                          cores=opt.cores,
                          init_ray_on_spark=True)
    else:
        init_orca_context(cluster_mode=opt.cluster_mode,cores=opt.cores, init_ray_on_spark=True)

    import pandas as pd
    df = pd.read_csv(opt.path, encoding='latin-1')
    df.rename(
        columns={
            " FIPS": "FIPS",
            "Age-Adjusted Incidence Rate(Ê) - cases per 100,000": "Age-Adjusted Incidence Rate",
            "Recent 5-Year Trend () in Incidence Rates": "Recent 5-Year Trend"},
        inplace=True)

    # removes rows with cell value _, # or *
    rows = [i for i, x in df.iterrows() if (df.iat[i, 2].find(
        '_') != -1 or df.iat[i, 2].find('#') != -1 or df.iat[i, 6].find('*') != -1)]
    df.drop(rows, inplace=True)
    df.reset_index(drop=True, inplace=True)

    num_rows = 1500  # number of rows to be used in this notebook; max: 2593
    df = df[0:num_rows]
    df["Lower 95% Confidence Interval"] = df["Lower 95% Confidence Interval"].astype(
        "float")
    df["Upper 95% Confidence Interval"] = df["Upper 95% Confidence Interval"].astype(
        "float")
    df["Average Annual Count"] = df["Average Annual Count"].astype("float")
    df["Recent 5-Year Trend"] = df["Recent 5-Year Trend"].astype("float")
    df["Age-Adjusted Incidence Rate"] = df["Age-Adjusted Incidence Rate"].astype(
        "float")

    feature_cols = [
        "FIPS",
        "Lower 95% Confidence Interval",
        "Upper 95% Confidence Interval",
        "Average Annual Count",
        "Recent 5-Year Trend"]
    target_col = "Age-Adjusted Incidence Rate"
    X = df[feature_cols]
    y = df[[target_col]]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=2)

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
            raise RuntimeError("Environment Variable 'SIGOPT_KEY' is not set")
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

    auto_xgb_reg = AutoXGBRegressor(
        cpus_per_trial=2,
        name="auto_xgb_regressor",
        **config)
    auto_xgb_reg.fit(data=(X_train, y_train),
                     validation_data=(X_val, y_val),
                     metric="rmse",
                     n_sampling=num_rand_samples,
                     search_space=search_space,
                     search_alg=search_alg,
                     search_alg_params=None,
                     scheduler=scheduler,
                     scheduler_params=scheduler_params)

    print("Training completed.")
    best_model = auto_xgb_reg.get_best_model()
    y_hat = best_model.predict(X_val)

    from bigdl.orca.automl.metrics import Evaluator
    rmse = Evaluator.evaluate(metric="rmse", y_true=y_val, y_pred=y_hat)
    print(f"Evaluate: the square root of mean square error is {rmse:.2f}")
    stop_orca_context()
