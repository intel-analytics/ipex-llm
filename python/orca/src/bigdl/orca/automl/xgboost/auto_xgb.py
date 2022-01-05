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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from bigdl.orca.automl.xgboost.XGBoost import XGBoostModelBuilder
from bigdl.orca.automl.auto_estimator import AutoEstimator


class AutoXGBClassifier(AutoEstimator):
    def __init__(self,
                 logs_dir="/tmp/auto_xgb_classifier_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost classifier

        Example:
            >>> search_space = {"n_estimators": hp.grid_search([50, 1000]),
                                "max_depth": hp.grid_search([2, 15]),
                                "lr": hp.loguniform(1e-4, 1e-1)}
            >>> auto_xgb_clf = AutoXGBClassifier(cpus_per_trial=4,
                                                 name="auto_xgb_classifier",
                                                 **config)
            >>> auto_xgb_clf.fit(data=(X_train, y_train),
                                 validation_data=(X_val, y_val),
                                 metric="error",
                                 metric_mode="min",
                                 n_sampling=1,
                                 search_space=search_space)
            >>> best_model = auto_xgb_clf.get_best_model()

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
            The value will also be assigned to n_jobs in xgboost,
            which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='classifier',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)

    def fit(self,
            data,
            epochs=1,
            validation_data=None,
            metric=None,
            metric_mode=None,
            metric_threshold=None,
            n_sampling=1,
            search_space=None,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            feature_cols=None,
            target_cols=None,
            ):
        """
        Automatically fit the model and search for the best hyperparameters.

        :param data: A Spark DataFrame, a tuple of ndarrays or a function.
            If data is a tuple of ndarrays, it should be in the form of (x, y),
            where x is training input data and y is training target data.
            If data is a function, it should takes config as argument and returns a tuple of
            ndarrays in the form of (x, y).
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
            If you have also set metric_threshold, a trial will stop if either it has been
            optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param validation_data: Validation data. Validation data type should be the same as data.
        :param metric: String or customized evaluation metric function.
            If string, metric is the evaluation metric name to optimize, e.g. "mse".
            If callable function, it signature should be func(y_true, y_pred), where y_true and
            y_pred are numpy ndarray. The function should return a float value as evaluation result.
        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.
            You have to specify metric_mode if you use a customized metric function.
            You don't have to specify metric_mode if you use the built-in metric in
            bigdl.orca.automl.metrics.Evaluator.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: Number of times to sample from the search_space. Defaults to 1.
            If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.
            If this is -1, (virtually) infinite samples are generated
            until a stopping condition is met.
        :param search_space: a dict for search space
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
            metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param target_cols: target column names if data is Spark DataFrame.
        """
        data, validation_data = _maybe_convert_spark_df_to_ndarray(data,
                                                                   validation_data,
                                                                   feature_cols,
                                                                   target_cols)
        super().fit(data=data,
                    epochs=epochs,
                    validation_data=validation_data,
                    metric=metric,
                    metric_mode=metric_mode,
                    metric_threshold=metric_threshold,
                    n_sampling=n_sampling,
                    search_space=search_space,
                    search_alg=search_alg,
                    search_alg_params=search_alg_params,
                    scheduler=scheduler,
                    scheduler_params=scheduler_params,
                    )


class AutoXGBRegressor(AutoEstimator):
    def __init__(self,
                 logs_dir="/tmp/auto_xgb_regressor_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost regressor

        Example:
            >>> search_space = {"n_estimators": hp.grid_search([800, 1000]),
                                "max_depth": hp.grid_search([10, 15]),
                                "lr": hp.loguniform(1e-4, 1e-1),
                                "min_child_weight": hp.choice([1, 2, 3]),
                                }
            >>> auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                                name="auto_xgb_regressor",
                                                **config)
            >>> auto_xgb_reg.fit(data=(X_train, y_train),
                                 validation_data=(X_val, y_val),
                                 metric="rmse",
                                 n_sampling=1,
                                 search_space=search_space)
            >>> best_model = auto_xgb_reg.get_best_model()

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. The value will also be assigned
            to n_jobs, which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='regressor',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)

    def fit(self,
            data,
            epochs=1,
            validation_data=None,
            metric=None,
            metric_mode=None,
            metric_threshold=None,
            n_sampling=1,
            search_space=None,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            feature_cols=None,
            target_cols=None,
            ):
        """
        Automatically fit the model and search for the best hyperparameters.

        :param data: A Spark DataFrame, a tuple of ndarrays or a function.
            If data is a tuple of ndarrays, it should be in the form of (x, y),
            where x is training input data and y is training target data.
            If data is a function, it should takes config as argument and returns a tuple of
            ndarrays in the form of (x, y).
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
            If you have also set metric_threshold, a trial will stop if either it has been
            optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param validation_data: Validation data. Validation data type should be the same as data.
        :param metric: String or customized evaluation metric function.
            If string, metric is the evaluation metric name to optimize, e.g. "mse".
            If callable function, it signature should be func(y_true, y_pred), where y_true and
            y_pred are numpy ndarray. The function should return a float value as evaluation result.
        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.
            You have to specify metric_mode if you use a customized metric function.
            You don't have to specify metric_mode if you use the built-in metric in
            bigdl.orca.automl.metrics.Evaluator.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: Number of times to sample from the search_space. Defaults to 1.
            If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.
            If this is -1, (virtually) infinite samples are generated
            until a stopping condition is met.
        :param search_space: a dict for search space
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
            metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param target_cols: target column names if data is Spark DataFrame.
        """
        data, validation_data = _maybe_convert_spark_df_to_ndarray(data,
                                                                   validation_data,
                                                                   feature_cols,
                                                                   target_cols)
        super().fit(data=data,
                    epochs=epochs,
                    validation_data=validation_data,
                    metric=metric,
                    metric_mode=metric_mode,
                    metric_threshold=metric_threshold,
                    n_sampling=n_sampling,
                    search_space=search_space,
                    search_alg=search_alg,
                    search_alg_params=search_alg_params,
                    scheduler=scheduler,
                    scheduler_params=scheduler_params,
                    feature_cols=feature_cols,
                    target_cols=target_cols)


def _maybe_convert_spark_df_to_ndarray(data,
                                       validation_data,
                                       feature_cols,
                                       target_cols):
    def convert_df_to_ndarray(data, feature_cols, target_cols):
        df = data.toPandas()
        X = df[feature_cols]
        y = df[target_cols]
        arrays = (X, y)
        return arrays

    def check_cols(cols, cols_name):
        if not cols:
            raise ValueError(f"You must input valid {cols_name} for Spark DataFrame data input")
        if isinstance(cols, list):
            return cols
        if not isinstance(cols, str):
            raise ValueError(f"{cols_name} should be a string or a list of strings, "
                             f"but got {type(cols)}")
        return [cols]

    from pyspark.sql import DataFrame
    if isinstance(data, DataFrame):
        feature_cols = check_cols(feature_cols, cols_name="feature_cols")
        target_cols = check_cols(target_cols, cols_name="target_cols")
        train_arrays = convert_df_to_ndarray(data, feature_cols, target_cols)
        if validation_data:
            if not isinstance(validation_data, DataFrame):
                raise ValueError(f"data and validation_data should be both Spark DataFrame, "
                                 f"but got validation_data of type {type(data)}")
            val_arrays = convert_df_to_ndarray(validation_data, feature_cols, target_cols)
        else:
            val_arrays = None
        return train_arrays, val_arrays
    else:
        return data, validation_data
