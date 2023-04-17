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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union
if TYPE_CHECKING:
    from ray.tune.sample import Float, Function, Integer
    from functools import partial
    from pyspark.sql import DataFrame
    from numpy import ndarray


class AutoXGBClassifier(AutoEstimator):
    def __init__(self,
                 logs_dir: str="/tmp/auto_xgb_classifier_logs",
                 cpus_per_trial: int=1,
                 name: Optional[str]=None,
                 remote_dir: Optional[str]=None,
                 **xgb_configs
                 ) -> None:
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
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
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
                         remote_dir=remote_dir,
                         name=name)

    def fit(self,
            data: Union["partial", Tuple["ndarray", "ndarray"], "DataFrame"],
            epochs: int=1,
            validation_data: Optional[Union["partial",
                                      Tuple["ndarray", "ndarray"], "DataFrame"]]=None,
            metric: Optional[Union[Callable, str]]=None,
            metric_mode: Optional[str]=None,
            metric_threshold: Optional[Union[int, float]]=None,
            n_sampling: int=1,
            search_space: Optional[Dict]=None,
            search_alg: Optional[str]=None,
            search_alg_params: Optional[Dict]=None,
            scheduler: Optional[str]=None,
            scheduler_params: Optional[Dict]=None,
            feature_cols: Optional[List[str]]=None,
            label_cols: Optional[List[str]]=None,
            ) -> None:
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
        :param label_cols: target column names if data is Spark DataFrame.
        """
        data, validation_data, feature_cols, label_cols = _merge_cols_for_spark_df(data,
                                                                                   validation_data,
                                                                                   feature_cols,
                                                                                   label_cols)

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
                    label_cols=label_cols)


class AutoXGBRegressor(AutoEstimator):
    def __init__(self,
                 logs_dir: str="/tmp/auto_xgb_regressor_logs",
                 cpus_per_trial: int=1,
                 name: Optional[str]=None,
                 remote_dir: Optional[str]=None,
                 **xgb_configs
                 ) -> None:
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
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
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
                         remote_dir=remote_dir,
                         name=name)

    def fit(self,
            data: Union["partial", Tuple["ndarray", "ndarray"], "DataFrame"],
            epochs: int=1,
            validation_data: Optional[Union["partial",
                                            Tuple["ndarray", "ndarray"], "DataFrame"]]=None,
            metric: Optional[Union[Callable, str]]=None,
            metric_mode: Optional[str]=None,
            metric_threshold: Optional[Union["float", "int"]]=None,
            n_sampling: int=1,
            search_space: Optional[Dict]=None,
            search_alg: Optional[str]=None,
            search_alg_params: Optional[Dict]=None,
            scheduler: Optional[str]=None,
            scheduler_params: Optional[Dict]=None,
            feature_cols: Optional[List[str]]=None,
            label_cols: Optional[List[str]]=None,
            ) -> None:
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
        :param label_cols: target column names if data is Spark DataFrame.
        """
        data, validation_data, feature_cols, label_cols = _merge_cols_for_spark_df(data,
                                                                                   validation_data,
                                                                                   feature_cols,
                                                                                   label_cols)

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
                    label_cols=label_cols)


def _merge_cols_for_spark_df(data: Union["DataFrame", Tuple["ndarray", "ndarray"], "partial"],
                             validation_data: Optional[Union["partial",
                                                       Tuple["ndarray", "ndarray"], "DataFrame"]],
                             feature_cols: Optional[List[str]],
                             label_cols: Optional[List[str]]) -> Tuple:
    # merge feature_cols/label_cols to one column, to adapt to the meanings of feature_cols and
    # label_cols in AutoEstimator, which correspond to the model inputs/outputs.
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import array

    def concat_cols(data, feature_cols, label_cols):
        combined_feature_name = "combined_features"
        combined_target_name = "combined_targets"
        data = data.select(array(*feature_cols).alias(combined_feature_name),
                           array(*label_cols).alias(combined_target_name))
        return data, combined_feature_name, combined_target_name

    feature_cols, label_cols = AutoEstimator._check_spark_dataframe_input(data,
                                                                          validation_data,
                                                                          feature_cols,
                                                                          label_cols)
    if isinstance(data, DataFrame):
        data, combined_feature_name, combined_target_name = concat_cols(data,
                                                                        feature_cols,
                                                                        label_cols)
        if validation_data is not None:
            validation_data, _, _ = concat_cols(validation_data, feature_cols, label_cols)
        feature_cols = [combined_feature_name]
        label_cols = [combined_target_name]
    return data, validation_data, feature_cols, label_cols
