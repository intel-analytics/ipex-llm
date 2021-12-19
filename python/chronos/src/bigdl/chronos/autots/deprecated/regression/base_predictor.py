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
import pandas as pd
import os
from abc import abstractmethod

from bigdl.orca.automl.metrics import Evaluator
from bigdl.chronos.autots.deprecated.pipeline.time_sequence import TimeSequencePipeline
from bigdl.orca.automl.search.utils import process
from bigdl.chronos.autots.deprecated.config.recipe import *
from bigdl.orca.ray import RayContext
from bigdl.orca.automl.auto_estimator import AutoEstimator


ALLOWED_FIT_METRICS = ("mse", "mae", "r2")


class BasePredictor(object):

    def __init__(self,
                 name="automl",
                 logs_dir="~/bigdl_automl_logs",
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
                 ):

        self.logs_dir = logs_dir
        self.name = name
        self.search_alg = search_alg
        self.search_alg_params = search_alg_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    @abstractmethod
    def get_model_builder(self):
        raise NotImplementedError

    def _check_df(self, df):
        assert isinstance(df, pd.DataFrame) and df.empty is False, \
            "You should input a valid data frame"

    @staticmethod
    def _check_fit_metric(metric):
        if metric not in ALLOWED_FIT_METRICS:
            raise ValueError(f"metric {metric} is not supported for fit. "
                             f"Input metric should be among {ALLOWED_FIT_METRICS}")

    def fit(self,
            input_df,
            validation_df=None,
            metric="mse",
            recipe=SmokeRecipe(),
            mc=False,
            resources_per_trial={"cpu": 2},
            upload_dir=None,
            ):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are
                       "mean_squared_error" or "r_square"
        :param recipe: a Recipe object. Various recipes covers different search space and stopping
                      criteria. Default is SmokeRecipe().
        :param resources_per_trial: Machine resources to allocate per trial,
            e.g. ``{"cpu": 64, "gpu": 8}`
        :param upload_dir: Optional URI to sync training results and checkpoints. We only support
            hdfs URI for now. It defaults to
            "hdfs:///user/{hadoop_user_name}/ray_checkpoints/{predictor_name}".
            Where hadoop_user_name is specified in init_orca_context or init_spark_on_yarn,
            which defaults to "root". predictor_name is the name used in predictor instantiation.
        )
        :return: a pipeline constructed with the best model and configs.
        """
        self._check_df(input_df)
        if validation_df is not None:
            self._check_df(validation_df)

        ray_ctx = RayContext.get()
        is_local = ray_ctx.is_local
        # BasePredictor._check_fit_metric(metric)
        if not is_local:
            if not upload_dir:
                hadoop_user_name = os.getenv("HADOOP_USER_NAME")
                upload_dir = os.path.join(os.sep, "user", hadoop_user_name,
                                          "ray_checkpoints", self.name)
            cmd = "hadoop fs -mkdir -p {}".format(upload_dir)
            process(cmd)
        else:
            upload_dir = None

        self.pipeline = self._hp_search(
            input_df,
            validation_df=validation_df,
            metric=metric,
            recipe=recipe,
            mc=mc,
            resources_per_trial=resources_per_trial,
            remote_dir=upload_dir)
        return self.pipeline

    def evaluate(self,
                 input_df,
                 metric=None
                 ):
        """
        Evaluate the model on a list of metrics.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param metric: A list of Strings Available string values are "mean_squared_error",
                      "r_square".
        :return: a list of metric evaluation results.
        """
        Evaluator.check_metric(metric)
        return self.pipeline.evaluate(input_df, metric)

    def predict(self,
                input_df):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of
            the past sequence.
            values are the predicted future sequence values.
            Example :
            datetime    value_0     value_1   ...     value_2
            2019-01-03  2           3                   9
        """
        return self.pipeline.predict(input_df)

    def _detach_recipe(self, recipe):
        self.search_space = recipe.search_space()

        stop = recipe.runtime_params()
        self.metric_threshold = None
        if "reward_metric" in stop.keys():
            self.mode = Evaluator.get_metric_mode(self.metric)
            self.metric_threshold = -stop["reward_metric"] if \
                self.mode == "min" else stop["reward_metric"]
        self.epochs = stop["training_iteration"]
        self.num_samples = stop["num_samples"]

    def _hp_search(self,
                   input_df,
                   validation_df,
                   metric,
                   recipe,
                   mc,
                   resources_per_trial,
                   remote_dir):

        model_builder = self.get_model_builder()

        self.metric = metric
        self._detach_recipe(recipe)

        # prepare parameters for search engine
        auto_est = AutoEstimator(model_builder,
                                 logs_dir=self.logs_dir,
                                 resources_per_trial=resources_per_trial,
                                 name=self.name,
                                 remote_dir=remote_dir)
        auto_est.fit(data=input_df,
                     validation_data=validation_df,
                     search_space=self.search_space,
                     n_sampling=self.num_samples,
                     epochs=self.epochs,
                     metric_threshold=self.metric_threshold,
                     search_alg=self.search_alg,
                     search_alg_params=self.search_alg_params,
                     scheduler=self.scheduler,
                     scheduler_params=self.scheduler_params,
                     metric=metric)

        best_model = auto_est._get_best_automl_model()
        pipeline = TimeSequencePipeline(name=self.name, model=best_model)
        return pipeline
