# +
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.chronos.model.prophet import ProphetBuilder


# -

class AutoProphet:

    def __init__(self,
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 seasonality_mode='additive',
                 changepoint_range=0.8,
                 metric='mse',
                 logs_dir="/tmp/auto_prophet_logs",
                 cpus_per_trial=1,
                 name="auto_prophet",
                 **prophet_config
                 ):
        """
        Create an automated Prophet Model.
        User need to specify either the exact value or the search space of the
        Prophet model hyperparameters. For details of the Prophet model hyperparameters, refer to
        https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning.

        :param changepoint_prior_scale: Int or hp sampling function from an integer space
            for hyperparameter changepoint_prior_scale for the Prophet model.
            For hp sampling, see zoo.chronos.orca.automl.hp for more details.
            e.g. hp.loguniform(0.001, 0.5).
        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the
            Prophet model.
            e.g. hp.loguniform(0.01, 10).
        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the
            Prophet model.
            e.g. hp.loguniform(0.01, 10).
        :param seasonality_mode: hyperparameter seasonality_mode for the
            Prophet model.
            e.g. hp.choice(['additive', 'multiplicative']).
        :param changepoint_range: hyperparameter changepoint_range for the
            Prophet model.
            e.g. hp.uniform(0.8, 0.95).
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_prophet_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoProphet. It defaults to "auto_prophet"
        :param prophet_config: Other Prophet hyperparameters.
        """
        self.search_space = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": 'additive',
            "changepoint_range": changepoint_range,
        }
        self.metric = metric
        model_builder = ProphetBuilder()
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={"cpu": cpus_per_trial},
                                      name=name)

    def fit(self,
            data,
            epochs=1,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        """
        Automatically fit the model and search for the best hyperparameters.

        :param data: Training data, A 1-D numpy array.
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
               If you have also set metric_threshold, a trial will stop if either it has been
               optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param validation_data: Validation data. A 1-D numpy array.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: Number of times to sample from the search_space. Defaults to 1.
               If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.
               If this is -1, (virtually) infinite samples are generated
               until a stopping condition is met.
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
               metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        """
        self.auto_est.fit(data=data,
                          validation_data=validation_data,
                          metric=self.metric,
                          metric_threshold=metric_threshold,
                          n_sampling=n_sampling,
                          search_space=self.search_space,
                          search_alg=search_alg,
                          search_alg_params=search_alg_params,
                          scheduler=scheduler,
                          scheduler_params=scheduler_params
                          )

    def get_best_model(self):
        """
        Get the best Prophet model.
        """
        return self.auto_est.get_best_model()
