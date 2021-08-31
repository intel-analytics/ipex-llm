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
from zoo.chronos.model.arima import ARIMABuilder


# -

class AutoARIMA:

    def __init__(self,
                 p=2,
                 q=2,
                 seasonal=True,
                 P=1,
                 Q=1,
                 m=7,
                 metric='mse',
                 logs_dir="/tmp/auto_arima_logs",
                 cpus_per_trial=1,
                 name="auto_arima",
                 remote_dir=None,
                 **arima_config
                 ):
        """
        Create an automated ARIMA Model.
        User need to specify either the exact value or the search space of
        the ARIMA model hyperparameters. For details of the ARIMA model hyperparameters, refer to
        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.

        :param p: Int or hp sampling function from an integer space for hyperparameter p
               of the ARIMA model.
               For hp sampling, see zoo.chronos.orca.automl.hp for more details.
               e.g. hp.randint(0, 3).
        :param q: Int or hp sampling function from an integer space for hyperparameter q
               of the ARIMA model.
               e.g. hp.randint(0, 3).
        :param seasonal: Bool or hp sampling function from an integer space for whether to add
               seasonal components to the ARIMA model.
               e.g. hp.choice([True, False]).
        :param P: Int or hp sampling function from an integer space for hyperparameter P
               of the ARIMA model.
               For hp sampling, see zoo.chronos.orca.automl.hp for more details.
               e.g. hp.randint(0, 3).
        :param Q: Int or hp sampling function from an integer space for hyperparameter Q
               of the ARIMA model.
               e.g. hp.randint(0, 3).
        :param m: Int or hp sampling function from an integer space for hyperparameter p
               of the ARIMA model.
               e.g. hp.choice([4, 7, 12, 24, 365]).
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param logs_dir: Local directory to save logs and results. It defaults to
               "/tmp/auto_arima_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoARIMA. It defaults to "auto_arima"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
        :param arima_config: Other ARIMA hyperparameters.

        """
        self.search_space = {
            "p": p,
            "q": q,
            "seasonal": seasonal,
            "P": P,
            "Q": Q,
            "m": m,
        }
        self.metric = metric
        model_builder = ARIMABuilder()
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={
                                          "cpu": cpus_per_trial},
                                      remote_dir=remote_dir,
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
        Get the best ARIMA model.
        """
        return self.auto_est.get_best_model()
