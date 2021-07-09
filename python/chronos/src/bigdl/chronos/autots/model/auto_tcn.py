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
from zoo.automl.model.base_pytorch_model import PytorchModelBuilder
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.chronos.model.tcn import model_creator


class AutoTCN:
    def __init__(self,
                 input_feature_num,
                 output_target_num,
                 past_seq_len,
                 future_seq_len,
                 optimizer,
                 loss,
                 metric,
                 hidden_units=None,
                 levels=None,
                 num_channels=None,
                 kernel_size=7,
                 lr=0.001,
                 dropout=0.2,
                 backend="torch",
                 logs_dir="/tmp/auto_tcn",
                 cpus_per_trial=1,
                 name="auto_tcn"):
        """
        Create an AutoTCN.

        :param input_feature_num: Int. The number of features in the input
        :param output_target_num: Int. The number of targets in the output
        :param past_seq_len: Int. The number of historical steps used for forecasting.
        :param future_seq_len: Int. The number of future steps to forecast.
        :param optimizer: String or pyTorch optimizer creator function or
               tf.keras optimizer instance.
        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param hidden_units: Int or hp sampling function from an integer space. The number of hidden
               units or filters for each convolutional layer. It is similar to `units` for LSTM.
               It defaults to 30. We will omit the hidden_units value if num_channels is specified.
               For hp sampling, see zoo.orca.automl.hp for more details.
               e.g. hp.grid_search([32, 64]).
        :param levels: Int or hp sampling function from an integer space. The number of levels of
               TemporalBlocks to use. It defaults to 8. We will omit the levels value if
               num_channels is specified.
        :param num_channels: List of integers. A list of hidden_units for each level. You could
               specify num_channels if you want different hidden_units for different levels.
               By default, num_channels equals to
               [hidden_units] * (levels - 1) + [output_target_num].
        :param kernel_size: Int or hp sampling function from an integer space.
               The size of the kernel to use in each convolutional layer.
        :param lr: float or hp sampling function from a float space. Learning rate.
               e.g. hp.choice([0.001, 0.003, 0.01])
        :param dropout: float or hp sampling function from a float space. Learning rate. Dropout
               rate. e.g. hp.uniform(0.1, 0.3)
        :param backend: The backend of the TCN model. We only support backend as "torch" for now.
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_tcn"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoTCN. It defaults to "auto_tcn"
        """
        # todo: support search for past_seq_len.
        # todo: add input check.
        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")
        self.search_space = dict(
            input_feature_num=input_feature_num,
            output_feature_num=output_target_num,
            past_seq_len=past_seq_len,
            future_seq_len=future_seq_len,
            nhid=hidden_units,
            levels=levels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            lr=lr,
            dropout=dropout,
        )
        self.metric = metric
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss,
                                            )
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={"cpu": cpus_per_trial},
                                      name=name)

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        """
        Automatically fit the model and search for the best hyper parameters.

        :param data: train data.
               For backend of "torch", data can be a tuple of ndarrays or a function that takes a
               config dictionary as parameter and returns a PyTorch DataLoader.
               For backend of "keras", data can be a tuple of ndarrays.
               If data is a tuple of ndarrays, it should be in the form of (x, y),
                where x is training input data and y is training target data.
        :param epochs: Max number of epochs to train in each trial. Defaults to 1.
               If you have also set metric_threshold, a trial will stop if either it has been
               optimized to the metric_threshold or it has been trained for {epochs} epochs.
        :param batch_size: Int or hp sampling function from an integer space. Training batch size.
               It defaults to 32.
        :param validation_data: Validation data. Validation data type should be the same as data.
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

        :return:
        """
        self.search_space["batch_size"] = batch_size
        self.auto_est.fit(
            data=data,
            epochs=epochs,
            validation_data=validation_data,
            metric=self.metric,
            metric_threshold=metric_threshold,
            n_sampling=n_sampling,
            search_space=self.search_space,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
        )

    def get_best_model(self):
        """
        Get the best tcn model.
        """
        return self.auto_est.get_best_model()

    def get_best_config(self):
        """
        Get the best configuration

        :return: A dictionary of best hyper parameters
        """
        return self.auto_est.get_best_config()

    def _get_best_automl_model(self):
        return self.auto_est._get_best_automl_model()
