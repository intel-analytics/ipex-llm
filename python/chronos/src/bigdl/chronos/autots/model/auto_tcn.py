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

from .base_automodel import BaseAutomodel


class AutoTCN(BaseAutomodel):
    def __init__(self,
                 input_feature_num,
                 output_target_num,
                 past_seq_len,
                 future_seq_len,
                 optimizer,
                 loss,
                 metric,
                 metric_mode=None,
                 hidden_units=None,
                 levels=None,
                 num_channels=None,
                 kernel_size=7,
                 lr=0.001,
                 dropout=0.2,
                 backend="torch",
                 logs_dir="/tmp/auto_tcn",
                 cpus_per_trial=1,
                 name="auto_tcn",
                 remote_dir=None,
                 ):
        """
        Create an AutoTCN.

        :param input_feature_num: Int. The number of features in the input
        :param output_target_num: Int. The number of targets in the output
        :param past_seq_len: Int. The number of historical steps used for forecasting.
        :param future_seq_len: Int. The number of future steps to forecast.
        :param optimizer: String or pyTorch optimizer creator function or
               tf.keras optimizer instance.
        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.
        :param metric: String or customized evaluation metric function.
            If string, metric is the evaluation metric name to optimize, e.g. "mse".
            If callable function, it signature should be func(y_true, y_pred), where y_true and
            y_pred are numpy ndarray. The function should return a float value as evaluation result.
        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.
            You have to specify metric_mode if you use a customized metric function.
            You don't have to specify metric_mode if you use the built-in metric in
            bigdl.orca.automl.metrics.Evaluator.
        :param hidden_units: Int or hp sampling function from an integer space. The number of hidden
               units or filters for each convolutional layer. It is similar to `units` for LSTM.
               It defaults to 30. We will omit the hidden_units value if num_channels is specified.
               For hp sampling, see bigdl.orca.automl.hp for more details.
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
        :param backend: The backend of the TCN model. support "keras" and "torch".
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_tcn"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoTCN. It defaults to "auto_tcn"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
        """

        # todo: support search for past_seq_len.
        # todo: add input check.
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
        self.metric_mode = metric_mode
        self.backend = backend
        self.optimizer = optimizer
        self.loss = loss

        self._auto_est_config = dict(logs_dir=logs_dir,
                                     resources_per_trial={"cpu": cpus_per_trial},
                                     remote_dir=remote_dir,
                                     name=name)
        if self.backend.startswith("torch"):
            from bigdl.chronos.model.tcn import model_creator
        elif self.backend.startswith("keras"):
            from bigdl.chronos.model.tf2.TCN_keras import model_creator
        else:
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(False,
                              f"We only support keras and torch as backend,"
                              f" but got {self.backend}")
        self._model_creator = model_creator
        super().__init__()
