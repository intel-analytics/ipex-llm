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


class AutoLSTM(BaseAutomodel):
    def __init__(self,
                 input_feature_num,
                 output_target_num,
                 past_seq_len,
                 optimizer,
                 loss,
                 metric,
                 metric_mode=None,
                 hidden_dim=32,
                 layer_num=1,
                 lr=0.001,
                 dropout=0.2,
                 backend="torch",
                 logs_dir="/tmp/auto_lstm",
                 cpus_per_trial=1,
                 name="auto_lstm",
                 remote_dir=None,
                 ):
        """
        Create an AutoLSTM.

        :param input_feature_num: Int. The number of features in the input
        :param output_target_num: Int. The number of targets in the output
        :param past_seq_len: Int or hp sampling function The number of historical
               steps used for forecasting.
        :param optimizer: String or pyTorch optimizer creator function or
               tf.keras optimizer instance.
        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.
        :param metric: String or customized evaluation metric function.
               If string, metric is the evaluation metric name to optimize, e.g. "mse".
               If callable function, it signature should be func(y_true, y_pred), where y_true and
               y_pred are numpy ndarray. The function should return a float value
               as evaluation result.
        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.
               You have to specify metric_mode if you use a customized metric function.
               You don't have to specify metric_mode if you use the built-in metric in
               bigdl.orca.automl.metrics.Evaluator.
        :param hidden_dim: Int or hp sampling function from an integer space. The number of features
               in the hidden state `h`. For hp sampling, see bigdl.chronos.orca.automl.hp for more
               details. e.g. hp.grid_search([32, 64]).
        :param layer_num: Int or hp sampling function from an integer space. Number of recurrent
               layers. e.g. hp.randint(1, 3)
        :param lr: float or hp sampling function from a float space. Learning rate.
               e.g. hp.choice([0.001, 0.003, 0.01])
        :param dropout: float or hp sampling function from a float space. Learning rate. Dropout
               rate. e.g. hp.uniform(0.1, 0.3)
        :param backend: The backend of the lstm model. support "keras" and "torch".
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_lstm"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoLSTM. It defaults to "auto_lstm"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
               defaults to None and doesn't take effects while running in local. While running in
               cluster, it defaults to "hdfs:///tmp/{name}".
        """
        # todo: support search for past_seq_len.
        self.search_space = dict(
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            lr=lr,
            dropout=dropout,
            input_feature_num=input_feature_num,
            output_feature_num=output_target_num,
            past_seq_len=past_seq_len,
            future_seq_len=1
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
            from bigdl.chronos.model.VanillaLSTM_pytorch import model_creator
        elif self.backend.startswith("keras"):
            from bigdl.chronos.model.tf2.VanillaLSTM_keras import model_creator
        else:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(False,
                              f"We only support keras and torch as backend,"
                              f" but got {self.backend}")
        self._model_creator = model_creator

        super().__init__()
