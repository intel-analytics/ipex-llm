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
from zoo.orca.automl.model.base_pytorch_model import PytorchModelBuilder
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.chronos.model.VanillaLSTM_pytorch import model_creator
from .base_automodel import BasePytorchAutomodel


class AutoLSTM(BasePytorchAutomodel):
    def __init__(self,
                 input_feature_num,
                 output_target_num,
                 past_seq_len,
                 optimizer,
                 loss,
                 metric,
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
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param hidden_dim: Int or hp sampling function from an integer space. The number of features
               in the hidden state `h`. For hp sampling, see zoo.chronos.orca.automl.hp for more
               details. e.g. hp.grid_search([32, 64]).
        :param layer_num: Int or hp sampling function from an integer space. Number of recurrent
               layers. e.g. hp.randint(1, 3)
        :param lr: float or hp sampling function from a float space. Learning rate.
               e.g. hp.choice([0.001, 0.003, 0.01])
        :param dropout: float or hp sampling function from a float space. Learning rate. Dropout
               rate. e.g. hp.uniform(0.1, 0.3)
        :param backend: The backend of the lstm model. We only support backend as "torch" for now.
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_lstm"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoLSTM. It defaults to "auto_lstm"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
        """
        super().__init__()
        # todo: support backend = 'keras'
        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")
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
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss,
                                            )
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={"cpu": cpus_per_trial},
                                      remote_dir=remote_dir,
                                      name=name)
