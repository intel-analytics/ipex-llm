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
from zoo.chronos.model.Seq2Seq_pytorch import model_creator
from .base_automodel import BasePytorchAutomodel


class AutoSeq2Seq(BasePytorchAutomodel):
    def __init__(self,
                 input_feature_num,
                 output_target_num,
                 past_seq_len,
                 future_seq_len,
                 optimizer,
                 loss,
                 metric,
                 lr=0.001,
                 lstm_hidden_dim=128,
                 lstm_layer_num=2,
                 dropout=0.25,
                 teacher_forcing=False,
                 backend="torch",
                 logs_dir="/tmp/auto_seq2seq",
                 cpus_per_trial=1,
                 name="auto_seq2seq",
                 remote_dir=None,
                 ):
        """
        Create an AutoSeq2Seq.

        :param input_feature_num: Int. The number of features in the input
        :param output_target_num: Int. The number of targets in the output
        :param past_seq_len: Int. The number of historical steps used for forecasting.
        :param future_seq_len: Int. The number of future steps to forecast.
        :param optimizer: String or pyTorch optimizer creator function or
               tf.keras optimizer instance.
        :param loss: String or pytorch/tf.keras loss instance or pytorch loss creator function.
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param lr: float or hp sampling function from a float space. Learning rate.
               e.g. hp.choice([0.001, 0.003, 0.01])
        :param lstm_hidden_dim: LSTM hidden channel for decoder and encoder.
               hp.grid_search([32, 64, 128])
        :param lstm_layer_num: LSTM layer number for decoder and encoder.
               e.g. hp.randint(1, 4)
        :param dropout: float or hp sampling function from a float space. Learning rate. Dropout
               rate. e.g. hp.uniform(0.1, 0.3)
        :param teacher_forcing: If use teacher forcing in training. e.g. hp.choice([True, False])
        :param backend: The backend of the Seq2Seq model. We only support backend as "torch"
               for now.
        :param logs_dir: Local directory to save logs and results. It defaults to
               "/tmp/auto_seq2seq"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoSeq2Seq. It defaults to "auto_seq2seq"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".
        """
        super().__init__()
        # todo: support search for past_seq_len.
        # todo: add input check.
        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")
        self.search_space = dict(
            input_feature_num=input_feature_num,
            output_feature_num=output_target_num,
            past_seq_len=past_seq_len,
            future_seq_len=future_seq_len,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layer_num=lstm_layer_num,
            lr=lr,
            dropout=dropout,
            teacher_forcing=teacher_forcing
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
