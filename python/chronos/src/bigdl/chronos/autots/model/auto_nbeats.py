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

from bigdl.orca.automl.auto_estimator import AutoEstimator
from bigdl.chronos.model.nbeats_pytorch import model_creator
from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder

from .base_automodel import BasePytorchAutomodel


class AutoNBeats(BasePytorchAutomodel):
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 stack_types,
                 nb_blocks_per_stack,
                 thetas_dim,
                 share_weights_in_stack,
                 hidden_layer_units,
                 nb_harmonics,
                 optimizer,
                 loss,
                 metric,
                 lr=0.001,
                 dropout=0.2,
                 backend="torch",
                 logs_dir="/tmp/auto_nbeats",
                 cpus_per_trial=1,
                 name="auto_nbeats",
                 remote_dir=None):
        """
        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param stack_types: Specifies the type of stack,
               including "generic", "trend", "seasnoality".
               This value defaults to ("generic", "generic").
        :param nb_blocks_per_stack:  Specify the number of blocks
               contained in each stack, This value defaults to 3.
        :param thetas_dim: Expansion Coefficients of Multilayer FC Networks.
               if type is "generic", Extended length factor, if type is "trend"
               then polynomial coefficients, if type is "seasonality"
               expressed as a change within each step.
        :param share_weights_in_stack: Share block weights for each stack.,
               This value defaults to False.
        :param hidden_layer_units: Number of fully connected layers with per block.
               This values defaults to 256.
        :param nb_harmonics: Only available in "seasonality" type,
               specifies the time step of backward, This value defaults is None.
        :param dropout: Specify the dropout close possibility
               (i.e. the close possibility to a neuron). This value defaults to 0.001.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: Specify the loss function used for training. This value
               defaults to "mse". You can choose from "mse", "mae" and
               "huber_loss".
        :param metric: A list contains metrics for evaluating the quality of
               forecasting. You may only choose from "mse" and "mae" for a
               distributed forecaster. You may choose from "mse", "mae",
               "rmse", "r2", "mape", "smape", for a non-distributed forecaster.
        :param backend: The backend of the nbeats model. We only support backend as "torch" for now.
        :param logs_dir: Local directory to save logs and results. It defaults to "/tmp/auto_nbeats"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoNBeats. It defaults to "auto_nbeats"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
               defaults to None and doesn't take effects while running in local. While running in
               cluster, it defaults to "hdfs:///tmp/{name}".
        """
        super().__init__()

        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")

        self.search_space = dict(share_weights_in_stack=share_weights_in_stack,
                                 hidden_layer_units=hidden_layer_units,
                                 stack_types=stack_types,
                                 thetas_dim=thetas_dim,
                                 nb_blocks_per_stack=nb_blocks_per_stack,
                                 nb_harmonics=nb_harmonics,
                                 lr=lr,
                                 dropout=dropout,
                                 past_seq_len=past_seq_len,
                                 future_seq_len=future_seq_len)

        self.metric = metric
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss)
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={'cpu': cpus_per_trial},
                                      remote_dir=remote_dir,
                                      name=name)
