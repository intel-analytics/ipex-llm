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


class AutoNbeats(BasePytorchAutomodel):
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 stack_types,
                 nb_blocks_per_stack,
                 thetas_dim,
                 share_weigets_in_stack,
                 hidden_layer_units,
                 nb_harmonics,
                 optimizer,
                 loss,
                 metric,
                 input_feature_num=1,
                 output_target_num=1,
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
        :param stack_types: Specifies the architecture of the stack,
               This value defaults to ("generic", "generic").
        :param nb_blocks_per_stack: Blocks contained in the stack,
               This value defaults to 3.
        :param thetas_dim: Number of fully connected layers
               with ReLu activation per block.
        :param share_weigets_in_stack: Shared weights between stacks,
               This value defaults to False.
        :param hidden_layer_units: The number of layers in a fully
               connected neural network, This values defaults to 256.
        :param nb_harmonics: Number of fully connected layers
               with ReLu activation per block.
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
        """
        super().__init__()

        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")

        if isinstance(input_feature_num, int) and input_feature_num != 1:
            raise ValueError(f"NBeat only supports univariate prediction.")

        self.search_space = dict(share_weigets_in_stack=share_weigets_in_stack,
                                 hidden_layer_units=hidden_layer_units,
                                 stack_types=stack_types,
                                 thetas_dim=thetas_dim,
                                 nb_blocks_per_stack=nb_blocks_per_stack,
                                 nb_harmonics=nb_harmonics,
                                 lr=lr,
                                 dropout=dropout,
                                 past_seq_len=past_seq_len,
                                 future_seq_len=future_seq_len,
                                 input_feature_num=input_feature_num,
                                 output_feature_num=output_target_num)

        self.metric = metric
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss)
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={'cpu': cpus_per_trial},
                                      remote_dir=remote_dir,
                                      name=name)
