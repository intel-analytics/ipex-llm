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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from bigdl.chronos.forecaster.base_forecaster import BasePytorchForecaster
from bigdl.chronos.model.nbeats_pytorch import model_creator, loss_creator, optimizer_creator


class NBeatsForecaster(BasePytorchForecaster):
    """
        Example:
            >>> # NBeatsForecaster test.
            >>> forecaster = NBeatsForecaster(paste_seq_len=10,
                                              future_seq_len=1,
                                              input_feature_num=1,
                                              output_feature_num=1,
                                              ...)
            >>> forecaster.fit((x_train, y_train))
            >>> forecaster.to_local() # if you set distributed=True
    """
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 stack_types=("generic", "generic"),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weigets_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.01,
                 metircs=["mse"],
                 seed=None,
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="torch_distributed"):
        """
        Build a NBeats Forecaster Model.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param stack_types: pass
        :param nb_blocks_per_stack: pass
        :param thetas_dim: pass
        :param share_weigets_in_stack: pass
        :param hidden_layer_units: pass
        :param nb_harmonics: pass
        """

        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": 1,  # nbeats only support input single feature.
            "output_feature_num": 1,
        }

        self.model_config = {
            "stack_types": stack_types,
            "nb_blocks_per_stack": nb_blocks_per_stack,
            "thetas_dim": thetas_dim,
            "share_weigets_in_stack": share_weigets_in_stack,
            "hidden_layer_units": hidden_layer_units,
            "nb_harmonics": nb_harmonics
        }

        self.loss_config = {
            "loss": loss
        }
        
        self.optim_config = {
            "lr": lr,
            "optim": optimizer
        }

        # model creator settings
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator

        # distributed settings
        self.distributed = distributed
        self.distributed_backend = distributed_backend
        self.workers_per_node = workers_per_node

        # other settings
        self.lr = lr
        self.seed = seed
        self.metrics = metircs
        
        # nano settings
        current_num_threads = torch.get_num_threads()
        self.num_processes = max(1, current_num_threads//8) # 8 is a magic num
        self.use_ipex = False
        self.onnx_available = True
        self.checkpoint_callback = False

        super().__init__()
