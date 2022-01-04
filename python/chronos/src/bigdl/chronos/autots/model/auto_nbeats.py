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
from .base_automodel import BasePytorchAutomodel
from bigdl.orca.automl.auto_estimator import AutoEstimator
from bigdl.chronos.model.nbeats_pytorch import model_creator
from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder


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
                 lr=0.01,
                 backend="torch",
                 logs_dir="/tmp/auto_nbeats",
                 cpus_per_trial=1,
                 name="auto_nbeats",
                 remote_dir=None):
        super().__init__()

        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")

        self.search_space = dict(stack_types,
                                 nb_blocks_per_stack,
                                 thetas_dim,
                                 share_weigets_in_stack,
                                 hidden_layer_units,
                                 nb_harmonics,
                                 past_seq_len,
                                 future_seq_len,
                                 lr,
                                 input_feature_num=1,
                                 output_feature_num=1)

        self.metric = metric
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss)
        self.auto_nbeats = AutoEstimator(model_builder=model_builder,
                                         logs_dir=logs_dir,
                                         resources_per_trial={'cpu': cpus_per_trial},
                                         remote_dir=remote_dir,
                                         name=name)



