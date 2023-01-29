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

import torch.nn as nn


class NormalizeTSModel(nn.Module):
    def __init__(self, model, output_feature_dim):
        """
        Build a Normalization model wrapper.

        param model: basic forecaster model.
        :param output_feature_dim: Specify the output dimension.
        """
        super(NormalizeTSModel, self).__init__()
        self.model = model
        self.output_feature_dim = output_feature_dim

    def forward(self, x):
        seq_last = x[:, -1:, :]
        x = x - seq_last
        y = self.model(x)
        y = y + seq_last[:, :, :self.output_feature_dim]
        return y
