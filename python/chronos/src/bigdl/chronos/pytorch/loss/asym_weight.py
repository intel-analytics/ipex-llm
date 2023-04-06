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
import torch.nn as nn
import torch.nn.functional as F

from bigdl.nano.utils.common import invalidInputError


class AsymWeightLoss(nn.Module):
    '''
    AsymWeightLoss is an asymmetric loss.
    '''
    def __init__(self, underestimation_penalty=1, L1=False):
        '''
        :param underestimation_penalty: when underestimation_penalty is set to 1, the loss is MSE,
               if set larger than 1, this loss panelize underestimate more and vice versa.
        :param L1: if use L1 loss rather than MSE(L2) loss.
        '''
        super().__init__()
        invalidInputError(underestimation_penalty > 0,
                          "underestimation_penalty should be larger than 0")
        self.L1 = L1
        self.underestimation_penalty = underestimation_penalty

    def forward(self, y_hat, y):
        if self.L1:
            loss = F.relu(y_hat - y) + F.relu(y - y_hat) * self.underestimation_penalty
        else:
            loss = torch.pow(F.relu(y_hat - y), 2) +\
                torch.pow(F.relu(y - y_hat), 2) * self.underestimation_penalty
        return torch.mean(loss)
