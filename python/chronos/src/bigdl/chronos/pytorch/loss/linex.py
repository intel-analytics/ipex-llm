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

from bigdl.nano.utils.log4Error import invalidInputError


class LinexLoss(nn.Module):
    '''
    LinexLoss is an asymmetric loss talked about in
    https://www.scirp.org/journal/paperinformation.aspx?paperid=97986

    '''
    def __init__(self, a=1):
        '''
        :param a: when a is set to ~0, this loss is similar to MSE.
               when a is set to > 0, this loss panelize underestimate more.
               when a is set to < 0, this loss panelize overestimate more.
        '''
        super().__init__()
        invalidInputError(a != 0, "a should not be set to 0")
        self.a = a
        self.b = 2/(a**2)

    def forward(self, y_hat, y):
        delta = y - y_hat
        loss = self.b * (torch.exp(self.a*delta) - self.a*delta - 1)
        return torch.mean(loss)
