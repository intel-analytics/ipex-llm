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

from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
nn = LazyImport('torch.nn')
TensorDataset = LazyImport('torch.utils.data')
DataLoader = LazyImport('torch.utils.data')

class CustomizedNet(nn.Module):
    def __init__(self,
                 dropout,
                 input_size,
                 input_feature_num,
                 hidden_dim,
                 output_size):
        '''
        Simply use linear layers for multi-variate single-step forecasting.
        '''
        super().__init__()
        self.fc1 = nn.Linear(input_size*input_feature_num, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x.shape = (num_sample, input_size, input_feature_num)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x.shape = (num_sample, output_size)
        x = torch.unsqueeze(x, 1)
        # x.shape = (num_sample, 1, output_size)
        return x
