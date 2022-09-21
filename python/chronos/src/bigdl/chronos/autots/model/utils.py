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
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

input_feature_dim = 10
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1


def get_x_y(size):
    x = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return x.astype(np.float32), y.astype(np.float32)

class RandomDataset(Dataset):
    def __init__(self, size=1000):
        x, y = get_x_y(size)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_dataloader_creator(config):
    return DataLoader(RandomDataset(size=1000),
                      batch_size=config["batch_size"],
                      shuffle=True)

def valid_dataloader_creator(config):
    return DataLoader(RandomDataset(size=400),
                      batch_size=config["batch_size"],
                      shuffle=True)

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

def model_creator_pytorch(config):
    '''
    Pytorch customized model creator
    '''
    return CustomizedNet(dropout=config["dropout"],
                         input_size=config["past_seq_len"],
                         input_feature_num=config["input_feature_num"],
                         hidden_dim=config["hidden_dim"],
                         output_size=config["output_feature_num"])
