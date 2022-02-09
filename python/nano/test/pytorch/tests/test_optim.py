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
# ==============================================================================

import numpy as np
from bigdl.nano.pytorch.optim import LazyAdam
import torch
from torch import nn


class Example18(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sparse=False):
        super(Example18, self).__init__()
        self.model = nn.Sequential()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        return self.model(embeds)


def test_optim_lazyadam():
    # # Test different results of LazyAdam
    # # when applied in sparse= False and sparse=True
    # model1: sparse=False
    # model2: sparse=True
    model1 = Example18(30, 16, sparse=False)
    model2 = Example18(30, 16, sparse=True)

    loss_function = nn.MSELoss()

    optimizer1 = LazyAdam(model1.parameters())
    optimizer2 = LazyAdam(model2.parameters())

    nums_epochs = 1
    nums_batches = 2
    input = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    labels_np = np.random.randint(30, size=model1(input).shape)
    labels = torch.FloatTensor(labels_np)

    sparse_dic = {}

    for epoch in range(nums_epochs):
        for i in range(nums_batches):
            x = input[i]
            y = labels[i]

            model1.train()
            total_loss = torch.Tensor([0])
            model1.zero_grad()
            result = model1(x)
            loss = loss_function(result, y)
            loss.backward()
            optimizer1.step()
            total_loss += loss.data

            sparse_dic["result{0}".format(i)] = model1.state_dict()['embedding.weight'][0:5, :].clone()

    for epoch in range(nums_epochs):
        for i in range(2):
            x = input[i]
            y = labels[i]

            model2.train()
            total_loss = torch.Tensor([0])
            model2.zero_grad()
            result = model2(x)
            loss = loss_function(result, y)
            loss.backward()
            optimizer2.step()
            total_loss += loss.data

            sparse_dic["result{0}".format(i + 2)] = model2.state_dict()['embedding.weight'][0:5, :].clone()

    # check if the 0-5 cows of model1 weights have changed after training
    # check if the 0-5 cows of model2 weights remain unchanged after training
    assert ((sparse_dic['result0'] != sparse_dic['result1']).all())
    assert ((sparse_dic['result2'] == sparse_dic['result3']).all())

