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
# from bigdl.nano.pytorch.optim import LazyAdam
from lazyadam import LazyAdam
import torch
from torch import nn
from torch.nn import Embedding, Sequential, MSELoss


class Example18(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sparse=False):
        super(Example18, self).__init__()
        self.model = Sequential()
        self.embedding = Embedding(vocab_size, embedding_dim, sparse=sparse)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        return self.model(embeds)


def test_optim_lazyadam():
    model = Example18(30, 16)
    loss_function = MSELoss()
    optimizer = LazyAdam(model.parameters())

    N_EPOCHES = 1
    input = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    labels_np = np.random.randint(30, size=model(input).shape)
    labels = torch.FloatTensor(labels_np)

    for epoch in range(N_EPOCHES):
        for i in range(2):
            x = input[i]
            y = labels[i]

            model.train()

            total_loss = torch.Tensor([0])
            model.zero_grad()
            result = model(x)
            loss = loss_function(result, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            print("first:", model.state_dict()['embedding.weight'][0:10, :])
    # before_weights = model.get_weights()
    # model.fit(input_array, labels, epochs=4, batch_size=2)
    # after_weights = model.get_weights()
    #
    # # check if only the 0-10 cows of weights
    # # have changed after training
    # assert ((before_weights[0][0:10, :] != after_weights[0][0:10, :]).all())
    # assert ((before_weights[0][10:30, :] == after_weights[0][10:30, :]).all())


test_optim_lazyadam()
