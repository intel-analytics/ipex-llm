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
# ==============================================================================
# Most of the pytorch code is adapted from guoyang9's NCF implementation for
# ml-1m dataset.
# https://github.com/guoyang9/NCF
#

import numpy as np
import pandas as pd
from bigdl.dllib.utils.log4Error import *

# Step 1: Init Orca Context

from bigdl.orca import init_orca_context, stop_orca_context
init_orca_context()

# Step 2: Define Train Dataset

from sklearn.model_selection import train_test_split
import torch.utils.data as data


class NCFData(data.Dataset):
    def __init__(self, features,
                 item_num=0, train_mat=None, num_ng=4, is_training=False):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.item_num = item_num
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        invalidInputError(self.is_training, 'no need to sampling when testing')

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.item_num)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.item_num)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        if self.is_training:
            return (self.num_ng + 1) * len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = float(labels[idx])
        return user, item, label


def train_loader_func(config, batch_size):
    data_X = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    data_X = data_X.values.tolist()

    # load ratings as a dok matrix
    import scipy.sparse as sp
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for x in data_X:
        train_mat[x[0], x[1]] = 1

    train_data, _ = train_test_split(data_X, test_size=0.2, random_state=100)

    train_dataset = NCFData(train_data, item_num=item_num, train_mat=train_mat,
                            num_ng=4, is_training=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader.dataset.ng_sample()  # sample negative items for training datasets
    return train_loader


def test_loader_func(config, batch_size):
    data_X = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})

    data_X = data_X.values.tolist()

    _, test_data = train_test_split(data_X, test_size=0.2, random_state=100)

    test_dataset = NCFData(test_data)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

# Step 3: Define the Model

from model import NCF
import torch.nn as nn
import torch.optim as optim


def model_creator(config):
    data_X = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    model = NCF(user_num, item_num,
                factor_num=32, num_layers=3, dropout=0.0, model="NeuMF-end")
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=0.001)

loss_function = nn.BCEWithLogitsLoss()

# Step 4: Fit with Orca Estimator

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

# Create the estimator
backend = "ray"  # "ray" or "spark"
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss_function, metrics=[Accuracy()], backend=backend)

# Fit the estimator
est.fit(data=train_loader_func, epochs=3, batch_size=256)

# Step 5: Save and Load the Model

# Evaluate the model
result = est.evaluate(data=test_loader_func, batch_size=256)
print('Evaluate results:')
for r in result:
    print(r, ":", result[r])

# Save the model
est.save("NCF_model")

# Stop orca context when program finishes
stop_orca_context()
