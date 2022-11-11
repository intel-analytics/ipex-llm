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

# Step 0: Import necessary libraries
import os
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Define train and test datasets as PyTorch DataLoader
class NCFData(data.Dataset):
    def __init__(self, features,
                 num_item=0, train_mat=None, num_ng=0):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_sampling = False
        self.labels = [1 for _ in range(len(features))]

    def ng_sample(self):
        self.is_sampling = True
        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_sampling \
            else self.features_ps
        labels = self.labels_fill if self.is_sampling \
            else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = float(labels[idx])
        return user, item, label


def train_loader_func(config, batch_size):
    data_X = pd.read_csv(
        os.path.join(config['dataset_dir'], 'ratings.dat'),
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

    # train test split
    train_data, _ = train_test_split(data_X, test_size=0.2, random_state=100)

    train_dataset = NCFData(train_data, item_num, train_mat, config['num_ng'])
    train_dataset.ng_sample()
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    data_X = pd.read_csv(
        os.path.join(config['dataset_dir'], 'ratings.dat'),
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

    # train test split
    _, test_data = train_test_split(data_X, test_size=0.2, random_state=100)

    test_dataset = NCFData(test_data, item_num, train_mat, config['num_ng'])
    test_dataset.ng_sample()
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    data_X = pd.read_csv(
        os.path.join(config['dataset_dir'], 'ratings.dat'),
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    model = NCF(user_num, item_num,
                factor_num=config['factor_num'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                model=config['model'])
    model.train()
    return model


def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config['lr'])

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
dataset_dir = "./ml-1m"
backend = "ray"  # "ray" or "spark"

est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=backend,
                           config={'dataset_dir': dataset_dir,
                                   'num_ng': 4,
                                   'factor_num': 16,
                                   'num_layers': 3,
                                   'dropout': 0.5,
                                   'lr': 0.001,
                                   'model': "NeuMF-end"})
est.fit(data=train_loader_func, epochs=10, batch_size=256)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=256)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
