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
    def __init__(self, data):
        self.data = data.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = int(self.data[idx][0])
        item = int(self.data[idx][1])
        label = float(self.data[idx][2])
        return user, item, label


def train_loader_func(config, batch_size):
    data_X = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    features_ps = data_X.values.tolist()

    # load ratings as a dok matrix
    import scipy.sparse as sp
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for x in features_ps:
        train_mat[x[0], x[1]] = 1

    # sample negative items for training datasets
    np.random.seed(0)
    features_ng = []
    for x in features_ps:
        u = x[0]
        for t in range(4):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            features_ng.append([u, j])
    features = features_ps + features_ng
    labels_ps = [1 for _ in range(len(features_ps))]
    labels_ng = [0 for _ in range(len(features_ng))]
    labels = labels_ps + labels_ng
    data_X = pd.DataFrame(features, columns=["user", "item"], dtype=np.int64)
    data_X["label"] = labels

    # train test split
    data_X = data_X.values.tolist()
    train_data, _ = train_test_split(data_X, test_size=0.2, random_state=100)
    train_data = pd.DataFrame(train_data, columns=["user", "item", "label"], dtype=np.int64)
    train_data["label"] = train_data["label"].astype(np.float)

    train_dataset = NCFData(train_data)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    data_X = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    features_ps = data_X.values.tolist()

    # load ratings as a dok matrix
    import scipy.sparse as sp
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for x in features_ps:
        train_mat[x[0], x[1]] = 1

    # sample negative items for training datasets
    np.random.seed(0)
    features_ng = []
    for x in features_ps:
        u = x[0]
        for t in range(4):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            features_ng.append([u, j])
    features = features_ps + features_ng
    labels_ps = [1 for _ in range(len(features_ps))]
    labels_ng = [0 for _ in range(len(features_ng))]
    labels = labels_ps + labels_ng
    data_X = pd.DataFrame(features, columns=["user", "item"], dtype=np.int64)
    data_X["label"] = labels

    # train test split
    data_X = data_X.values.tolist()
    _, test_data = train_test_split(data_X, test_size=0.2, random_state=100)
    test_data = pd.DataFrame(test_data, columns=["user", "item", "label"], dtype=np.int64)
    test_data["label"] = test_data["label"].astype(np.float)

    test_dataset = NCFData(test_data)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
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

loss = nn.BCEWithLogitsLoss()


# Step 4: Distributed training with Orca PyTorch Estimator
batch_size = 1024
backend = "ray"  # "ray" or "spark"
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss, metrics=[Accuracy(), Precision(), Recall()],
                           backend=backend)
est.fit(data=train_loader_func, epochs=10, batch_size=batch_size)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=batch_size)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
