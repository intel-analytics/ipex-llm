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
from sklearn.model_selection import train_test_split

from model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Define train and test datasets using Orca XShards
dataset_dir = "./ml-1m"


def ng_sampling(data):
    data_X = data.values.tolist()

    # calculate a dok matrix
    import scipy.sparse as sp
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for row in data_X:
        train_mat[row[0], row[1]] = 1

    # negative sampling
    num_ng = 4  # sample 4 negative items for training
    features_ps = data_X
    features_ng = []
    for x in features_ps:
        u = x[0]
        for t in range(num_ng):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            features_ng.append([u, j])

    labels_ps = [1 for _ in range(len(features_ps))]
    labels_ng = [0 for _ in range(len(features_ng))]

    features_fill = features_ps + features_ng
    labels_fill = labels_ps + labels_ng
    data_XY = pd.DataFrame(data=features_fill, columns=["user", "item"], dtype=np.int64)
    data_XY["label"] = labels_fill
    data_XY["label"] = data_XY["label"].astype(np.float)
    return data_XY


def split_dataset(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=100)
    return train_data, test_data

data = read_csv(os.path.join(dataset_dir, 'ratings.dat'),
                sep="::", header=None, names=['user', 'item'],
                usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
data = data.partition_by("user")

user_set = set(data["user"].unique())
item_set = set(data["item"].unique())
user_num = max(user_set) + 1
item_num = max(item_set) + 1

data = data.transform_shard(ng_sampling)
train_data, test_data = data.transform_shard(split_dataset).split()


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    model = NCF(config['user_num'], config['item_num'],
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
backend = "spark"  # "ray" or "spark"

est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,
                           loss=loss,
                           metrics=[Accuracy(), Precision(), Recall()],
                           backend=backend,
                           config={'user_num': user_num, 'item_num': item_num,
                                   'dataset_dir': dataset_dir,
                                   'factor_num': 16,
                                   'num_layers': 3,
                                   'dropout': 0.0,
                                   'lr': 0.001,
                                   'model': "NeuMF-end"})
est.fit(data=train_data, epochs=10, batch_size=256,
        feature_cols=["user", "item"], label_cols=["label"])


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_data, batch_size=256,
                      feature_cols=["user", "item"], label_cols=["label"])
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
