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

# Step 1: Init Orca Context

from bigdl.orca import init_orca_context, stop_orca_context
init_orca_context()

# Step 2: Define Dataset

from bigdl.orca.data.pandas import read_csv
from sklearn.model_selection import train_test_split


def preprocess_data():
    data_X = read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    data_X = data_X.partition_by("user")

    user_set = set(data_X["user"].unique())
    item_set = set(data_X["item"].unique())
    user_num = max(user_set) + 1
    item_num = max(item_set) + 1
    return data_X, user_num, item_num


def ng_sampling(data):
    data_X = data.values.tolist()

    # calculate a dok matrix
    import scipy.sparse as sp
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for row in data_X:
        train_mat[row[0], row[1]] = 1

    # negative sampling
    features_ps = data_X
    features_ng = []
    for x in features_ps:
        u = x[0]
        for t in range(4):  # sample 4 negative items for training
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
    # split training set and testing set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=100)
    return train_data, test_data

# Prepare the train and test datasets
data_X, user_num, item_num = preprocess_data()

# Construct the train and test xshards
data_X = data_X.transform_shard(ng_sampling)
train_shards, test_shards = data_X.transform_shard(split_dataset).split()

# Step 3: Define the Model

import torch.nn as nn
import torch.optim as optim
from model import NCF


def model_creator(config):
    model = NCF(config['user_num'], config['item_num'],
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
est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=loss_function,
                           metrics=[Accuracy()],
                           config={'user_num': user_num, 'item_num': item_num},
                           backend=backend)

# Fit the estimator
est.fit(data=train_shards, epochs=1, batch_size=256,
        feature_cols=["user", "item"], label_cols=["label"])

# Step 5: Evaluate and save the Model

# Evaluate the model
result = est.evaluate(data=test_shards,
                      feature_cols=["user", "item"],
                      label_cols=["label"], batch_size=256)
print('Evaluate results:')
for r in result:
    print(r, ":", result[r])

# Save the model
est.save("NCF_model")

# Stop orca context when program finishes
stop_orca_context()
