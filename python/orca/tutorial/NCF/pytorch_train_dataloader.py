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

# Step 0: Import necessary libraries
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from pytorch_dataset import NCFData, load_dataset
from pytorch_model import NCF

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Define train and test datasets as PyTorch DataLoader
def train_loader_func(config, batch_size):
    data_X, user_num, item_num, cat_feats_dims, \
        feature_cols, label_cols = load_dataset(config['dataset_dir'],
                                                num_ng=config['num_ng'],
                                                cal_cat_feats_dims=False)
    total_cols = feature_cols + label_cols

    # train test split
    data_values = data_X.values.tolist()
    train_data, _ = train_test_split(data_values, test_size=0.2, random_state=100)
    train_data = pd.DataFrame(train_data, columns=total_cols, dtype=np.int64)
    train_data["label"] = train_data["label"].astype(np.float)

    train_dataset = NCFData(train_data)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    return train_loader


def test_loader_func(config, batch_size):
    data_X, user_num, item_num, cat_feats_dims, \
        feature_cols, label_cols = load_dataset(config['dataset_dir'],
                                                num_ng=config['num_ng'],
                                                cal_cat_feats_dims=False)
    total_cols = feature_cols + label_cols

    # train test split
    data_values = data_X.values.tolist()
    _, test_data = train_test_split(data_values, test_size=0.2, random_state=100)
    test_data = pd.DataFrame(test_data, columns=total_cols, dtype=np.int64)
    test_data["label"] = test_data["label"].astype(np.float)

    test_dataset = NCFData(test_data)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    return test_loader


# Step 3: Define the model, optimizer and loss
def model_creator(config):
    data_X, user_num, item_num, cat_feats_dims, \
        feature_cols, label_cols = load_dataset(config['dataset_dir'],
                                                num_ng=0,
                                                cal_cat_feats_dims=True)

    model = NCF(user_num=user_num,
                item_num=item_num,
                factor_num=config['factor_num'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                model=config['model'],
                cat_feats_dims=cat_feats_dims,
                num_numeric_feats=config['num_numeric_feats'])
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
                                   'model': "NeuMF-end",
                                   'num_numeric_feats': 1})
est.fit(data=train_loader_func, epochs=10, batch_size=1024)


# Step 5: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=1024)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 6: Save the trained PyTorch model
est.save("NCF_model")


# Step 7: Stop Orca Context when program finishes
stop_orca_context()
