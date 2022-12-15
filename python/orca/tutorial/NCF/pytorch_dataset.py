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
# Most of the code is adapted from
# https://github.com/guoyang9/NCF/blob/master/data_utils.py
#

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class NCFData(data.Dataset):
    def __init__(self, data,
                 num_item=0, train_mat=None, num_ng=0):
        super(NCFData, self).__init__()
        self.data = data
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_sampling = False

        if 'label' not in data.columns:
            self.data['label'] = [1.0 for _ in range(len(self.data))]

    def ng_sample(self):
        self.is_sampling = True

        features_ps = self.data.values[:, :-1].tolist()
        features_ng = []
        for x in features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])
        labels_ps = [1.0 for _ in range(len(features_ps))]
        labels_ng = [0.0 for _ in range(len(features_ng))]
        features_fill = features_ps + features_ng
        labels_fill = labels_ps + labels_ng
        self.data = pd.DataFrame(features_fill,
                                 columns=["user", "item"], dtype=np.int32)
        self.data['label'] = labels_fill

    def train_test_split(self, test_size=0.2):
        train_dataset, test_dataset = train_test_split(self.data, test_size=test_size,
                                                       random_state=100)
        train_dataset.reset_index(drop=True, inplace=True)
        test_dataset.reset_index(drop=True, inplace=True)
        return NCFData(train_dataset), NCFData(test_dataset)

    def merge_features(self, users, items, total_cols=None):
        self.data = users.merge(self.data, on='user')
        self.data = self.data.merge(items, on='item')
        # To make the order of data columns as expected.
        if total_cols is not None:
            self.data = self.data.loc[:, total_cols]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(self.data.at[idx, i] for i in self.data.columns)


def process_users_items(dataset_dir):
    sparse_features = ['gender', 'zipcode', 'category']
    dense_features = ['age']

    users = pd.read_csv(
        os.path.join(dataset_dir, 'users.dat'),
        sep="::", header=None, names=['user', 'gender', 'age', 'occupation', 'zipcode'],
        usecols=[0, 1, 2, 3, 4],
        dtype={0: np.int32, 1: str, 2: np.int32, 3: np.int32, 4: str})
    items = pd.read_csv(
        os.path.join(dataset_dir, 'movies.dat'),
        sep="::", header=None, names=['item', 'category'],
        usecols=[0, 2], dtype={0: np.int32, 1: str})

    user_num = users['user'].max() + 1
    item_num = items['item'].max() + 1

    # categorical encoding
    for i in sparse_features:
        df = users if i in users.columns else items
        df[i], _ = pd.Series(df[i]).factorize()
    sparse_features.append('occupation')  # occupation is already indexed.

    # scale dense features
    for i in dense_features:
        scaler = MinMaxScaler()
        df = users if i in users.columns else items
        # MinMaxScaler needs the input to be 2-dim tensor, not 1-dim.
        values = df[i].values.reshape(-1, 1)
        values = scaler.fit_transform(values)
        values = [np.array(v, dtype=np.float32) for v in values]
        df[i] = values

    feature_cols = ['user', 'item'] + sparse_features + dense_features
    label_cols = ["label"]
    return users, items, user_num, item_num, \
        sparse_features, dense_features, feature_cols+label_cols


def get_input_dims(users, items, sparse_features, dense_features):
    # Calculate input_dims for each sparse features
    sparse_feats_input_dims = []
    for i in sparse_features:
        df = users if i in users.columns else items
        sparse_feats_input_dims.append(df[i].max()+1)

    num_dense_feats = len(dense_features)
    return sparse_feats_input_dims, num_dense_feats


def process_ratings(dataset_dir, user_num, item_num):
    ratings = pd.read_csv(
        os.path.join(dataset_dir, 'ratings.dat'),
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int32)
    for x in ratings.values.tolist():
        train_mat[x[0], x[1]] = 1
    return ratings, train_mat


def load_dataset(dataset_dir, num_ng=4):
    """
    dataset_dir: the path of the datasets;
    num_ng: number of negative samples to be sampled here.
    """
    users, items, user_num, item_num, sparse_features, dense_features, \
        total_cols = process_users_items(dataset_dir)
    ratings, train_mat = process_ratings(dataset_dir, user_num, item_num)

    # sample negative items
    dataset = NCFData(ratings, item_num, train_mat, num_ng)
    dataset.ng_sample()

    # merge features
    dataset.merge_features(users, items, total_cols)

    # train test split
    train_dataset, test_dataset = dataset.train_test_split()
    return train_dataset, test_dataset


if __name__ == "__main__":
    load_dataset("./ml-1m")
