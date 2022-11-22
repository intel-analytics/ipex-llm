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
from sklearn.preprocessing import MinMaxScaler


class NCFData(data.Dataset):
    def __init__(self, data, users=None, movies=None,
                 num_ng=4, user_num=0, item_num=0,
                 merge_features=False, total_cols=None):
        self.data = data

        if num_ng > 0:
            self.ng_sampling(num_ng, user_num, item_num)
        else:
            self.data["label"] = [1.0 for _ in range(len(self.data))]

        if merge_features:
            self.merge_features(users, movies, total_cols)

        self.data = list(map(lambda row: list(row[1:]), self.data.itertuples()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(self.data[idx])

    # sample negative items for datasets
    def ng_sampling(self, num_ng, user_num, item_num):
        # load ratings as a dok matrix
        features_ps = self.data.values.tolist()
        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
        for x in features_ps:
            train_mat[x[0], x[1]] = 1

        features_ng = []
        for x in features_ps:
            u = x[0]
            for t in range(num_ng):
                j = np.random.randint(item_num)
                while (u, j) in train_mat:
                    j = np.random.randint(item_num)
                features_ng.append([u, j])
        features = features_ps + features_ng
        labels_ps = [1.0 for _ in range(len(features_ps))]
        labels_ng = [0.0 for _ in range(len(features_ng))]
        labels = labels_ps + labels_ng
        self.data = pd.DataFrame(features, columns=["user", "item"], dtype=np.int64)
        self.data["label"] = labels
            
    def merge_features(self, users, movies, total_cols):
        self.data = users.merge(self.data, on='user')
        self.data = self.data.merge(movies, on='item')
        self.data = self.data.loc[:, total_cols]


def load_dataset(dataset_dir, cal_sparse_feats_input_dims=True,
                 num_ng=4, merge_features=True):
    """
    dataset_dir: the path of the datasets;
    cal_sparse_feats_input_dims: if True, will calculate sparse_feats_input_dims;
    num_ng: number of negative samples to be sampled here;
    merge_features: if True, will merge features in NCF_data.
    """
    feature_cols = ['user', 'item',
                    'gender', 'occupation', 'zipcode', 'category',  # sparse features
                    'age']  # dense features
    label_cols = ["label"]

    users = pd.read_csv(
        os.path.join(dataset_dir, 'users.dat'),
        sep="::", header=None, names=['user', 'gender', 'age', 'occupation', 'zipcode'],
        usecols=[0, 1, 2, 3, 4],
        dtype={0: np.int64, 1: np.object, 2: np.int64, 3: np.int64, 4: np.object})
    ratings = pd.read_csv(
        os.path.join(dataset_dir, 'ratings.dat'),
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    movies = pd.read_csv(
        os.path.join(dataset_dir, 'movies.dat'),
        sep="::", header=None, names=['item', 'category'],
        usecols=[0, 2], dtype={0: np.int64, 1: np.object})

    user_num = users['user'].max() + 1
    item_num = movies['item'].max() + 1

    # categorical encoding
    users.gender, _ = pd.Series(users.gender).factorize()
    users.zipcode, _ = pd.Series(users.zipcode).factorize()
    movies.category, _ = pd.Series(movies.category).factorize()

    # Calculate input_dims for each sparse features
    sparse_feats_input_dims = []
    if cal_sparse_feats_input_dims:
        sparse_feats_input_dims.append(users['gender'].max()+1)
        sparse_feats_input_dims.append(users['occupation'].max()+1)
        sparse_feats_input_dims.append(users['zipcode'].max()+1)
        sparse_feats_input_dims.append(movies['category'].max()+1)

    # scale dense features
    scaler = MinMaxScaler()
    age = users.age.values.reshape(-1, 1) 
    age = scaler.fit_transform(age)
    users.age = pd.Series(age[:, 0], dtype=np.float32)

    dataset = NCFData(ratings, users, movies,
                      num_ng, user_num, item_num,
                      merge_features, feature_cols+label_cols)
    return dataset, user_num, item_num, sparse_feats_input_dims


if __name__ == "__main__":
    load_dataset("./ml-1m")
