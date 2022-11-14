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


class NCFData(data.Dataset):
    def __init__(self, data):
        self.data = list(map(lambda row: list(row[1:]), data.itertuples()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(self.data[idx])


def load_dataset(dataset_dir, num_ng=4, total_cols=['user', 'item', "label"],
                 cal_num_embed=False):
    """
    dataset_dir: the path of the datasets;
    num_ng: number of negative samples to be sampled here;
    total_cols: the list of total features;
    cal_num_embed: if True, will calculate num_embed_cat_feats.
    """
    data_1 = pd.read_csv(
        os.path.join(dataset_dir, 'users.dat'),
        sep="::", header=None, names=['user', 'gender', 'age', 'occupation', 'zipcode'],
        usecols=[0, 1, 2, 3, 4],
        dtype={0: np.int64, 1: np.object, 2: np.int64, 3: np.int64, 4: np.object})
    data_2 = pd.read_csv(
        os.path.join(dataset_dir, 'ratings.dat'),
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    data_3 = pd.read_csv(
        os.path.join(dataset_dir, 'movies.dat'),
        sep="::", header=None, names=['item', 'category'],
        usecols=[0, 2], dtype={0: np.int64, 1: np.object})

    user_num = data_2['user'].max() + 1
    item_num = data_2['item'].max() + 1

    # category encoding
    data_1.gender, _ = pd.Series(data_1.gender).factorize()
    data_1.zipcode, _ = pd.Series(data_1.zipcode).factorize()
    data_3.category, _ = pd.Series(data_3.category).factorize()

    # sample negative items for training datasets
    if num_ng > 0:
        # load ratings as a dok matrix
        features_ps = data_2.values.tolist()
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
        labels_ps = [1 for _ in range(len(features_ps))]
        labels_ng = [0 for _ in range(len(features_ng))]
        labels = labels_ps + labels_ng
        data_2 = pd.DataFrame(features, columns=["user", "item"], dtype=np.int64)
        data_2["label"] = labels
    else:
        data_2["label"] = [1 for _ in range(len(data_2))]

    # merge dataframes
    data_X = data_1.merge(data_2, on='user')
    data_X = data_X.merge(data_3, on='item')
    data_X = data_X.loc[:, total_cols]

    # Calculate num_embeddings for each categorical features
    num_embed_cat_feats = []
    if cal_num_embed:
        num_embed_cat_feats.append(user_num)
        num_embed_cat_feats.append(item_num)
        num_embed_cat_feats.append(data_1['gender'].max()+1)
        num_embed_cat_feats.append(data_1['occupation'].max()+1)
        num_embed_cat_feats.append(data_1['zipcode'].max()+1)
        num_embed_cat_feats.append(data_3['category'].max()+1)

    return data_X, num_embed_cat_feats


if __name__ == "__main__":
    load_dataset("./ml-1m")
