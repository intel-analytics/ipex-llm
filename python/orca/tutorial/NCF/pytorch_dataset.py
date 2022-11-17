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


def load_dataset(dataset_dir):
    data_X = pd.read_csv(
        os.path.join(dataset_dir, 'ratings.dat'),
        sep="::", header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int64, 1: np.int64})

    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int64)
    for x in data_X.values.tolist():
        train_mat[x[0], x[1]] = 1

    return data_X, user_num, item_num, train_mat


if __name__ == "__main__":
    load_dataset("./ml-1m")
