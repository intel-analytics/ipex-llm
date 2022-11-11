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
# Some of the code is adapted from
# https://github.com/guoyang9/NCF/blob/master/data_utils.py
#

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from bigdl.orca.data.pandas import read_csv


def ng_sampling(data, user_num, item_num):
    data_X = data.values.tolist()

    # calculate a dok matrix
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


def prepare_data(dataset_dir):
    data = read_csv(os.path.join(dataset_dir, 'ratings.dat'),
                    sep="::", header=None, names=['user', 'item'],
                    usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
    data = data.partition_by("user")

    user_set = set(data["user"].unique())
    item_set = set(data["item"].unique())
    user_num = max(user_set) + 1
    item_num = max(item_set) + 1

    data = data.transform_shard(lambda shard: ng_sampling(shard, user_num, item_num))
    train_data, test_data = data.transform_shard(split_dataset).split()
    return train_data, test_data, user_num, item_num


if __name__ == "__main__":
    from bigdl.orca import init_orca_context, stop_orca_context

    sc = init_orca_context()
    train_data, test_data, user_num, item_num = prepare_data("./ml-1m")
    train_data.save_pickle("train_xshards")
    test_data.save_pickle("test_xshards")
    stop_orca_context()
