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
from bigdl.orca.data.transformer import StringIndexer, MinMaxScaler

# user/item ids and sparse features are converted to int64 to be compatible with
# lower versions of PyTorch such as 1.7.1.

sparse_features = ["zipcode", "gender", "category", "occupation"]
dense_features = ["age"]


def ng_sampling(df, user_num, item_num, num_ng):
    data_X = df.values.tolist()

    # Calculate a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int32)
    for row in data_X:
        train_mat[row[0], row[1]] = 1

    features_ps = data_X
    features_ng = []
    for x in features_ps:
        u = x[0]
        for t in range(num_ng):
            j = np.random.randint(item_num)
            while (u, j) in train_mat:
                j = np.random.randint(item_num)
            features_ng.append([u, j])

    labels_ps = [1.0 for _ in range(len(features_ps))]
    labels_ng = [0.0 for _ in range(len(features_ng))]

    features_fill = features_ps + features_ng
    labels_fill = labels_ps + labels_ng
    data_XY = pd.DataFrame(data=features_fill, columns=["user", "item"], dtype=np.int64)
    data_XY["label"] = labels_fill
    return data_XY


def split_dataset(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=100)
    return train_data, test_data


def prepare_data(data_dir="./", dataset="ml-1m", num_ng=4):
    print("Loading data...")
    if dataset == "ml-1m":
        # Need spark3 to support delimiter with more than one character.
        users = read_csv(
            os.path.join(data_dir, dataset, "users.dat"),
            sep="::", header=None, names=["user", "gender", "age", "occupation", "zipcode"],
            usecols=[0, 1, 2, 3, 4],
            dtype={0: np.int64, 1: str, 2: np.int32, 3: np.int64, 4: str})
        ratings = read_csv(
            os.path.join(data_dir, dataset, "ratings.dat"),
            sep="::", header=None, names=["user", "item"],
            usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
        items = read_csv(
            os.path.join(data_dir, dataset, "movies.dat"),
            sep="::", header=None, names=["item", "category"],
            usecols=[0, 2], dtype={0: np.int64, 2: str})
    else:  # ml-100k
        users = read_csv(
            os.path.join(data_dir, dataset, "u.user"),
            sep="|", header=None, names=["user", "age", "gender", "occupation", "zipcode"],
            usecols=[0, 1, 2, 3, 4],
            dtype={0: np.int64, 1: np.int32, 2: str, 3: str, 4: str})
        ratings = read_csv(
            os.path.join(data_dir, dataset, "u.data"),
            sep="\t", header=None, names=["user", "item"],
            usecols=[0, 1], dtype={0: np.int64, 1: np.int64})
        items = read_csv(
            os.path.join(data_dir, dataset, "u.item"),
            sep="|", header=None,
            names=["item"]+[f"col{i}" for i in range(19)],
            usecols=[0]+list(range(5, 24)),
            dtype=np.int64)

        # Merge multiple one-hot columns into one movie category column
        def merge_one_hot_cols(df):
            df["category"] = df.iloc[:, 1:].apply(lambda x: "".join(str(x)), axis=1)
            return df.drop(columns=[f"col{i}" for i in range(19)])

        items = items.transform_shard(merge_one_hot_cols)

    # Calculate user and item num
    user_set = set(users["user"].unique())
    item_set = set(items["item"].unique())
    user_num = int(max(user_set) + 1)
    item_num = int(max(item_set) + 1)

    print("Processing features...")

    def convert_to_long(df, col):
        df[col] = df[col].astype(np.int64)
        return df

    # Categorical encoding
    for col in sparse_features:
        indexer = StringIndexer(inputCol=col)
        if col in users.get_schema()["columns"]:
            users = indexer.fit_transform(users)
            users = users.transform_shard(lambda df: convert_to_long(df, col))
        else:
            items = indexer.fit_transform(items)
            items = items.transform_shard(lambda df: convert_to_long(df, col))

    # Calculate input_dims for each sparse features
    sparse_feats_input_dims = []
    for col in sparse_features:
        data = users if col in users.get_schema()["columns"] else items
        sparse_feat_set = set(data[col].unique())
        sparse_feats_input_dims.append(int(max(sparse_feat_set) + 1))

    # Scale dense features
    def rename(df, col):
        df = df.drop(columns=[col]).rename(columns={col + "_scaled": col})
        return df

    for col in dense_features:
        scaler = MinMaxScaler(inputCol=[col], outputCol=col + "_scaled")
        if col in users.get_schema()["columns"]:
            users = scaler.fit_transform(users)
            users = users.transform_shard(lambda df: rename(df, col))
        else:
            items = scaler.fit_transform(items)
            items = items.transform_shard(lambda df: rename(df, col))

    # Negative sampling
    print("Negative sampling...")
    ratings = ratings.partition_by("user")
    ratings = ratings.transform_shard(lambda df: ng_sampling(df, user_num, item_num, num_ng))

    # Merge XShards
    print("Merge data...")
    data = users.merge(ratings, on="user")
    data = data.merge(items, on="item")

    # Split dataset
    print("Split data...")
    train_data, test_data = data.transform_shard(split_dataset).split()

    return train_data, test_data, user_num, item_num, \
        sparse_feats_input_dims, len(dense_features), get_feature_cols(), get_label_cols()


def get_feature_cols():
    return ["user", "item"] + sparse_features + dense_features


def get_label_cols():
    return ['label']


if __name__ == "__main__":
    from utils import init_orca, stop_orca_context

    init_orca("local")
    train_data, test_data, user_num, item_num, sparse_feats_input_dims, num_dense_feats, \
        feature_cols, label_cols = prepare_data()
    train_data.save_pickle("./train_processed_xshards")
    test_data.save_pickle("./test_processed_xshards")
    stop_orca_context()
