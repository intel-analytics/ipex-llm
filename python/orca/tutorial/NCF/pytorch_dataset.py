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
import tempfile
import numpy as np
import pandas as pd

import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch.utils.data as data

from bigdl.dllib.utils.file_utils import is_local_path
from bigdl.orca.data.file import get_remote_dir_to_local, get_remote_file_to_local

# user/item ids and sparse features are converted to int64 to be compatible with
# lower versions of PyTorch such as 1.7.1.


class NCFData(data.Dataset):
    def __init__(self, features, labels=None,
                 num_item=0, train_mat=None, num_ng=0):
        super(NCFData, self).__init__()
        self.features = features
        self.labels = labels
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_sampling = False

        if labels is None:
            self.labels = [1.0 for _ in range(len(self.features))]

    def ng_sample(self):
        self.is_sampling = True

        features_ps = self.features
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
        self.features = features_ps + features_ng
        self.labels = labels_ps + labels_ng

    def merge_features(self, users, items, feature_cols=None):
        df = pd.DataFrame(self.features, columns=["user", "item"], dtype=np.int64)
        df["labels"] = self.labels
        df = users.merge(df, on="user")
        df = df.merge(items, on="item")

        # To make the order of data columns as expected.
        if feature_cols:
            self.features = df.loc[:, feature_cols]
        self.features = tuple(map(list, self.features.itertuples(index=False)))
        self.labels = df["labels"].values.tolist()

    def train_test_split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels,
                                                            test_size=test_size, random_state=100)
        return NCFData(X_train, y_train), NCFData(X_test, y_test)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx] + [self.labels[idx]]


def process_users_items(data_dir, dataset):
    sparse_features = ["gender", "zipcode", "category", "occupation"]
    dense_features = ["age"]

    print("Loading user and movie data...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        if is_local_path(data_dir):
            local_dir = os.path.join(data_dir, dataset)
        else:
            get_remote_dir_to_local(remote_dir=os.path.join(data_dir, dataset),
                                    local_dir=tmpdirname)
            local_dir = os.path.join(tmpdirname, dataset)

        if dataset == "ml-1m":
            users = pd.read_csv(
                os.path.join(local_dir, "users.dat"),
                sep="::", header=None, names=["user", "gender", "age", "occupation", "zipcode"],
                usecols=[0, 1, 2, 3, 4],
                dtype={0: np.int64, 1: str, 2: np.int32, 3: np.int64, 4: str},
                engine="python")
            items = pd.read_csv(
                os.path.join(local_dir, "movies.dat"),
                sep="::", header=None, names=["item", "category"],
                usecols=[0, 2], dtype={0: np.int64, 1: str},
                engine="python", encoding="latin-1")
        else:  # ml-100k
            users = pd.read_csv(
                os.path.join(local_dir, "u.user"),
                sep="|", header=None, names=["user", "age", "gender", "occupation", "zipcode"],
                usecols=[0, 1, 2, 3, 4],
                dtype={0: np.int64, 1: np.int32, 2: str, 3: str, 4: str})
            items = pd.read_csv(
                os.path.join(local_dir, "u.item"),
                sep="|", header=None, names=["item"]+[f"col{i}" for i in range(19)],
                usecols=[0]+list(range(5, 24)),
                dtype=np.int64, encoding="latin-1")

            # Merge multiple one-hot columns into one category column
            items["category"] = items.iloc[:, 1:].apply(lambda x: "".join(str(x)), axis=1)
            items.drop(columns=[f"col{i}" for i in range(19)], inplace=True)

    user_num = users["user"].max() + 1
    item_num = items["item"].max() + 1

    # Categorical encoding
    for i in sparse_features:
        df = users if i in users.columns else items
        df[i], _ = pd.Series(df[i]).factorize()

    # Scale dense features
    for i in dense_features:
        scaler = MinMaxScaler()
        df = users if i in users.columns else items
        # MinMaxScaler needs the input to be 2-dim tensor, not 1-dim.
        values = df[i].values.reshape(-1, 1)
        values = scaler.fit_transform(values)
        values = [np.array(v, dtype=np.float32) for v in values]
        df[i] = values

    feature_cols = ["user", "item"] + sparse_features + dense_features
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


def process_ratings(data_dir, dataset, user_num, item_num):
    print("Loading ratings...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = "ratings.dat" if dataset == "ml-1m" else "u.data"
        sep = "::" if dataset == "ml-1m" else "\t"

        if is_local_path(data_dir):
            local_path = os.path.join(data_dir, dataset, file_name)
        else:
            remote_path = os.path.join(data_dir, dataset, file_name)
            local_path = os.path.join(tmpdirname, file_name)
            get_remote_file_to_local(remote_path=remote_path, local_path=local_path)

        ratings = pd.read_csv(
            local_path,
            sep=sep, header=None, names=["user", "item"],
            usecols=[0, 1], dtype={0: np.int64, 1: np.int64},
            engine="python")

    # Load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.int32)
    for x in ratings.values.tolist():
        train_mat[x[0], x[1]] = 1
    return ratings, train_mat


def load_dataset(data_dir="./", dataset="ml-1m", num_ng=4):
    """
    data_dir: the path to the dataset;
    dataset: the name of the dataset;
    num_ng: number of negative samples to be sampled here.
    """
    users, items, user_num, item_num, sparse_features, dense_features, \
        total_cols = process_users_items(data_dir, dataset)
    ratings, train_mat = process_ratings(data_dir, dataset, user_num, item_num)

    # Negative sampling
    dataset = NCFData(ratings.values.tolist(),
                      num_item=item_num, train_mat=train_mat, num_ng=num_ng)
    dataset.ng_sample()

    # Merge features
    dataset.merge_features(users, items, total_cols[: -1])

    # Split dataset
    train_dataset, test_dataset = dataset.train_test_split()
    return train_dataset, test_dataset


if __name__ == "__main__":
    load_dataset()
