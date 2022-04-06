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
# 
# This example is adapted from
# https://github.com/ray-project/ray/blob/releases/1.9.0/doc/examples/datasets_train/datasets_train.py
#
# MIT License
#
# Copyright (c) 2019 Zihao Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import random
import argparse

import pandas as pd
import sklearn


parser = argparse.ArgumentParser(description='Make and Upload Dataset')
parser.add_argument('--data_dir', type=str, default="./data", help='The path of dataset')
parser.add_argument('--num_examples', type=int, default=20000, help='The number of examples')
parser.add_argument('--use_s3', type=str, default=False, help='Use data from s3 for testing.')
args = parser.parse_args()

def make_and_upload_dataset(
        dir_path=args.data_dir, num_examples=20000, num_features=20,
        parquet_file_chunk_size_rows=50_000, upload_to_s3=args.use_s3):

    num_files = num_examples // parquet_file_chunk_size_rows

    def create_data_chunk(n, d, seed, include_label=False):
        X, y = sklearn.datasets.make_classification(
            n_samples=n,
            n_features=d,
            n_informative=10,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=3,
            weights=None,
            flip_y=0.03,
            class_sep=0.8,
            hypercube=True,
            shift=0.0,
            scale=1.0,
            shuffle=False,
            random_state=seed)

        # turn into dataframe with column names
        col_names = ["feature_%0d" % i for i in range(1, d + 1, 1)]
        df = pd.DataFrame(X)
        df.columns = col_names

        # add some bogus categorical data columns
        options = ["apple", "banana", "orange"]
        df["fruit"] = df.feature_1.map(
            lambda x: random.choice(options)
        )  # bogus, but nice to test categoricals

        # add some nullable columns
        options = [None, 1, 2]
        df["nullable_feature"] = df.feature_1.map(
            lambda x: random.choice(options)
        )  # bogus, but nice to test categoricals

        # add label column
        if include_label:
            df["label"] = y
        return df

    # create data files
    print("Creating synthetic dataset...")
    data_path = os.path.join(dir_path, "data")
    os.makedirs(data_path, exist_ok=True)
    for i in range(num_files):
        path = os.path.join(data_path, f"data_{i:05d}.parquet.snappy")
        if not os.path.exists(path):
            tmp_df = create_data_chunk(
                n=parquet_file_chunk_size_rows,
                d=num_features, seed=i,
                include_label=True)
            tmp_df.to_parquet(path, compression="snappy", index=False)
        print(f"Wrote {path} to disk...")

    print("Creating synthetic inference dataset...")
    inference_path = os.path.join(dir_path, "inference")
    os.makedirs(inference_path, exist_ok=True)
    for i in range(num_files):
        path = os.path.join(inference_path, f"data_{i:05d}.parquet.snappy")
        if not os.path.exists(path):
            tmp_df = create_data_chunk(
                n=parquet_file_chunk_size_rows,
                d=num_features, seed=i,
                include_label=False)
            tmp_df.to_parquet(path, compression="snappy", index=False)
        print(f"Wrote {path} to disk...")

    if upload_to_s3:
        os.system("aws s3 sync ./data s3://cuj-big-data/data")
        os.system("aws s3 sync ./inference s3://cuj-big-data/inference")
