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

import numpy as np
import pandas as pd
import argparse
import time
from bigdl.ppml.fl import *
from bigdl.ppml.fl.data_utils import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error
import os
import sys


def is_local_and_existing_uri(uri):
    parsed_uri = urlparse(uri)

    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file',
                                "Not Local File!")

    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost',
                                "Not Local File!")

    log4Error.invalidInputError(exists(parsed_uri.path),
                                "File Not Exist!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PPML FGBoost Benchmark',
        description='Run PPML FGBoost Benchmark using real data')
    parser.add_argument('--train_path',
                        type=str,
                        help='The train data path.')
    parser.add_argument('--test_path',
                        type=str,
                        help='The test data path.')
    parser.add_argument('--data_size',
                        type=int,
                        default=1,
                        help='The size of data copy, e.g. 10 means using 10 copies of data.')
    parser.add_argument('--num_round',
                        type=int,
                        default=10,
                        help='The boosting rounds.')
    args = parser.parse_args()
    # process path traversal issue
    safe_dir_train = "/safe_dir/"
    dir_name_train = os.path.dirname(args.train_path)
    if '../' in dir_name_train:
        sys.exit(1)
    safe_dir_train = dir_name_train
    file_name_train = os.path.basename(args.train_path)
    train_path = os.path.join(safe_dir_train, file_name_train)
    
    safe_dir_test = "/safe_dir/"
    dir_name_test = os.path.dirname(args.test_path)
    if '../' in dir_name_test:
        sys.exit(1)
    safe_dir_test = dir_name_test
    file_name_test = os.path.basename(args.test_path)
    test_path = os.path.join(safe_dir_test, file_name_test)

    init_fl_context(1)
    is_local_and_existing_uri(train_path)
    df_train = pd.read_csv(train_path)
    fgboost_regression = FGBoostRegression()
    
    df_x = df_train.drop('SalePrice', 1)
    df_y = df_train.filter(items=['SalePrice'])
    x = convert_to_numpy(df_x)
    y = convert_to_numpy(df_y)
    x_stacked = []
    y_stacked = []
    for i in range(args.data_size):
        x_stacked.append(x)
        y_stacked.append(y)
    
    x_stacked = np.array(x_stacked)
    y_stacked = np.array(y_stacked)
    
    fgboost_regression.fit(x_stacked.reshape(-1, x_stacked.shape[-1]),
                           y_stacked.reshape(-1, y_stacked.shape[-1]),
                           num_round=args.num_round)
    
    is_local_and_existing_uri(test_path)
    df_test = pd.read_csv(test_path)
    result = fgboost_regression.predict(df_test, feature_columns=df_x.columns)
    
