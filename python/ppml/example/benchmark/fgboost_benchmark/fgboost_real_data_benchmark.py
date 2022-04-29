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

    init_fl_context()
    df_train = pd.read_csv(args.train_path)
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
    
    df_test = pd.read_csv(args.test_path)
    result = fgboost_regression.predict(df_test, feature_columns=df_x.columns)
    
