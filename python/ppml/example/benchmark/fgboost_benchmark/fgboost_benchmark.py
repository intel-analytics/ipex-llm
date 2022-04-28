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
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PPML FGBoost Benchmark',
        description='Run PPML FGBoost Benchmark using dummy data, shape=[data_size, 100]')
    parser.add_argument('--data_size',
                        type=int,
                        default=100,
                        help='The data size of dummy training data.')
    parser.add_argument('--data_dim',
                        type=int,
                        default=10,
                        help='The data dimension of dummy training data.')
    parser.add_argument('--num_round',
                        type=int,
                        default=10,
                        help='The boosting rounds.')
    parser.add_argument('--has_label',
                        type=bool,
                        default=True,
                        help='If this client has label in data, default True.')
    args = parser.parse_args()
    x = np.random.rand(args.data_size, args.data_dim)
    y = np.random.rand(args.data_size)


    init_fl_context()
    fgboost_regression = FGBoostRegression(max_depth=3)
    ts = time.time()
    if args.has_label:
        fgboost_regression.fit(x, y, num_round=args.num_round)
    else:
        fgboost_regression.fit(x, num_round=args.num_round)
    te = time.time()
    # result = fgboost_regression.predict(x)
    # pe = time.time()
    # result = list(map(lambda x: math.exp(x), result))

    train_time = round(te - ts, 3)
    # predict_time = round(pe - te, 3)
    print (f"data: [{args.data_size}, {args.data_dim}], boost_round: {args.num_round}")
    # print (f"training time: {train_time}, predict time: {predict_time}")
    # result
