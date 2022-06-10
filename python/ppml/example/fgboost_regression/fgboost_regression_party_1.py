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


from bigdl.ppml.fl import *
from bigdl.ppml.fl.algorithms.fgboost_regression import FGBoostRegression
import pandas as pd

init_fl_context()
df_train = pd.read_csv('house-prices-train-1.csv')
fgboost_regression = FGBoostRegression()

# party 1 does not own label, so directly pass all the features
fgboost_regression.fit(df_train, feature_columns=df_train.columns, num_round=100)

df_test = pd.read_csv('house-prices-test-1')
result = fgboost_regression.predict(df_test, feature_columns=df_test.columns)

fgboost_regression.save_model('/tmp/fgboost_model_1.json')
loaded = FGBoostRegression.load_model('/tmp/fgboost_model_1.json')
