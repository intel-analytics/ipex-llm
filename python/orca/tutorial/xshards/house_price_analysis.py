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
# This example is adapted from
# https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/notebook

import bigdl.orca.data.pandas
from bigdl.orca.data.transformer import *

import numpy as np

path = '/home/ding/data/house_price/train.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path, nullValue="NA")

null_cnt_shard = data_shard.get_null_sum()
print(null_cnt_shard.collect())

missing_data_shards = null_cnt_shard.sort_values(col_names="total", ascending=False)
print(missing_data_shards.collect())

#dealing with missing data
new_shards = data_shard.drop_missing_value()

# verify missing value has been removed
new_shards3 = new_shards.get_null_sum()
max_value = new_shards3.max_values('total')

def drop_data(df):
    df = df.drop(df[df['Id'] == 0].index)
    df = df.drop(df[df['Id'] == 1].index)
    return df
new_shards3 = new_shards.transform_shard(drop_data)

#applying log transformation
def generate_new_sale_price(df):
    df['SalePrice'] = np.log(df['SalePrice'])
    return df
new_shards4 = new_shards3.transform_shard(generate_new_sale_price)

#create column for new variable (one is enough because it's a binary categorical feature)
def generate_HasBsmt(df):
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    return df
new_shards5 = new_shards4.transform_shard(generate_HasBsmt)

#transform data
def generate_HasBsmt(df):
    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
    return df
new_shards6 = new_shards5.transform_shard(generate_HasBsmt)

#convert categorical variable into dummy
# df_train = pd.get_dummies(df_train)
t = 0