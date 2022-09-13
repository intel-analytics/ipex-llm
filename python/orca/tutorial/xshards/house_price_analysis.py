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
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.transformer import *

import numpy as np

import pandas as pd
df_train = pd.read_csv('/home/ding/data/house_price/train.csv')

init_orca_context(memory="4g")

path = '/home/ding/data/house_price/train.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path, nullValue="NA")

def get_na_sum(df):
    series = df.isnull().sum()
    df2 = pd.DataFrame({'col':series.index, 'total':series.values})
    return df2
data_shard2 = data_shard.transform_shard(get_na_sum)

from functools import reduce
sum_rdd = data_shard2.rdd.mapPartitions(
    lambda iter: [reduce(lambda l1, l2: l1.add(l2), iter)])
sum_shards = SparkXShards(sum_rdd)

missing_data_shards = sum_shards.sort_values(col_names="total", ascending=False)

zip_shards = data_shard.zip(missing_data_shards)

def drop_missing_data(tuple):
    df_train = tuple[0]
    missing_data = tuple[1]
    df2 = df_train.drop((missing_data[missing_data['total'] > 1]['col']), 1)
    return df2

#dealing with missing data
new_shards = zip_shards.transform_shard(drop_missing_data)
def drop_missing_data2(df):
    df2 = df.drop(df.loc[df['Electrical'].isnull()].index)
    return df2

new_shards2 = new_shards.transform_shard(drop_missing_data2)

new_shards3 = new_shards2.transform_shard(get_na_sum)
max_value = new_shards3.max_values('total')

def drop_data(df):
    df = df.drop(df[df_train['Id'] == 1299].index)
    df = df.drop(df[df_train['Id'] == 524].index)
    return df
new_shards3 = new_shards2.transform_shard(drop_data)

#applying log transformation
def generate_new_sale_price(df):
    df['SalePrice'] = np.log(df_train['SalePrice'])
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