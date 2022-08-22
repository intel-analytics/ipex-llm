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
# https://www.kaggle.com/code/prashant111/comprehensive-data-analysis-with-pandas/notebook

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
import bigdl.orca.data.pandas

init_orca_context(cluster_mode="local", cores=4, memory="3g")
OrcaContext.pandas_read_backend = "pandas"

# read
file_path = "train.csv"
data_shard = bigdl.orca.data.pandas.read_csv(file_path)


# fillna
def fill_na(df):
    df = df.fillna(method='pad')
    return df
data_shard = data_shard.transform_shard(fill_na)


# apply
def apply_func(df):
    dic = {'A': 'Beijing', 'B': 'Shanghai', 'C': 'Guangzhou'}
    df['City_Category'] = df['City_Category'].apply(lambda x: dic[x])
    return df
data_shard = data_shard.transform_shard(apply_func)


# map
def map_func(df):
    df = df['City_Category'].map({'A': 'Beijing', 'B': 'Shanghai', 'C': 'Guangzhou'})
    return df
data_shard = data_shard.transform_shard(map_func)

stop_orca_context()
