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
# https://www.kaggle.com/code/chuanguy/titanic-data-processing-with-python-0-813/notebook

from bigdl.orca import init_orca_context, stop_orca_context
import bigdl.orca.data.pandas

init_orca_context(cluster_mode="local", cores=4, memory="3g")

file_path = "titanic.csv"
data_shard = bigdl.orca.data.pandas.read_csv(file_path)


# drop
def drop_passenger(df):
    df = df.drop(['PassengerId'], axis=1)
    return df
data_shard = data_shard.transform_shard(drop_passenger)


# fillna, apply, replace, map
def process_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('X')
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
    df['Cabin'] = df['Cabin'].replace(['A', 'D', 'E', 'T'], 'M')
    df['Cabin'] = df['Cabin'].replace(['B', 'C'], 'H')
    df['Cabin'] = df['Cabin'].replace(['F', 'G'], 'L')
    df['Cabin'] = df['Cabin'].map({'X': 0, 'L': 1, 'M': 2, 'H': 3})
    df['Cabin'] = df['Cabin'].astype(int)
    return df
data_shard = data_shard.transform_shard(process_cabin)


# astype, loc
def encode(data):
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
    data['Pclass'] = data['Pclass'].map({1: 3, 2: 2, 3: 1}).astype(int)
    data.loc[data['Sex'] == 0, 'SexByPclass'] = data.loc[data['Sex'] == 0, 'Pclass']
    data.loc[data['Sex'] == 1, 'SexByPclass'] = data.loc[data['Sex'] == 1, 'Pclass'] + 3
    return data
data_shard = data_shard.transform_shard(encode)

# save
data_shard.save_pickle('./result')

stop_orca_context()
