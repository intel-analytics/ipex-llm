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


path = 'answer_correctness/train.csv'
used_data_types_list = [
    'timestamp',
    'user_id',
    'content_id',
    'answered_correctly',
    'prior_question_elapsed_time',
    'prior_question_had_explanation'
]
data_shard = bigdl.orca.data.pandas.read_csv(path,
                                             usecols=used_data_types_list,
                                             index_col=0)


def get_feature(df):
    feature_df = df.iloc[:int(9/10 * len(df))]
    return feature_df
feature_shard = data_shard.transform_shard(get_feature)


def get_train_questions_only(df):
    train_questions_only_df = df[df['answered_correctly'] != -1]
    return train_questions_only_df
train_questions_only_shard = feature_shard.transform_shard(get_train_questions_only)

train_questions_only_shard = \
    train_questions_only_shard.group_by(columns='user_id', agg={"answered_correctly": ['mean',
                                                                                       'count',
                                                                                       'stddev',
                                                                                       'skewness']
                                                                }, join=True)
target = 'answered_correctly'


def filter_non_target(df):
    train_df = df[df[target] != -1]
    return train_df
train_shard = train_questions_only_shard.transform_shard(filter_non_target)


def fill_na(df, val):
    train_df = df.fillna(value=val)
    return train_df
train_shard = train_shard.transform_shard(fill_na, 0.5)
