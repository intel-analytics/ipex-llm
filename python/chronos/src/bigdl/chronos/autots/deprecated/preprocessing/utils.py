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


def train_val_test_split(df,
                         val_ratio=0,
                         test_ratio=0.1,
                         look_back=0,
                         horizon=1):
    """
    split input dataframe into train_df, val_df and test_df according to split ratio.
    The dataframe is splitted in its originally order in timeline.
    e.g. |......... train_df(80%) ........ | ... val_df(10%) ...| ...test_df(10%)...|
    :param df: dataframe to be splitted
    :param val_ratio: validation ratio
    :param test_ratio: test ratio
    :param look_back: the length to look back
    :param horizon: num of steps to look forward
    :return:
    """
    # suitable to nyc taxi dataset.
    import pandas as pd

    total_num = df.index.size
    test_num = int(total_num * test_ratio)
    val_num = int(total_num * val_ratio)

    test_split_index = test_num + look_back + horizon - 1
    val_split_index = test_split_index + val_num

    train_df = df.iloc[:-(test_num + val_num)]
    val_df = df.iloc[-val_split_index: -test_num]
    test_df = df.iloc[-test_split_index:]

    if not pd.api.types.is_datetime64_any_dtype(df.index.dtype):
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df
