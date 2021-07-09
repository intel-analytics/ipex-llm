#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.common.utils import callZooFunc
from pyspark.sql.types import IntegerType, ShortType, LongType, FloatType, DecimalType, \
    DoubleType


def compute(df):
    return callZooFunc("float", "compute", df)


def log_with_clip(df, columns, clip=True):
    return callZooFunc("float", "log", df, columns, clip)


def generate_string_idx(df, columns, freq_limit):
    return callZooFunc("float", "generateStringIdx", df, columns, freq_limit)


def fill_na(df, fill_val, columns):
    return callZooFunc("float", "fillNa", df, fill_val, columns)


def fill_na_int(df, fill_val, columns):
    return callZooFunc("float", "fillNaInt", df, fill_val, columns)


def clip(df, columns, min, max):
    return callZooFunc("float", "clip", df, columns, min, max)


def fill_median(df, columns):
    return callZooFunc("float", "fillMedian", df, columns)


def median(df, columns, relative_error=0.001):
    return callZooFunc("float", "median", df, columns, relative_error)


def cross_columns(df, cross_column_list, bucket_sizes):
    return callZooFunc("float", "crossColumns", df, cross_column_list, bucket_sizes)


def check_col_exists(df, columns):
    df_cols = df.columns
    col_not_exist = list(filter(lambda x: x not in df_cols, columns))
    if len(col_not_exist) > 0:
        raise ValueError(str(col_not_exist) + " do not exist in this Table")


def add_negative_samples(df, item_size, item_col, label_col, neg_num):
    return callZooFunc("float", "addNegSamples", df, item_size, item_col, label_col, neg_num)


def add_hist_seq(df, cols, user_col, sort_col, min_len, max_len):
    return callZooFunc("float", "addHistSeq", df, cols, user_col, sort_col, min_len, max_len)


def add_neg_hist_seq(df, item_size, item_history_col, neg_num):
    return callZooFunc("float", "addNegHisSeq", df, item_size, item_history_col, neg_num)


def mask(df, mask_cols, seq_len):
    return callZooFunc("float", "mask", df, mask_cols, seq_len)


def pad(df, cols, seq_len, mask_cols):
    df = callZooFunc("float", "mask", df, mask_cols, seq_len) if mask_cols else df
    df = callZooFunc("float", "postPad", df, cols, seq_len)
    return df


def check_column_numeric(df, column):
    return df.schema[column].dataType in [IntegerType(), ShortType(),
                                          LongType(), FloatType(),
                                          DecimalType(), DoubleType()]


def ordinal_shuffle_partition(df):
    return callZooFunc("float", "ordinalShufflePartition", df)


def write_parquet(df, path, mode):
    callZooFunc("float", "dfWriteParquet", df, path, mode)
