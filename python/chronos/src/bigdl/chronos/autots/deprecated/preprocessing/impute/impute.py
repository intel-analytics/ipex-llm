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

from bigdl.chronos.autots.deprecated.preprocessing.impute.abstract import BaseImputation
from bigdl.chronos.autots.deprecated.preprocessing.impute.LastFill import LastFill
from bigdl.chronos.utils import deprecated


@deprecated('Please use `bigdl.chronos.data.TSDataset` instead.')
class LastFillImpute(BaseImputation):
    """
    LastFill imputation
    """

    def __init__(self):
        self.imputer = LastFill()

    def impute(self, input_df):
        from bigdl.nano.utils.log4Error import invalidInputError
        invalidInputError(self.imputer is not None, "imputer cannot be empty")
        df = self.imputer.impute(input_df)
        return df

    def restore(self, **config):
        self.imputer = LastFill()


@deprecated('Please use `bigdl.chronos.data.TSDataset` instead.')
class FillZeroImpute(BaseImputation):
    """
    FillZero imputation
    """

    def impute(self, input_df):
        df = input_df.fillna(0)
        return df


@deprecated('Please use `bigdl.chronos.data.TSDataset` instead.')
class TimeMergeImputor(BaseImputation):
    """
    Merge rows whose timestamp are seconds apart
    """

    def __init__(self, time_interval, timestamp_column_name, mode=""):
        self.time_interval = time_interval
        self.timestamp_column_name = timestamp_column_name
        self.mode = mode

    def impute(self, input_df):
        import pyspark.sql.functions as f

        ori_column_name = self.timestamp_column_name + "_ori"
        df = input_df.withColumnRenamed(
            self.timestamp_column_name, ori_column_name)
        merged_df = df.withColumn(
            "add_seconds",
            (f.round(
                f.second(ori_column_name) /
                self.time_interval) *
                self.time_interval) -
            f.second(ori_column_name)) .withColumn(
            self.timestamp_column_name,
            f.from_unixtime(
                f.unix_timestamp(ori_column_name) +
                f.col("add_seconds"))).drop("add_seconds")
        if self.mode == "max":
            merged_df = merged_df.groupby(self.timestamp_column_name).max()
        elif self.mode == "min":
            merged_df = merged_df.groupby(self.timestamp_column_name).min()
        elif self.mode == "mean":
            merged_df = merged_df.groupby(self.timestamp_column_name).mean()
        elif self.mode == "sum":
            merged_df = merged_df.groupby(self.timestamp_column_name).sum()
        elif self.mode == "":
            merged_df
        else:
            from bigdl.nano.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "Currently only support max/min/mean/sum mode")

        return merged_df
