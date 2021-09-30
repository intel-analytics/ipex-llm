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


def get_drop_dates_and_len(df, allow_missing_num=3):
    """
    Find missing values and get records to drop
    """
    missing_num = df.total.isnull().astype(int)\
        .groupby(df.total.notnull().astype(int).cumsum()).sum()
    drop_missing_num = missing_num[missing_num > allow_missing_num]
    drop_datetimes = df.iloc[drop_missing_num.index].index
    drop_len = drop_missing_num.values
    return drop_datetimes, drop_len


def rm_missing_weeks(start_dts, missing_lens, df):
    """
    Drop weeks that contains more than 3 consecutive missing values.
    If consecutive missing values across weeks, we remove all the weeks.
    """
    for start_time, l in zip(start_dts, missing_lens):
        start = start_time - pd.Timedelta(days=start_time.dayofweek)
        start = start.replace(hour=0, minute=0, second=0)
        start_week_end = start + pd.Timedelta(days=6)
        start_week_end = start_week_end.replace(hour=22, minute=0, second=0)

        end_time = start_time + l*pd.Timedelta(hours=2)
        if start_week_end < end_time:
            end = end_time + pd.Timedelta(days=6-end_time.dayofweek)
            end = end.replace(hour=22, minute=0, second=0)
        else:
            end = start_week_end
        df = df.drop(df[start:end].index)
    return df

import pandas as pd
from bigdl.orca import init_orca_context, stop_orca_context

raw_df = pd.read_csv("data/data.csv")

df = pd.DataFrame(pd.to_datetime(raw_df.StartTime))
df['AvgRate'] = \
    raw_df.AvgRate.apply(lambda x: float(x[:-4]) if x.endswith("Mbps") else float(x[:-4]) * 1000)
df["total"] = raw_df["total"]
df.set_index("StartTime", inplace=True)
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='2H')
df = df.reindex(full_idx)
drop_dts, drop_len = get_drop_dates_and_len(df)
df = rm_missing_weeks(drop_dts, drop_len, df)
df.ffill(inplace=True)
df.index.name = "datetime"
df = df.reset_index()

init_orca_context(cores=4, memory="4g", init_ray_on_spark=True)

from bigdl.chronos.autots.deprecated.forecast import AutoTSTrainer
from bigdl.chronos.autots.deprecated.config.recipe import *

trainer = AutoTSTrainer(dt_col="datetime",
                        target_col=["AvgRate", "total"],
                        horizon=1,
                        extra_features_col=None)

look_back = (36, 84)
from bigdl.chronos.autots.deprecated.preprocessing.utils import train_val_test_split

train_df, val_df, test_df = train_val_test_split(df,
                                                 val_ratio=0.1,
                                                 test_ratio=0.1,
                                                 look_back=look_back[0])

ts_pipeline = trainer.fit(train_df, val_df,
                          recipe=MTNetGridRandomRecipe(
                              num_rand_samples=1,
                              time_step=[12],
                              long_num=[6],
                              ar_size=[6],
                              cnn_height=[4],
                              cnn_hid_size=[32],
                              training_iteration=1,
                              epochs=20,
                              batch_size=[1024]),
                          metric="mse")

pred_df = ts_pipeline.predict(test_df)

mse, smape = ts_pipeline.evaluate(test_df, metrics=["mse", "smape"])
print("Evaluate: the mean square error is", mse)
print("Evaluate: the smape value is", smape)

stop_orca_context()
