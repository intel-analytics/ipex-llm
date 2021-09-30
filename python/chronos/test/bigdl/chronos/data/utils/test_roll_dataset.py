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

import pytest
import numpy as np
import pandas as pd
import random
from bigdl.chronos.data import TSDataset
from bigdl.chronos.data.utils.roll_dataset import RollDataset


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def get_multi_id_ts_df():
    sample_num = 100
    train_df = pd.DataFrame({"value": np.random.randn(sample_num),
                             "id": np.array(['00']*50 + ['01']*50),
                             "extra feature": np.random.randn(sample_num)})
    train_df["datetime"] = pd.date_range('1/1/2019', periods=sample_num)
    train_df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    return train_df


class TestRollDataset:

    @staticmethod
    def assert_equal_with_tsdataset(df,
                                    horizon,
                                    lookback,
                                    feature_num=1,
                                    ):
        # get results rolled by tsdata.roll
        extra_feature_col = None if feature_num == 0 else ["extra feature"]
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=extra_feature_col, id_col="id")
        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()

        # get results rolled by RollDataset
        roll_dataset = RollDataset(df=df,
                                   lookback=lookback,
                                   horizon=horizon,
                                   feature_col=tsdata.feature_col,
                                   target_col=tsdata.target_col,
                                   id_col=tsdata.id_col)

        assert len(roll_dataset) == len(x)
        for i in range(len(x)):
            if horizon != 0:
                # for train and y is not None.
                xi, yi = x[i], y[i]
                roll_dataset_xi, roll_dataset_yi = roll_dataset[i]
                np.testing.assert_array_almost_equal(xi, roll_dataset_xi.detach().numpy())
                np.testing.assert_array_almost_equal(yi, roll_dataset_yi.detach().numpy())
            else:
                # for test, y is None.
                xi = x[i]
                roll_dataset_xi = roll_dataset[i]
                np.testing.assert_array_almost_equal(xi, roll_dataset_xi.detach().numpy())

    @staticmethod
    def combination_tests_for_df(df):
        lookback = random.randint(1, 20)

        horizon_tests = [
            random.randint(1, 10),  # train & horizon is int
            [1, 4, 16],  # train & horizon is list of ints
            0,  # test
        ]
        # todo: add tests for multiple targets and feature_num > 1
        feature_num_tests = [0, 1]

        for horizon in horizon_tests:
            for feature_num in feature_num_tests:
                TestRollDataset.assert_equal_with_tsdataset(df=df,
                                                            horizon=horizon,
                                                            lookback=lookback,
                                                            feature_num=feature_num)

    def test_single_id(self):
        df = get_ts_df()
        TestRollDataset.combination_tests_for_df(df)

    def test_multi_id(self):
        df = get_multi_id_ts_df()
        TestRollDataset.combination_tests_for_df(df)

    def test_df_nan(self):
        df = get_ts_df()
        df["value"][0] = np.nan
        with pytest.raises(AssertionError):
            RollDataset(df=df,
                        lookback=2,
                        horizon=1,
                        feature_col=["extra feature"],
                        target_col=["value"],
                        id_col="id")
