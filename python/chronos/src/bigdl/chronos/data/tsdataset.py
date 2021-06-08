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

import pandas as pd
import numpy as np
import functools

from zoo.chronos.data.utils.feature import generate_dt_features, generate_global_features
from zoo.chronos.data.utils.impute import impute_timeseries_dataframe
from zoo.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe
from zoo.chronos.data.utils.roll import roll_timeseries_dataframe
from zoo.chronos.data.utils.scale import unscale_timeseries_numpy
from zoo.chronos.data.utils.resample import resample_timeseries_dataframe
from zoo.chronos.data.utils.split import split_timeseries_dataframe

_DEFAULT_ID_COL_NAME = "id"
_DEFAULT_ID_PLACEHOLDER = "0"


class TSDataset:
    def __init__(self, data, **schema):
        '''
        TSDataset is an abstract of time series dataset.
        Cascade call is supported for most of the transform methods.
        '''
        self.df = data
        self.id_col = schema["id_col"]
        self.dt_col = schema["dt_col"]
        self.feature_col = schema["feature_col"]
        self.target_col = schema["target_col"]

        self._check_basic_invariants()

        self._id_list = list(np.unique(self.df[self.id_col]))
        self._is_pd_datetime = pd.api.types.is_datetime64_any_dtype(self.df[self.dt_col].dtypes)

        self.numpy_x = None
        self.numpy_y = None
        self.roll_feature = None
        self.roll_target = None
        self.scaler = None
        self.id_sensitive = None

    @staticmethod
    def from_pandas(df,
                    dt_col,
                    target_col,
                    id_col=None,
                    extra_feature_col=None,
                    with_split=False,
                    val_ratio=0,
                    test_ratio=0.1,
                    largest_look_back=0,
                    largest_horizon=1):
        '''
        Initialize a tsdataset(s) from pandas dataframe.

        :param df: a pandas dataframe for your raw time series data.
        :param dt_col: a str indicates the col name of datetime
               column in the input data frame.
        :param target_col: a str or list indicates the col name of target column
               in the input data frame.
        :param id_col: (optional) a str indicates the col name of dataframe id.
        :param extra_feature_col: (optional) a str or list indicates the col name
               of extra feature columns that needs to predict the target column.
        :param with_split: (optional) bool, state if we need to split the dataframe
               to train, validation and test set. The value defaults to False.
        :param val_ratio: (optional) float, validation ratio. Only effective when
               with_split is set to True. The value defaults to 0.
        :param test_ratio: (optional) float, test ratio. Only effective when with_split
               is set to True. The value defaults to 0.1.
        :param largest_look_back: (optional) int, the largest length to look back.
               Only effective when with_split is set to True. The value defaults to 0.
        :param largest_horizon: (optional) int, the largest num of steps to look
               forward. Only effective when with_split is set to True. The value defaults
               to 1.

        :return: a TSDataset instance when with_split is set to False,
                 three TSDataset instances when with_split is set to True.


        Here is a df example:

        id        datetime      value   "extra feature 1"   "extra feature 2"
        00        2019-01-01    1.9     1                   2
        01        2019-01-01    2.3     0                   9
        00        2019-01-02    2.4     3                   4
        01        2019-01-02    2.6     0                   2

        Create a tsdataset instance by:

        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",
        >>>                                   target_col="value", id_col="id",
        >>>                                   extra_feature_col=["extra feature 1",
        >>>                                                      "extra feature 2"])
        '''

        _check_type(df, "df", pd.DataFrame)

        tsdataset_df = df.copy(deep=True)
        target_col = _to_list(target_col, name="target_col")
        feature_col = _to_list(extra_feature_col, name="extra_feature_col")

        if id_col is None:
            tsdataset_df[_DEFAULT_ID_COL_NAME] = _DEFAULT_ID_PLACEHOLDER
            id_col = _DEFAULT_ID_COL_NAME

        if with_split:
            tsdataset_dfs = split_timeseries_dataframe(df=tsdataset_df,
                                                       id_col=id_col,
                                                       val_ratio=val_ratio,
                                                       test_ratio=test_ratio,
                                                       look_back=largest_look_back,
                                                       horizon=largest_horizon)
            return [TSDataset(data=tsdataset_dfs[i],
                              id_col=id_col,
                              dt_col=dt_col,
                              target_col=target_col,
                              feature_col=feature_col) for i in range(3)]

        return TSDataset(data=tsdataset_df,
                         id_col=id_col,
                         dt_col=dt_col,
                         target_col=target_col,
                         feature_col=feature_col)

    def impute(self, mode="last", const_num=0):
        '''
        Impute the tsdataset by imputing each univariate time series
        distinguished by id_col and feature_col

        :param mode: imputation mode, select from "last", "const" or "linear".
               "last": impute by propagating the last non N/A number to its following N/A.
                       if there is no non N/A number ahead, 0 is filled instead.
               "const": impute by a const value input by user.
               "linear": impute by linear interpolation.
        :param const_num:  indicate the const number to fill, which only effective when mode
               is set to "const".

        :return: the tsdataset instance.

        Note: It is preferred that `impute` is called after `resample` while before
              `roll` if needed.
        '''
        df_list = [impute_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name],
                                               dt_col=self.dt_col,
                                               mode=mode,
                                               const_num=const_num)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        return self

    def deduplicate(self):
        '''
        Remove those duplicated rows which has exactly the same values in each feature_col for
        each multivariate timeseries distinguished by id_col

        :return: the tsdataset instance.

        Note: It is preferred that `deduplicate` is called before all other operations if needed.
        '''
        df_list = [deduplicate_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name],
                                                    dt_col=self.dt_col)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        return self

    def resample(self, interval, start_time, end_time, merge_mode="mean"):
        '''
        resample on an new interval for each univariate time series distinguished
        by id_col and feature_col.

        :param interval: pandas offset aliases, indicating time interval of the output dataframe.
        :param start_time: start time of the output dataframe.
        :param end_time: end time of the output dataframe.
        :param merge_mode: if current interval is smaller than output interval,
            we need to merge the values in a mode. "max", "min", "mean"
            or "sum" are supported for now.
        :return: the tsdataset instance.

        Note: It if preferred to call `impute` right after `resample`.
        '''
        df_list = []
        for id_name in self._id_list:
            df_id = resample_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name]
                                                  .drop(self.id_col, axis=1),
                                                  dt_col=self.dt_col,
                                                  interval=interval,
                                                  start_time=start_time,
                                                  end_time=end_time,
                                                  merge_mode=merge_mode)
            df_id[self.id_col] = id_name
            df_list.append(df_id.copy())
        self.df = pd.concat(df_list)
        return self

    def gen_dt_feature(self):
        '''
        Generate datetime feature for each row. Currently we generate following features:
        "MINUTE", "DAY", "DAYOFYEAR", "HOUR", "WEEKDAY", "WEEKOFYEAR", "MONTH", "IS_AWAKE",
        "IS_BUSY_HOURS", "IS_WEEKEND"

        :return: the tsdataset instance.

        Note: it should be called before scale if needed.
        '''
        df_list = [generate_dt_features(input_df=self.df[self.df[self.id_col] == id_name],
                                        dt_col=self.dt_col)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        from zoo.chronos.data.utils.feature import TIME_FEATURE, \
            ADDITIONAL_TIME_FEATURE_HOUR, ADDITIONAL_TIME_FEATURE_WEEKDAY
        increased_attrbutes = list(TIME_FEATURE) +\
            list(ADDITIONAL_TIME_FEATURE_HOUR) +\
            list(ADDITIONAL_TIME_FEATURE_WEEKDAY)
        self.feature_col += [attr + "({})".format(self.dt_col) for attr in increased_attrbutes]
        return self

    def gen_global_feature(self, settings="comprehensive", full_settings=None):
        '''
        Generate per-time-series feature for each time series.
        This method will be implemented by tsfresh.

        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"
               "minimal" and "efficient". If a dict is set then it should follow the instruction
               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".
        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in
               tsfresh. The value is defaulted to None.

        :return: the tsdataset instance.

        '''
        if full_settings is not None:
            self.df = generate_global_features(input_df=self.df,
                                               column_id=self.id_col,
                                               column_sort=self.dt_col,
                                               kind_to_fc_parameters=full_settings)
            return self

        from tsfresh.feature_extraction import ComprehensiveFCParameters,\
            MinimalFCParameters, EfficientFCParameters
        default_params = {"comprehensive": ComprehensiveFCParameters(),
                          "minimal": MinimalFCParameters(),
                          "efficient": EfficientFCParameters()}

        if isinstance(settings, str):
            assert settings in ["comprehensive", "minimal", "efficient"], \
                f"settings str should be one of \"comprehensive\", \"minimal\", \"efficient\"\
                    , but found {settings}."
            default_fc_parameters = default_params[settings]
        else:
            default_fc_parameters = settings

        self.df,\
            addtional_feature =\
            generate_global_features(input_df=self.df,
                                     column_id=self.id_col,
                                     column_sort=self.dt_col,
                                     default_fc_parameters=default_fc_parameters)

        self.feature_col += addtional_feature

        return self

    def roll(self,
             lookback,
             horizon,
             feature_col=None,
             target_col=None,
             id_sensitive=False):
        '''
        Sampling by rolling for machine learning/deep learning models.

        :param lookback: int, lookback value
        :param horizon: int or list,
               if `horizon` is an int, we will sample `horizon` step
               continuously after the forecasting point.
               if `horizon` is an list, we will sample discretely according
               to the input list.
               specially, when `horizon` is set to 0, ground truth will be generated as None.
        :param feature_col: str or list, indicate the feature col name. Default to None,
               where we will take all avaliable feature in rolling.
        :param target_col: str or list, indicate the target col name. Default to None,
               where we will take all target in rolling. it should be a subset of target_col
               you used to initialized the tsdataset.
        :param id_sensitive: bool,
               if `id_sensitive` is False, we will rolling on each id's sub dataframe
               and fuse the sampings.
               The shape of rolling will be
               x: (num_sample, lookback, num_feature_col)
               y: (num_sample, horizon, num_target_col)
               where num_sample is the summation of sample number of each dataframe

               if `id_sensitive` is True, we will rolling on the wide dataframe whose
               columns are cartesian product of id_col and feature_col
               The shape of rolling will be
               x: (num_sample, lookback, num_feature_col)
               y: (num_sample, horizon, num_target_col)
               where num_sample is the sample number of the wide dataframe,
               num_feature_col is the product of the number of id and the number of feature_col,
               num_target_col is the product of the number of id and the number of target_col.

        :return: the tsdataset instance.
        '''
        feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
            else self.feature_col
        target_col = _to_list(target_col, "target_col") if target_col is not None \
            else self.target_col

        num_id = len(self._id_list)
        num_feature_col = len(self.feature_col)
        num_target_col = len(self.target_col)
        self.roll_feature = feature_col
        self.roll_target = target_col
        self.id_sensitive = id_sensitive

        # get rolling result for each sub dataframe
        rolling_result = [roll_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name],
                                                    lookback=lookback,
                                                    horizon=horizon,
                                                    feature_col=feature_col,
                                                    target_col=target_col)
                          for id_name in self._id_list]

        # concat the result on required axis
        concat_axis = 2 if id_sensitive else 0
        self.numpy_x = np.concatenate([rolling_result[i][0]
                                       for i in range(num_id)],
                                      axis=concat_axis)
        if horizon != 0:
            self.numpy_y = np.concatenate([rolling_result[i][1]
                                           for i in range(num_id)],
                                          axis=concat_axis)
        else:
            self.numpy_y = None

        # target first
        if self.id_sensitive:
            feature_start_idx = num_target_col*num_id
            reindex_list = [list(range(i*num_target_col, (i+1)*num_target_col)) +
                            list(range(feature_start_idx+i*num_feature_col,
                                       feature_start_idx+(i+1)*num_feature_col))
                            for i in range(num_id)]
            reindex_list = functools.reduce(lambda a, b: a+b, reindex_list)
            self.numpy_x = self.numpy_x[:, :, reindex_list]

        return self

    def to_numpy(self):
        '''
        export rolling result in form of a tuple of numpy ndarray (x, y)

        :return: a 2-dim tuple. each item is a 3d numpy ndarray
        '''
        if self.numpy_x is None:
            raise RuntimeError("Please call \"roll\" method\
                    before transform a TSDataset to numpy ndarray!")
        return self.numpy_x, self.numpy_y

    def to_pandas(self):
        '''
        export the pandas dataframe

        :return: the internal dataframe.
        '''
        return self.df.copy()

    def scale(self, scaler, fit=True):
        '''
        scale the time series dataset's feature column and target column.
        :param scaler: sklearn scaler instance, StandardScaler, MaxAbsScaler,
               MinMaxScaler and RobustScaler are supported.
        :param fit: if we need to fit the scaler. Typically, the value should
               be set to True for training set, while False for validation and
               test set. The value is defaulted to True.
        :return: the tsdataset instance.
        '''
        if fit:
            self.df[self.target_col + self.feature_col] = \
                scaler.fit_transform(self.df[self.target_col + self.feature_col])
        else:
            self.df[self.target_col + self.feature_col] = \
                scaler.transform(self.df[self.target_col + self.feature_col])
        self.scaler = scaler
        return self

    def unscale(self):
        '''
        unscale the time series dataset's feature column and target column.

        :return: the tsdataset instance.
        '''
        self.df[self.target_col + self.feature_col] = \
            self.scaler.inverse_transform(self.df[self.target_col + self.feature_col])
        return self

    def _unscale_numpy(self, data):
        '''
        unscale the time series forecastor's numpy prediction result/ground truth.
        :param data: a numpy ndarray with 3 dim whose shape should be exactly the
               same with self.numpy_y.
        :return: the unscaled numpy ndarray
        '''
        num_roll_target = len(self.roll_target)
        repeat_factor = len(self._id_list) if self.id_sensitive else 1
        scaler_index = [self.target_col.index(self.roll_target[i])
                        for i in range(num_roll_target)] * repeat_factor
        return unscale_timeseries_numpy(data, self.scaler, scaler_index)

    def _check_basic_invariants(self):
        '''
        This function contains a bunch of assertions to make sure strict rules(the invariants)
        for the internal dataframe(self.df) must stands. If not, clear and user-friendly error
        or warning message should be provided to the users.
        This function will be called after each method(e.g. impute, deduplicate ...)
        '''
        # check type
        _check_type(self.df, "df", pd.DataFrame)
        _check_type(self.id_col, "id_col", str)
        _check_type(self.dt_col, "dt_col", str)
        _check_type(self.target_col, "target_col", list)
        _check_type(self.feature_col, "feature_col", list)

        # check valid name
        _check_col_within(self.df, self.id_col)
        _check_col_within(self.df, self.dt_col)
        for target_col_name in self.target_col:
            _check_col_within(self.df, target_col_name)
        for feature_col_name in self.feature_col:
            _check_col_within(self.df, feature_col_name)

        # check no n/a in critical col
        _check_col_no_na(self.df, self.dt_col)
        _check_col_no_na(self.df, self.id_col)


def _to_list(item, name, expect_type=str):
    if isinstance(item, list):
        return item
    if item is None:
        return []
    _check_type(item, name, expect_type)
    return [item]


def _check_type(item, name, expect_type):
    assert isinstance(item, expect_type),\
        f"a {str(expect_type)} is expected for {name} but found {type(item)}"


def _check_col_within(df, col_name):
    assert col_name in df.columns,\
        f"{col_name} is expected in dataframe while not found"


def _check_col_no_na(df, col_name):
    _check_col_within(df, col_name)
    assert df[col_name].isna().sum() == 0,\
        f"{col_name} column should not have N/A."
